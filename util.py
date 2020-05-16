import random
import numpy as np
import json
import constant
from CCG.constant import NER2ID,TYPE2ID

def get_word(tokens, word2idx_dict, pad=None):
    res = []
    for token in tokens:
        i = 1
        for each in (token, token.lower(), token.capitalize(), token.upper()):
            if each in word2idx_dict:
                i = word2idx_dict[each]
                break
        res.append(i)
    if pad is not None:
        res = res[:pad]
        res += [0 for _ in range(pad - len(res))]
    return res

def get_lfs(config, word2idx_dict, filt=None):
    with open(config.pattern_file, "r") as fh:
        logic_forms = json.load(fh)
    raw_keywords,rels, lfs,keywords,keywords_rels = [], [],[],[],[]
    for cnt,logic_form in enumerate(logic_forms):
        rel, lf,pat,keyword_s = logic_form
        rel_id = constant.LABEL_TO_ID[rel]
        if filt is not None:
            if rel_id in filt:
                continue
        rel = [0. for _ in range(config.num_class)]
        rel[rel_id] = 1.
        rels.append(rel)
        lfs.append(lf)

        if len(pat)>0:
            select_pat = pat[0]
            if select_pat not in keywords:
                keywords.append(select_pat)
                keywords_rels.append(rel)

        for word in keyword_s:
            if word not in raw_keywords:
                raw_keywords.append(word)

    num_lfs = len(lfs)
    rels = np.asarray(rels, dtype=np.float32)
    weights = np.ones([num_lfs], dtype=np.float32)
    return {"rels": rels, "lfs": lfs, "weights": weights,'keywords':keywords,'keywords_rels':keywords_rels,'raw_keywords':raw_keywords}


def get_feeddict(model, batch, pretrain_batch,is_train=True):
    if is_train:
        return {model.hard_match_func_idx: batch["hard_func"], model.sent_word: batch['sent'],model.phrases_input:batch['phrases_input'],
                model.pretrain_sents:pretrain_batch['sents'], model.pretrain_pats:pretrain_batch['pats'], model.pretrain_labels:pretrain_batch['labels'],
                model.rel: batch["rel"],  model.is_train: is_train}
    else:
        return {model.sent_word: batch['sent'],model.phrases_input:batch['phrases_input'],
                model.pretrain_sents: pretrain_batch['sents'], model.pretrain_pats: pretrain_batch['pats'],model.pretrain_labels: pretrain_batch['labels'],
                model.rel: batch["rel"], model.is_train: is_train}


def get_batch(config, data, word2idx_dict, rel_dict=None, shuffle=True, pseudo=False):
    if shuffle:
        random.shuffle(data)
    batch_size = config.pseudo_size if pseudo else config.batch_size
    length = config.length
    for i in range(len(data) // batch_size):
        batch = data[i * batch_size: (i + 1) * batch_size]
        ner = np.asarray(list(map(lambda x: get_word(x["phrase"].ner,NER2ID, pad=length), batch)), dtype=np.int32)
        subj_obj = np.asarray(list(map(lambda x: [x['phrase'].subj_posi,x['phrase'].obj_posi,TYPE2ID[x['phrase'].subj.split('-')[-1]],TYPE2ID[x['phrase'].obj.split('-')[-1]]], batch)), dtype=np.int32)
        sent = np.asarray(list(map(lambda x: get_word(x["phrase"].token, word2idx_dict, pad=length), batch)), dtype=np.int32)
        rel = np.asarray(list(map(lambda x: [1.0 if i == x["rel"] else 0. for i in range(config.num_class)], batch)), dtype=np.float32)
        phrases_input = np.concatenate((sent,ner,subj_obj),axis=1)
        hard_func = []
        if 'logic_form' in batch[0]:
            for one_data in batch:
                hard_func.append(one_data['logic_form'])
        hard_func = np.asarray(hard_func,np.int32)

        yield {"sent": sent,  "rel": rel,'phrases_input':phrases_input,'hard_func':hard_func,'raw_sent':[x['phrase'].sentence for x in batch]}


def merge_batch(batch1, batch2):
    batch = {}
    key_concat = ['sent','rel','hard_func','phrases_input']
    for key in key_concat:
        val1 = batch1[key]
        val2 = batch2[key]
        val = np.concatenate([val1, val2], axis=0)
        batch[key] = val
    return batch

def get_pretrain_batch(config,pretrain_data,word2idx_dict,shuffle=True,pretrain=True,all=False):
    pats, sents, labels = pretrain_data
    order = list(np.arange(len(sents)))
    if shuffle:
        random.shuffle(order)
    batch_size = config.pretrain_size if pretrain else config.pretrain_size_together
    if all:
        batch_size = len(sents)
    length = config.length
    for i in range(len(sents) // batch_size):
        batch_order = order[i * batch_size: (i + 1) * batch_size]
        batch_labels = np.take(labels,batch_order,axis=0)
        batch_pats = [pats[i] for i in batch_order]
        batch_sents = [sents[i] for i in batch_order]
        batch_pats = np.array([get_word(pat,word2idx_dict,10) for pat in batch_pats],np.int32)
        batch_sents = np.array([get_word(sent,word2idx_dict,length) for sent in batch_sents],np.int32)
        yield {'sents':batch_sents,'pats':batch_pats,'labels':batch_labels}
