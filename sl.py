import tensorflow as tf
import numpy as np
import sys
from util import get_batch, get_feeddict, get_lfs, merge_batch,get_pretrain_batch
from tqdm import tqdm
import random
from models.pat_match import Pat_Match
from models.soft_match import Soft_Match
from main import log
import pickle
import os
import json
import constant
def f_score(predict,golden,mode='f'):
    assert len(predict)==len(golden)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(predict)):
        if predict[i]==golden[i] and predict[i] != 0:
            TP+=1
        elif predict[i]!=golden[i]:
            if predict[i]==0:
                FN+=1
            elif golden[i]==0:
                FP+=1
            else:
                FN+=1
                FP+=1
        else:
            TN+=1
    try:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F = 2*P*R/(P+R)
    except:
        P=R=F=0

    if mode=='f':
        return P,R,F
    else:
        return TP,FN,FP,TN


def pseudo_labeling(config, data):
    tf.reset_default_graph()
    word2idx_dict, fixed_emb, traiable_emb, train_data, dev_data, test_data,pretrain_data,pretrain_data2 = data

    pretrain_test_data = (pretrain_data[0][:config.pretrain_test_size],pretrain_data[1][:config.pretrain_test_size],pretrain_data[2][:config.pretrain_test_size,:])
    pretrain_data = (pretrain_data[0][config.pretrain_test_size:config.pretrain_test_size+config.pretrain_train_size],pretrain_data[1][config.pretrain_test_size:config.pretrain_test_size+config.pretrain_train_size],pretrain_data[2][config.pretrain_test_size:config.pretrain_test_size+config.pretrain_train_size,:])

    lfs = get_lfs(config, word2idx_dict)
    identifier = "_{}".format(config.tag)

    with tf.variable_scope("models", reuse=tf.AUTO_REUSE):
        regex = Pat_Match(config)
        match = Soft_Match(config,lfs['lfs'],np.array(lfs['rels'],np.float32),lfs['keywords'],lfs['keywords_rels'], lfs['raw_keywords'],mat=((fixed_emb, traiable_emb, )), word2idx_dict=word2idx_dict, pseudo=True)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with open('./CCG/exp2pat.json','r') as f:
        exp2pat = json.load(f)
    exp2pat = {int(key):val for key,val in exp2pat.items()}
    lab_d = []
    unlab_d = []

    tacred_labeled = []
    tacred_unlabeled = []
    labeled_data = []
    unlabeled_data = []
    idxx = -1

    idx2rel = {val:key for key,val in constant.LABEL_TO_ID.items()}

    for x in tqdm(train_data):
        idxx+=1
        batch = [x["phrase"]]
        res, pred = regex.match(batch)
        lfs["weights"] += res[0]
        new_dict = {}
        if np.amax(res) > 0:

            x["rel"] = pred.tolist()[0]
            x["logic_form"] = np.argmax(res, axis=1).tolist()[0]
            new_dict['tokens'] = x['phrase'].token
            new_dict['start'] = min(x['phrase'].subj_posi,x['phrase'].obj_posi)+1
            new_dict['end'] = max(x['phrase'].subj_posi,x['phrase'].obj_posi)-1
            new_dict['rel'] = pred.tolist()[0]
            try:
                new_dict['pat'] = exp2pat[np.argmax(res, axis=1).tolist()[0]]
                lab_d.append(new_dict)
            except:
                new_dict['pat'] = -1
                unlab_d.append(new_dict)
            tacred_labeled.append((idxx,idx2rel[x['rel']]))
            labeled_data.append(x)
        else:
            tacred_unlabeled.append(idxx)
            new_dict['tokens'] = x['phrase'].token
            new_dict['start'] = min(x['phrase'].subj_posi,x['phrase'].obj_posi)+1
            new_dict['end'] = max(x['phrase'].subj_posi,x['phrase'].obj_posi)-1
            new_dict['rel'] = pred.tolist()[0]
            new_dict['pat']=-1
            x["rel"] = 0
            unlab_d.append(new_dict)
            unlabeled_data.append(x)

    new_weight = np.array([elem for i, elem in enumerate(list(lfs['weights'])) if i in exp2pat],np.float32)
    new_weight = new_weight/np.sum(new_weight)
    lfs["weights"] = lfs["weights"] / np.sum(lfs["weights"])


    random.shuffle(unlabeled_data)

    print('unlabdel data:',str(len(unlabeled_data)),'labeled data:',str(len(labeled_data)))

    sess=tf.Session(config=sess_config)

    saver = tf.train.Saver()
    try:
        saver.restore(sess,'dump_models/explanation.ckpt')
    except:
        sess.run(tf.global_variables_initializer())

    print('---train-----')
    for epoch in range(config.pretrain_epoch):
        loss_list,pretrain_loss_lis,sim_loss_lis = [],[],[]
        for batch in get_pretrain_batch(config, pretrain_data, word2idx_dict):
            pretrain_loss_prt,sim_loss_prt,loss,_ = sess.run([match.pretrain_loss,match.sim_loss,match.pretrain_loss_v2,match.pre_train_op],feed_dict={match.pretrain_sents: batch['sents'], match.pretrain_pats: batch['pats'],match.pretrain_labels: batch['labels'],match.is_train:True})
            loss_list.append(loss)
            pretrain_loss_lis.append(pretrain_loss_prt)
            sim_loss_lis.append(sim_loss_prt)
        print("{} epoch:".format(str(epoch)))
        print("loss:{} pretrain_loss:{} sim_loss:{}".format(str(np.mean(loss_list)),str(np.mean(pretrain_loss_lis)),str(np.mean(sim_loss_lis))))
        pred_labels = []
        goldens = []
        prt_id = 0
        for batch in get_pretrain_batch(config,pretrain_data2,word2idx_dict,shuffle=False):
            prt_id+=1
            pp,ppp,pred_label = sess.run([match.prt_loss,match.prt_pred,match.pretrain_pred_labels],feed_dict={match.pretrain_sents: batch['sents'], match.pretrain_pats: batch['pats'],match.is_train:False,match.pretrain_labels: batch['labels']})
            pred_label = list(pred_label)
            golden = list(np.reshape(batch['labels'],[-1]))
            assert len(golden)==len(pred_label)
            pred_labels.extend(pred_label)
            goldens.extend(golden)
        p,r,f = f_score(pred_labels,goldens)
        print('PRF:',(p,r,f))
        if p>0.9 and r>0.9:
            break
        print('\n')
    saver.save(sess,'dump_models/explanation.ckpt')
    return unlabeled_data,sess,match,pretrain_data,np.array(lfs['rels'],np.float32)

