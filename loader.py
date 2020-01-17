from tqdm import tqdm
from collections import Counter
import numpy as np
import ujson as json
import constant
import pickle
from CCG.utils import Phrase

def entity_masks():
    masks = []
    subj_entities = list(constant.SUBJ_NER_TO_ID.keys())
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    return masks


def read_glove(path, counter, size, dim):
    embedding_dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, total=size):
            array = line.split()
            word = "".join(array[0:-dim])
            vector = list(map(float, array[-dim:]))
            if word in counter:
                embedding_dict[word] = vector
    return embedding_dict


def token2id(config, counter, embedding_dict):
    vec_size = len(list(embedding_dict.values())[0])
    masks = entity_masks()
    top_k = [key for key, value in counter.most_common(config.top_k)] + masks
    token2idx_dict = {}
    token2idx_dict[constant.PAD_TOKEN] = constant.PAD_ID
    token2idx_dict[constant.UNK_TOKEN] = constant.UNK_ID
    for token in embedding_dict.keys():
        if token not in top_k:
            token2idx_dict[token] = len(token2idx_dict)
    embedding_dict[constant.PAD_TOKEN] = [0. for _ in range(vec_size)]
    embedding_dict[constant.UNK_TOKEN] = [0. for _ in range(vec_size)]
    num_fixed = len(token2idx_dict)
    for token in list(top_k):
        token2idx_dict[token] = len(token2idx_dict)
        if token not in embedding_dict:
            embedding_dict[token] = [np.random.normal(-1., 1.) for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = np.array([idx2emb_dict[idx] for idx in range(len(token2idx_dict))], dtype=np.float32)
    word_emb, ext_emb = emb_mat[:num_fixed, :], emb_mat[num_fixed:, :]
    return token2idx_dict, word_emb, ext_emb


def get_counter(path):
    counter = Counter()
    with open(path, "rb") as f:
        data = pickle.load(f)
        for d in data:
            tokens = d[1].token
            for token in tokens:
                if not token.startswith('SUBJ-') and not token.startswith('OBJ-'):
                    counter[token] += 1
    return counter


def read_data(path):
    examples = []
    with open(path, "rb") as f:
        data = pickle.load(f)
        for d in data:
            rel = constant.LABEL_TO_ID[d[0]]
            examples.append({"phrase": d[1], "rel": rel})
    return examples

def read_pretrain(config,num=1):
    if num==2:
        sentence_path = config.pretrain_sent_file2
        seqlabel_path = config.pretrain_label_file2
    else:
        sentence_path = config.pretrain_sent_file
        seqlabel_path = config.pretrain_label_file
    with open(sentence_path,'r') as f:
        data = json.load(f)
    pats = [elem[0] for elem in data]
    sents = [elem[1] for elem in data]
    labels = np.load(seqlabel_path)
    return pats,sents,labels
