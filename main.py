import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
from collections import Counter
from util import get_batch, get_feeddict,get_pretrain_batch
import os
import constant

from loader import read_glove, get_counter, token2id, read_data,read_pretrain

tqdm.monitor_interval = 0
np.set_printoptions(threshold=np.nan)


def read(config):
    counter = get_counter(config.train_file)
    if os.path.exists(config.emb_dict):
        with open(config.emb_dict, "r") as fh:
            emb_dict = json.load(fh)
    else:
        emb_dict = read_glove(config.glove_word_file, counter, config.glove_word_size, config.glove_dim)
        with open(config.emb_dict, "w") as fh:
            json.dump(emb_dict, fh)
    word2idx_dict, fixed_emb, traiable_emb = token2id(config, counter, emb_dict)

    train_data = read_data(config.train_file)
    dev_data = read_data(config.dev_file)
    test_data = read_data(config.test_file)
    pretrain_data = read_pretrain(config)
    pretrain_data2 = read_pretrain(config,2)
    return word2idx_dict, fixed_emb, traiable_emb, train_data, dev_data, test_data,pretrain_data,pretrain_data2

def log(config, data, pretrain_data,word2idx_dict, model, sess,rels):
    global_step = sess.run(model.global_step) + 1
    sim_out = 0
    assert len(data) == 1
    for batch,_ in zip(get_batch(config, data, word2idx_dict,shuffle=False),get_pretrain_batch(config,pretrain_data,word2idx_dict,pretrain=False)):
        sim_out = sess.run(model.sim,feed_dict=get_feeddict(model, batch,_, is_train=False))
    sim_out = list(sim_out[0])
    assert len(sim_out)==rels.shape[0]
    sort_idx = np.argsort(np.array(sim_out))[-3:]
    if sim_out[sort_idx[-1]]<0.5:
        return {"Can't decide":1}
    else:
        return {constant.ID_TO_LABEL[np.argmax(rels[sort_idx[-1]])]:float(sim_out[sort_idx[-1]]),
                constant.ID_TO_LABEL[np.argmax(rels[sort_idx[-2]])]:float(sim_out[sort_idx[-2]]),
                constant.ID_TO_LABEL[np.argmax(rels[sort_idx[-3]])]:float(sim_out[sort_idx[-3]])}

