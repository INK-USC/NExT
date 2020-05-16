import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
from collections import Counter
from util import get_batch, get_feeddict,get_pretrain_batch
import os

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

def log(config, data, pretrain_data,word2idx_dict, model, sess, writer=None, label="train", entropy=None, bound=None):
    global_step = sess.run(model.global_step) + 1
    golds, preds, vals, sim_preds, sim_vals = [], [], [], [], []
    simss = []
    for batch,_ in zip(get_batch(config, data, word2idx_dict,shuffle=False),get_pretrain_batch(config,pretrain_data,word2idx_dict,pretrain=False)):
        gold, pred, val, sim_pred, sim_val = sess.run([model.gold, model.pred, model.max_val, model.sim_pred, model.sim_max_val],
                                                      feed_dict=get_feeddict(model, batch,_, is_train=False))
        prt_sim = sess.run(model.sim, feed_dict=get_feeddict(model, batch, _, is_train=False))

        batch_sents = batch['raw_sent']

        golds += gold.tolist()
        preds += pred.tolist()
        vals += val.tolist()
        sim_preds += sim_pred.tolist()
        sim_vals += sim_val.tolist()

    threshold = [0.01 * i for i in range(1, 200)]
    threshold2 = [0.05 * i for i in range(1, 20)]
    acc, recall, f1, jac = 0., 0., 0., 0.
    acc2, recall2, f12, jac2 = 0., 0., 0., 0.
    best_entro = 0.
    best_bound = 0.

    if entropy is None:
        for t in threshold:
            _preds = (np.asarray(vals, dtype=np.float32) <= t).astype(np.int32) * np.asarray(preds, dtype=np.int32)
            _preds = _preds.tolist()
            _acc, _recall, _f1, _jac = evaluate(golds, _preds)
            if _f1 > f1:
                acc, recall, f1, jac = _acc, _recall, _f1, _jac
                best_entro = t
    else:
        preds = (np.asarray(vals, dtype=np.float32) <= entropy).astype(np.int32) * np.asarray(preds, dtype=np.int32)
        preds = preds.tolist()
        acc, recall, f1, jac = evaluate(golds, preds)

    if bound is None:
        for t in threshold2:
            _sim_preds = (np.asarray(sim_vals, dtype=np.float32) >= t).astype(np.int32) * np.asarray(sim_preds, dtype=np.int32)
            _sim_preds = _sim_preds.tolist()
            _acc2, _recall2, _f12, _jac2 = evaluate(golds, _sim_preds)
            if _f12 > f12:
                acc2, recall2, f12, jac2 = _acc2, _recall2, _f12, _jac2
                best_bound = t
    else:
        sim_preds = (np.asarray(sim_vals, dtype=np.float32) >= bound).astype(np.int32) * np.asarray(sim_preds, dtype=np.int32)
        sim_preds = sim_preds.tolist()
        acc2, recall2, f12, jac2 = evaluate(golds, sim_preds)

    acc_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(label), simple_value=acc), ])
    rec_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/rec".format(label), simple_value=recall), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/f1".format(label), simple_value=f1), ])
    jac_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/jac".format(label), simple_value=jac), ])

    acc_sum2 = tf.Summary(value=[tf.Summary.Value(tag="{}/sim_acc".format(label), simple_value=acc2), ])
    rec_sum2 = tf.Summary(value=[tf.Summary.Value(tag="{}/sim_rec".format(label), simple_value=recall2), ])
    f1_sum2 = tf.Summary(value=[tf.Summary.Value(tag="{}/sim_f1".format(label), simple_value=f12), ])
    jac_sum2 = tf.Summary(value=[tf.Summary.Value(tag="{}/sim_jac".format(label), simple_value=jac2), ])

    entropy_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/entro".format(label), simple_value=sum(vals) / len(vals)), ])
    if writer is not None:
        writer.add_summary(acc_sum, global_step)
        writer.add_summary(rec_sum, global_step)
        writer.add_summary(f1_sum, global_step)
        writer.add_summary(jac_sum, global_step)
        writer.add_summary(acc_sum2, global_step)
        writer.add_summary(rec_sum2, global_step)
        writer.add_summary(f1_sum2, global_step)
        writer.add_summary(jac_sum2, global_step)
        writer.add_summary(entropy_sum, global_step)
    res = [golds, preds]
    return (acc, recall, f1), (acc2, recall2, f12), (best_entro, best_bound), res


def evaluate(key, prediction):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()
    union_relation = Counter()

    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == 0 and guess == 0:
            pass
        elif gold == 0 and guess != 0:
            guessed_by_relation[guess] += 1
        elif gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        elif gold != 0 and guess != 0:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            union_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    jaccard_micro = 0.0
    if sum(union_relation.values()) > 0:
        jaccard_micro = float(sum(correct_by_relation.values())) / float(sum(union_relation.values()))
    return prec_micro, recall_micro, f1_micro, jaccard_micro
