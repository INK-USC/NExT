import os
import tensorflow as tf
import constant

from main import read
from sl import pseudo_labeling
from CCG.get_data_for_classifier import transform
import json
import CCG.Parser as Parser
from main import log

flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

home = os.path.expanduser("~")
train_file = os.path.join("./data", "tacred", "train.pkl")
dev_file = os.path.join("./data", "tacred", "dev.pkl")
test_file = os.path.join("./data", "tacred", "test.pkl")
glove_word_file = os.path.join("./data", "glove", "glove.840B.300d.txt")
emb_dict = os.path.join("./data", "tacred", "emb_dict.json")
pattern_file = os.path.join("./", "CCG", "explanations.json")

target_dir = "data"
log_dir = "log/event"
save_dir = "log/model"

flags.DEFINE_string("dataset", "tacred", "")
flags.DEFINE_string("mode", "refine", "pretrain / refine / few")
flags.DEFINE_string("gpu", "1", "The GPU to run on")

flags.DEFINE_string("target_dir", target_dir, "")
flags.DEFINE_string("log_dir", log_dir, "")
flags.DEFINE_string("save_dir", save_dir, "")
flags.DEFINE_string("glove_word_file", glove_word_file, "")
flags.DEFINE_string("pattern_file", pattern_file, "")


flags.DEFINE_string("pretrain_sent_file", './data/tacred/PT_toks.json', "")
flags.DEFINE_string("pretrain_label_file", './data/tacred/PT_pattern_mask.npy', "")
flags.DEFINE_string("pretrain_sent_file2", './data/tacred/TK_tok_exp.json', "")
flags.DEFINE_string("pretrain_label_file2", './data/tacred/TK_label.npy', "")

flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("dev_file", dev_file, "")
flags.DEFINE_string("test_file", test_file, "")
flags.DEFINE_string("emb_dict", emb_dict, "")

flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("top_k", 100000, "Finetune top k words in embedding")
flags.DEFINE_integer("length", 110, "Limit length for sentence")
flags.DEFINE_integer("num_class", len(constant.LABEL_TO_ID), "Number of classes")
flags.DEFINE_string("tag", "", "The tag name of event files")

flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_integer("pseudo_size", 100, "Batch size for pseudo labeling")
flags.DEFINE_integer('pretrain_size_together',100,"Batch size for pretraining module")
flags.DEFINE_integer("num_epoch", 50, "Number of epochs")
flags.DEFINE_integer("period", 10, "period to save batch loss")
flags.DEFINE_string("optimizer", "adagrad", "Training method [sgd, adagrad, adam]")
flags.DEFINE_float("init_lr", 0.5, "Initial lr")
flags.DEFINE_float("lr_decay", 0.95, "Decay rate")
flags.DEFINE_float("keep_prob", 0.5, "Keep prob in dropout")
flags.DEFINE_float("word_keep_prob", 0.96, "Keep prob for word")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden", 200, "Hidden size")
flags.DEFINE_integer("att_hidden", 200, "Hidden size for attention")

flags.DEFINE_bool("use_sur", True, "Whether to use word-level matching")
flags.DEFINE_bool("use_cont", False, "Whether to use context-level matching")
flags.DEFINE_string("string_sim", "att", "the method of string matching [mean, att]")
flags.DEFINE_float("percent", 1.0, "Sample rate of unlabled data")
flags.DEFINE_float("alpha", 0.2, "pretrain loss")
flags.DEFINE_float("beta", 0.5, "simloss")
flags.DEFINE_float("gamma", 0.7, "unlabeled loss")

flags.DEFINE_float('pos_weight',1,'pos_weight of pretrain loss')
flags.DEFINE_float('pretrain_lr',0.1,'pretrain learning rate')
flags.DEFINE_float('pretrain_threshold',0.5,'pretrain test threshold')
flags.DEFINE_integer('pretrain_train_size',40000,'pretrain traning size')
flags.DEFINE_integer('pretrain_test_size',2000,'pretrain test size')
flags.DEFINE_integer('pretrain_epoch',10,'pretrain epoch')
flags.DEFINE_integer('pretrain_size',100,'pretrain size')   #set this equal to pretrain_size_together
flags.DEFINE_float('pretrain_alpha',0.5,'simloss rate')


def main(_):
    config = flags.FLAGS
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    data = read(config)


    paser_train = Parser.FromExp2LF(True)                                         #parsing
    paser_train.Train()
    transform()


    unlabeled_data,sess,model,pretrain_data,rels = pseudo_labeling(config, data) #train

    samples = []

    for i in range(2000):                                                        #predict
        predcitlist = log(config,[unlabeled_data[i]],pretrain_data,data[0], model, sess,rels)
        sample = {"sent":unlabeled_data[i]['phrase'].sentence,"help":predcitlist}
        samples.append(sample)
    sess.close()
    with open("170.json",'w') as f:
        json.dump(samples,f)

if __name__ == "__main__":
    tf.app.run()
