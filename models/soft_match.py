import tensorflow as tf
from func import Cudnn_RNN, dropout, attention, dense, log, cosine, mean
import numpy as np
from util import get_word
from CCG.utils import parse_sem,recurse_soft
from CCG.constant import  entity_type,TYPE2ID
import constant

class Soft_Match(object):
    def __init__(self, config,logic_forms,rels,keywords,keywords_rels,raw_keywords,mat=None, word2idx_dict=None, pseudo=False): #logic_forms [str],rels:[num_lf,num_class],keywords = [str],keywords_rel:[num_keywords,num_class]
        self.raw_logic_forms = logic_forms
        self.logic_forms = [parse_sem(logic_form) for logic_form in logic_forms]
        assert False not in self.logic_forms
        self.keywords_rels = tf.constant(np.array(keywords_rels,np.float32),tf.int32)
        self.keywords = keywords
        self.keyword_dict = {keyword: keyword_idx for keyword_idx, keyword in enumerate(self.keywords)}

        self.raw_keywords = raw_keywords
        self.raw_keyword_dict = {keyword: keyword_idx for keyword_idx, keyword in enumerate(self.raw_keywords)}

        self.mask_mat = np.zeros([config.length,config.length,config.length],np.int32)
        for i in range(config.length):
            for j in range(config.length):
                self.mask_mat[i,j,i:j+1] = 1            #[ ]

        self.entity_type = {constant.LABEL_TO_ID[rel]:[TYPE2ID[text.split()[0]],TYPE2ID[text.split()[1]]] for rel,text in entity_type.items()}
        self.rels_id = np.reshape(np.argmax(rels,axis=1),[-1])

        self.labeling_functions_soft = [recurse_soft(logic_form) for logic_form in self.logic_forms]
        assert False not in self.labeling_functions_soft
        self.rels = tf.constant(rels,dtype=tf.float32)

        self.config = config
        self.mat = mat
        self.pseudo = pseudo
        self.word2idx_dict = word2idx_dict
        self.global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.pre_global_step = tf.get_variable("pre_global_step", [], initializer=tf.constant_initializer(0), dtype=tf.float32)

        self.pretrain_sents = tf.placeholder(tf.int32,shape=(None,config.length),name='pretrain_sents')
        self.pretrain_pats = tf.placeholder(tf.int32, shape=(None, 10), name='pretrain_pats')
        self.pretrain_labels = tf.placeholder(tf.float32, shape=(None, config.length), name='pretrain_labels')

        self.phrases_input = tf.placeholder(tf.int32,shape=(None,config.length*2+4),name='phrases_input')
        self.hard_match_func_idx = tf.placeholder(tf.int32,shape=(None,),name='hard_match_func_idx')
        self.sent_word = tf.placeholder(tf.int32,shape=(None,config.length),name='sent_word')
        self.rel = tf.placeholder(tf.float32, shape=(None, config.num_class), name="rel")

        self.keywords_int = []
        for keyword in keywords:
            splt = keyword.split()
            tok = get_word(splt,word2idx_dict,10)
            self.keywords_int.append(tok)
        self.pats = tf.constant(self.keywords_int,dtype=tf.int32,name='pats')

        self.raw_keywords_int = []
        for keyword in raw_keywords:
            splt = keyword.split()
            tok = get_word(splt, word2idx_dict, 10)
            self.raw_keywords_int.append(tok)
        self.raw_pats = tf.constant(self.raw_keywords_int,dtype=tf.int32,name='raw_pats')

        self.num_logic_form = len(self.labeling_functions_soft)

        self.is_train = tf.placeholder(tf.bool)
        self.lr = tf.get_variable("lr", [], initializer=tf.constant_initializer(config.init_lr), trainable=False)
        self.alpha = tf.get_variable("alpha", [], initializer=tf.constant_initializer(config.alpha), trainable=False)
        self.beta = tf.get_variable("beta", [], initializer=tf.constant_initializer(config.beta), trainable=False)
        self.gamma = tf.get_variable("gamma", [], initializer=tf.constant_initializer(config.gamma), trainable=False)
        self.pretrain_lr = tf.get_variable("pretrain_lr", [], initializer=tf.constant_initializer(config.pretrain_lr), trainable=False)
        self.pretrain_alpha = tf.get_variable("pretrain_alpha", [], initializer=tf.constant_initializer(config.pretrain_alpha), trainable=False)

        with tf.variable_scope("embedding"):
            word_mat, ext_mat = mat
            word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)
            ext_word_mat = tf.get_variable("ext_word_mat", initializer=tf.constant(ext_mat, dtype=tf.float32))
            self.word_mat = tf.concat([word_mat, ext_word_mat], axis=0)
        self.unmatch_type_score = 0
        self.ready()

        if config.optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer
        elif config.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer
        else:
            optimizer = tf.train.AdamOptimizer

        with tf.variable_scope("optimizer"):
            opt = optimizer(learning_rate=self.lr)
            grads = opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)
        with tf.variable_scope('pretrain'):
            pre_opt = optimizer(learning_rate=self.pretrain_lr)
            pre_grads = pre_opt.compute_gradients(self.pretrain_loss_v2)
            pre_gradients, pre_variables = zip(*pre_grads)
            pre_capped_grads, _ = tf.clip_by_global_norm(pre_gradients, config.grad_clip)
            self.pre_train_op = pre_opt.apply_gradients(zip(pre_capped_grads, pre_variables), global_step=self.pre_global_step)

    def type_restrict(self,i):
        type_ = self.entity_type[self.rels_id[i]]
        restrict_subj = tf.cast(tf.equal(tf.reshape(self.phrases_input[:,-2],[1,-1]),type_[0]),tf.float32)
        restrict_obj = tf.cast(tf.equal(tf.reshape(self.phrases_input[:,-1],[1,-1]),type_[1]),tf.float32)
        restrict_subj = restrict_subj+(1.0-restrict_subj)*self.unmatch_type_score
        restrict_obj = restrict_obj+(1.0-restrict_obj)*self.unmatch_type_score
        return restrict_subj*restrict_obj

    def ready(self):
        config = self.config
        d = config.hidden

        batch_size = tf.shape(self.sent_word)[0]
        sent_mask = tf.cast(self.sent_word, tf.bool)
        sent_len = tf.reduce_sum(tf.cast(sent_mask, tf.int32), axis=1)
        sent_maxlen = config.length

        sent = self.sent_word

        pretrain_sent_mask = tf.cast(self.pretrain_sents,tf.bool)
        rnn = Cudnn_RNN(num_layers=2, num_units=d // 2, keep_prob=config.keep_prob, is_train=self.is_train)
        label_mat,_= FIND_module(sent,self.raw_pats,self.word_mat,config,tf.constant(False,tf.bool),rnn)
        label_mat = tf.sigmoid(label_mat)*tf.tile(tf.reshape(tf.cast(sent_mask,tf.float32),[batch_size,sent_maxlen,1]),[1,1,self.raw_pats.get_shape()[0]])

        # label_mat = tf.cast(tf.greater(label_mat,0.7),tf.float32)

        _,keywords_sim= FIND_module(sent,self.pats,self.word_mat,config,self.is_train,rnn)
        # keywords_sim = tf.sigmoid(keywords_sim)

        pretrain_pred_labels,_ = FIND_module(self.pretrain_sents,self.pretrain_pats,self.word_mat,config,self.is_train,rnn)
        pretrain_pred_labels = tf.transpose(pretrain_pred_labels,[0,2,1])
        gather_order = tf.tile(tf.reshape(tf.range(max(config.pretrain_size,config.pretrain_size_together)), [-1, 1]),[1,2])
        pretrain_pred_labels = tf.gather_nd(pretrain_pred_labels,gather_order)
        self.pretrain_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=self.pretrain_labels,logits=pretrain_pred_labels,pos_weight=config.pos_weight)*tf.cast(pretrain_sent_mask,tf.float32),axis=1)/tf.reduce_sum(tf.cast(pretrain_sent_mask,tf.float32),axis=1))#tf.losses.mean_squared_error(labels=self.pretrain_labels,predictions=pretrain_pred_labels)

        self.prt_loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.pretrain_labels,logits=pretrain_pred_labels,pos_weight=config.pos_weight)*tf.cast(pretrain_sent_mask,tf.float32)
        self.prt_pred = tf.sigmoid(pretrain_pred_labels)*tf.cast(pretrain_sent_mask,tf.float32)
        self.pretrain_pred_labels = tf.reshape(tf.cast(tf.greater(tf.sigmoid(pretrain_pred_labels)*tf.cast(pretrain_sent_mask,tf.float32),config.pretrain_threshold),tf.int32),[-1])

        neg_idxs = tf.matmul(self.keywords_rels, tf.transpose(self.keywords_rels, [1, 0]))
        pat_pos = tf.square(tf.maximum(0.9 - keywords_sim, 0.))
        pat_pos = tf.reduce_max(pat_pos - tf.cast(1 - neg_idxs,tf.float32)*tf.constant(1e30,tf.float32), axis=1)

        pat_neg = tf.square(tf.maximum(keywords_sim, 0.))
        pat_neg = tf.reduce_max(pat_neg - tf.constant(1e30,tf.float32) * tf.cast(neg_idxs,tf.float32), axis=1)
        pat_simloss = tf.reduce_mean(pat_pos + pat_neg,axis=0)

        # clustering的loss
        self.sim_loss = sim_loss = pat_simloss

        self.pretrain_loss_v2 = self.pretrain_loss+self.pretrain_alpha*self.sim_loss

        sim_raw = []

        for i, soft_labeling_function in enumerate(self.labeling_functions_soft):
            try:
                # soft_labeling_function(label_mat, self.raw_keyword_dict, self.mask_mat)(
                   # self.phrases_input) -> (B x 1)
                sim_raw.append(soft_labeling_function(label_mat, self.raw_keyword_dict, self.mask_mat)(
                    self.phrases_input) * self.type_restrict(i))
            except:
                print(i)
                sim_raw.append(tf.cast(tf.reshape(0*self.phrases_input[:,0],[1,-1]),tf.float32))

        self.sim =sim= tf.transpose(tf.concat(sim_raw,axis=0),[1,0]) #[tf.shape==(batch_size,1)]*num_functions->[batch_size,]
        with tf.variable_scope("classifier"):
            sent_emb = tf.nn.embedding_lookup(self.word_mat, sent)
            sent_emb = dropout(sent_emb, keep_prob=config.word_keep_prob, is_train=self.is_train, mode="embedding")
            rnn = Cudnn_RNN(num_layers=2, num_units=d // 2, keep_prob=config.keep_prob, is_train=self.is_train)
            cont, _ = rnn(sent_emb, seq_len=sent_len, concat_layers=False)
            cont_d = dropout(cont, keep_prob=config.keep_prob, is_train=self.is_train)
            att_a = attention(cont_d, config.att_hidden, mask=sent_mask)
            att2_d = tf.reduce_sum(tf.expand_dims(att_a, axis=2) * cont_d, axis=1)
            logit = dense(att2_d, config.num_class, use_bias=False)
            pred = tf.nn.softmax(logit)
            with tf.variable_scope("pred"):

                if not self.pseudo:

                    sent_loss = self.sent_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=self.rel), axis=0)
                else:

                    self.hard_train_loss = sent_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logit[:config.batch_size], labels=self.rel[:config.batch_size]), axis=0)

                    # lsim = sim[:config.batch_size]
                    # index_tensor = tf.reshape(tf.constant(np.arange(config.batch_size),tf.int32),[config.batch_size,1])
                    # select_tensor = tf.reshape(self.hard_match_func_idx,[config.batch_size,1])
                    # probs = tf.reshape(tf.gather_nd(lsim,tf.concat([index_tensor,select_tensor],axis=1)),[config.batch_size,1])
                    # self.labeled_loss = labeled_loss = tf.reduce_mean(tf.square((1-probs)))

                    xsim = tf.stop_gradient(sim[config.batch_size:])

                    pseudo_rel = tf.gather(self.rels, tf.argmax(xsim, axis=1))
                    bound = tf.reduce_max(xsim, axis=1)
                    weight = tf.nn.softmax(10.0 * bound)

                    self.unlabeled_loss = unlabeled_loss = tf.reduce_sum(weight * tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logit[config.batch_size:], labels=pseudo_rel), axis=0)

                    sent_loss = self.sent_loss = sent_loss + self.gamma * unlabeled_loss+self.alpha*self.pretrain_loss#+self.alpha*labeled_loss

        #算entropy来对no_relation推断
        self.max_val = entropy = tf.reduce_sum(pred * -log(pred), axis=1)
        #pred是test时候用到的
        self.pred = tf.argmax(pred, axis=1)
        self.loss = sent_loss + self.beta * sim_loss
        #similarity model预测出来的结果
        # self.sim_pred = tf.argmax(tf.gather(self.rels, tf.argmax(self.sim, axis=1)), axis=1)
        # self.sim_max_val = tf.reduce_max(self.sim, axis=1)
        #true label
        self.gold = tf.argmax(self.rel, axis=1)
        self.entropy = tf.reduce_mean(entropy, axis=0)



def FIND_module(sent,pats,word_mat,config,is_train,rnn,scope='Find_module'):#sents_emb [batch,maxlength_sent] pats [num_pats,maxlength_pat]   [batch,maxlength_sent,dim]
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        keep_prob = config.keep_prob
        d = config.hidden
        batch_size = tf.shape(sent)[0]
        maxlength_sent = tf.shape(sent)[1]
        dim = tf.shape(word_mat)[1]
        num_pats = tf.shape(pats)[0]

        sent_mask = tf.cast(sent, tf.bool)

        pat_mask = tf.cast(pats, tf.bool)
        pat_len = tf.reduce_sum(tf.cast(pat_mask, tf.int32), axis=1)

        with tf.variable_scope('embedding'):
            sent_emb = tf.nn.embedding_lookup(word_mat, sent) # Rahul: convert sentence tokens into vectors

            # sent_emb_d = dropout(sent_emb, keep_prob=config.word_keep_prob, is_train=is_train, mode="embedding") # ***** Rahul-Q: NOT NEEDED ? ***** (doesn't seem to be used)
            
            pat_emb = tf.nn.embedding_lookup(word_mat, pats)  # Rahul: convert pattern tokens into vectors

            # pat_emb_d = dropout(pat_emb, keep_prob=config.word_keep_prob, is_train=is_train,mode='embedding') # ***** Rahul-Q: NOT NEEDED ? ***** (doesn't seem to be used)

        with tf.variable_scope('stack'):
            pad = tf.zeros([batch_size,1,dim],tf.float32)

            sent_emb_pad = tf.concat([pad,sent_emb,pad],axis=1) # Rahul: adding padding to each sentence
            sent_emb_stack_2 = tf.reshape(sent_emb_pad,[batch_size,maxlength_sent+2,1,dim]) # Rahul: reshapping padded sentences to (batch_size, max_sentence_length+2, 1, embedding_dim)
            sent_emb_stack_2 = tf.concat([sent_emb_stack_2[:,0:-1,:],sent_emb_stack_2[:,1:,:]],axis=2) # Rahul: slicing original vector embeddings and creating bigrams
            sent_emb_stack_2 = tf.reshape(sent_emb_stack_2,[batch_size*(maxlength_sent+1),2,dim]) # Rahul: stacking bigrams into (batch_size *(maxlength_sent+1), 2, embedding_dim)

            sent_emb_pad2 = tf.concat([pad,pad,sent_emb,pad,pad],axis=1) # Rahul: similar actions here for trigrams
            sent_emb_stack_3 = tf.reshape(sent_emb_pad2,[batch_size,maxlength_sent+4,1,dim])
            sent_emb_stack_3 = tf.concat([sent_emb_stack_3[:, 0:-2, :], sent_emb_stack_3[:, 1:-1, :], sent_emb_stack_3[:, 2:, :]], axis=2)
            sent_emb_stack_3 = tf.reshape(sent_emb_stack_3,[batch_size*(maxlength_sent+2),3,dim])

            sent_emb_stack_1 = tf.reshape(sent_emb,[batch_size*maxlength_sent,1,dim]) # Rahul: similar actions here for unigrams

        # # ***** Rahul: NOT NEEDED ? *****
        # with tf.variable_scope('stack_d'):
        #     pad = tf.zeros([batch_size,1,dim],tf.float32)

        #     sent_emb_pad_d = tf.concat([pad,sent_emb_d,pad],axis=1)
        #     sent_emb_stack_2_d = tf.reshape(sent_emb_pad_d,[batch_size,maxlength_sent+2,1,dim])
        #     sent_emb_stack_2_d = tf.concat([sent_emb_stack_2_d[:,0:-1,:],sent_emb_stack_2_d[:,1:,:]],axis=2)
        #     sent_emb_stack_2_d = tf.reshape(sent_emb_stack_2_d,[batch_size*(maxlength_sent+1),2,dim])

        #     sent_emb_pad2_d = tf.concat([pad,pad,sent_emb_d,pad,pad],axis=1)
        #     sent_emb_stack_3_d = tf.reshape(sent_emb_pad2_d,[batch_size,maxlength_sent+4,1,dim])
        #     sent_emb_stack_3_d = tf.concat([sent_emb_stack_3_d[:, 0:-2, :], sent_emb_stack_3_d[:, 1:-1, :], sent_emb_stack_3_d[:, 2:, :]], axis=2)
        #     sent_emb_stack_3_d = tf.reshape(sent_emb_stack_3_d,[batch_size*(maxlength_sent+2),3,dim])

        #     sent_emb_stack_1_d = tf.reshape(sent_emb_d,[batch_size*maxlength_sent,1,dim])
        # ***** Rahul-Q: NOT NEEDED ? *****

        with tf.variable_scope("encoder"):
            with tf.variable_scope('encode_pat'):
                pat, _ = rnn(pat_emb, seq_len=pat_len, concat_layers=False)       #[numpats,d] <- Rahul-Q: isn't this [numpats, pat_len, encoding_dim]?
                                                                                  # Rahul-Q: also what does concat_layers=False do?, does this mean encoding_dim = 100 and not 200?

                pat_d = dropout(pat, keep_prob=config.keep_prob, is_train=is_train) # Rahul: dropout applied to encoding
            with tf.variable_scope('encode_sent'):
                cont_stack_3, _ = rnn(sent_emb_stack_3,seq_len=3 * tf.ones([batch_size * (maxlength_sent + 2)], tf.int32),concat_layers=False) # Rahul: encoding trigrams
                                                                                                                                               # Rahul-Q: Again what does concat_layers=False do?
                
                cont_stack_2, _ = rnn(sent_emb_stack_2, seq_len=2*tf.ones([batch_size*(maxlength_sent+1)],tf.int32), concat_layers=False) #[batch_size*(maxlength_sent+1),d]
                                                                                                                                          # Rahul: encoding bigrams
                                                                                                                                          # Rahul-Q: Again what does concat_layers=False do?
                
                cont_stack_1, _ = rnn(sent_emb_stack_1, seq_len=tf.ones([batch_size*maxlength_sent],tf.int32), concat_layers=False)  #[batch_size*maxlength_sent,d]
                                                                                                                                     # Rahul: encoding unigrams
                                                                                                                                     # Rahul-Q: Again what does concat_layers=False do?
                                                                                                                    
                cont_stack_3_d = dropout(cont_stack_3, keep_prob=keep_prob, is_train=is_train) # Rahul: apply dropout to trigrams

                cont_stack_2_d = dropout(cont_stack_2, keep_prob=keep_prob, is_train=is_train) # Rahul: apply dropout to bigrams

                cont_stack_1_d = dropout(cont_stack_1, keep_prob=keep_prob, is_train=is_train) # Rahul: apply dropout to unigrams

        with tf.variable_scope('attention'):
            pat_d_a = attention(pat_d,config.att_hidden, mask=pat_mask) # Rahul: Getting attention weights of LSTM output of each pattern

            cont_stack_2_d_a = attention(cont_stack_2_d,config.att_hidden) # Rahul: Getting attention weights of LSTM output of each bigram

            cont_stack_3_d_a = attention(cont_stack_3_d,config.att_hidden) # Rahul: Getting attention weights of LSTM output of each trigram

            cont_stack_3_att = tf.reduce_sum(tf.expand_dims(cont_stack_3_d_a, axis=2) * cont_stack_3, axis=1) # Rahul: Constructing weighted representation of each trigram using attention weights
                                                                                                              # Rahul-Q: You're using cont_stack_3, which is the vectors before dropout,
                                                                                                              #          however to get the weights you used vectors after dropout is this what you intended?
            
            cont_stack_2_att = tf.reduce_sum(tf.expand_dims(cont_stack_2_d_a, axis=2) * cont_stack_2, axis=1) # Rahul: Constructing weighted representation of each bigram using attention weights
                                                                                                              # Rahul-Q: You're using cont_stack_2, which is the vectors before dropout,
                                                                                                              #          however to get the weights you used vectors after dropout is this what you intended?
            
            pat_d_att = tf.reduce_sum(tf.expand_dims(pat_d_a, axis=2) * pat_d, axis=1) # Rahul: Constructing weighted representation of each pattern using attention weights
                                                                                       # Rahul-Q: You're using pat_d, which is the vectors after dropout, this is different than the
                                                                                       #          above strategy, is this what you intended?
                                                                                       # Rahul-Q: Below you also create another attention representation of a pattern, is this what you intended?
            
            pat_att = tf.reduce_sum(tf.expand_dims(pat_d_a, axis=2) * pat, axis=1) # Rahul-Q: Look at comment above, confusion around which attention representation of a pattern to use
            
            cont_stack_1_att = tf.squeeze(cont_stack_1) # Rahul: unigram representation requires no attention
        
        # ***** Rahul-Q: NOT NEEDED ? *****
        # with tf.variable_scope('emb_attention'):
        #     pat_emb_d_a = attention(pat_emb_d, config.att_hidden, mask=pat_mask)
        #     pat_emb_d_att = tf.reduce_sum(tf.expand_dims(pat_emb_d_a, axis=2) * pat_emb_d, axis=1)
        #     pat_emb_att = tf.reduce_sum(tf.expand_dims(pat_emb_d_a, axis=2) * pat_emb, axis=1)

        #     sent_emb_stack_3_d_a = attention(sent_emb_stack_3_d, config.att_hidden)
        #     sent_emb_stack_3_att = tf.reduce_sum(tf.expand_dims(sent_emb_stack_3_d_a, axis=2) * sent_emb_stack_3, axis=1)

        #     sent_emb_stack_2_d_a = attention(sent_emb_stack_2_d, config.att_hidden)
        #     sent_emb_stack_2_att = tf.reduce_sum(tf.expand_dims(sent_emb_stack_2_d_a, axis=2) * sent_emb_stack_2, axis=1)

        #     sent_emb_stack_1_att = tf.squeeze(sent_emb_stack_1)
         # ***** Rahul-Q: NOT NEEDED ? *****

        with tf.variable_scope('Score'):
            scores_stack_2 = cosine(cont_stack_2_att,pat_d_att,weighted=False) # Rahul: cosine similarity of bigram representations and pat_d_att representation
            scores_stack_1 = cosine(cont_stack_1_att,pat_d_att,weighted=False) # Rahul: cosine similarity of unigram representations and pat_d_att representation
            scores_stack_3 = cosine(cont_stack_3_att, pat_d_att, weighted=False) # Rahul: cosine similarity of trigram representations and pat_d_att representation

            scores_stack_3 = tf.reshape(scores_stack_3, [batch_size, 1, maxlength_sent + 2, num_pats]) # Rahul: reshaping of trigram scores
            scores_stack_2 = tf.reshape(scores_stack_2,[batch_size,1,maxlength_sent+1,num_pats]) # Rahul: reshaping of bigram scores
            scores_stack_1 = tf.reshape(scores_stack_1,[batch_size,1,maxlength_sent,num_pats]) # Rahul: reshaping of bigram scores
            scores_sim = cosine(pat_att, pat_d_att, weighted=False) # Rahul-Q: What is this? You're taking the cosine similarity between the attention representation of a pattern
                                                                    # with dropout, and attention representation of a pattern without dropout, not sure what this cosine represents.
                                                                    # It seems like this is being used for L_sim, but I don't see how this cosine value represents the needed
                                                                    # calculations for L_sim; Lsim = max q1∈Q+ {(τ − cos(zqD, zq1D))^2} + max q2∈Q− {cos(zqD, zq2D)^2}
        
        with tf.variable_scope('emb_Score'):
            # scores_stack_3_emb = cosine(sent_emb_stack_3_att,pat_emb_d_att) # ***** Rahul-Q: NOT NEEDED ? *****
            # scores_stack_2_emb = cosine(sent_emb_stack_2_att,pat_emb_d_att) # ***** Rahul-Q: NOT NEEDED ? *****
            # scores_stack_1_emb = cosine(sent_emb_stack_1_att,pat_emb_d_att) # ***** Rahul-Q: NOT NEEDED ? *****

            # scores_stack_3_emb = tf.reshape(scores_stack_3_emb, [batch_size, 1, maxlength_sent + 2, num_pats]) # ***** Rahul-Q: NOT NEEDED ? *****
            # scores_stack_2_emb = tf.reshape(scores_stack_2_emb,[batch_size,1,maxlength_sent+1,num_pats]) # ***** Rahul-Q: NOT NEEDED ? *****
            # scores_stack_1_emb = tf.reshape(scores_stack_1_emb,[batch_size,1,maxlength_sent,num_pats]) # ***** Rahul-Q: NOT NEEDED ? *****

            # phi = 0 # ***** Rahul-Q: NOT NEEDED ? *****
            # scores_stack_3 = phi * scores_stack_3_emb + (1 - phi) * scores_stack_3 # ***** Rahul-Q: NOT NEEDED ? *****
            # scores_stack_2 = phi*scores_stack_2_emb+(1-phi)*scores_stack_2 # ***** Rahul-Q: NOT NEEDED ? *****
            # scores_stack_1 = phi*scores_stack_1_emb+(1-phi)*scores_stack_1 # ***** Rahul-Q: NOT NEEDED ? *****

            # Rahul: Reshaping scores to then finally be combined by Linear layer to get weighted similarity score later
            scores = tf.concat([scores_stack_3[:,:,0:-2,:],scores_stack_3[:,:,1:-1,:],scores_stack_3[:,:,2:,:],scores_stack_2[:,:,0:-1,:],scores_stack_2[:,:,1:,:],scores_stack_1],axis=1)
            scores = tf.reshape(scores,[batch_size,6,maxlength_sent,num_pats])
            scores = tf.transpose(scores,[0,3,1,2])
            scores = tf.reshape(scores,[batch_size*num_pats,6,maxlength_sent])

            # scores_sim_emb = cosine(pat_emb_att, pat_emb_d_att) # ***** Rahul-Q: NOT NEEDED ? *****
            # scores_sim = phi*scores_sim_emb+(1-phi)*scores_sim # ***** Rahul-Q: NOT NEEDED ? *****

        # Rahul-Q: I'm not really sure what this piece of code is doing, if you could write some comments here that would be amazing :)
        with tf.variable_scope('SeqLabel'):
            seq = tf.layers.dense(tf.transpose(scores,[0,2,1]),1)
            seq = tf.squeeze(seq)
            seq = tf.reshape(seq,[batch_size,num_pats,maxlength_sent])
            #seq = tf.reshape(tf.reduce_max(scores,axis=1),[batch_size,num_pats,maxlength_sent])
            seq = tf.transpose(seq,[0,2,1])
            seq = seq*tf.tile(tf.cast(tf.reshape(sent_mask,[batch_size,maxlength_sent,1]),tf.float32),[1,1,num_pats])

        return seq,scores_sim
