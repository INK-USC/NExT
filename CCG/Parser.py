import numpy as np
import json
try:
    from CCG.constant import *
    from CCG.utils import *
except:
    from constant import *
    from utils import *
from stanfordcorenlp import StanfordCoreNLP
import os
import pickle
import torch.nn as nn
import torch
import copy

class Parser():                                                       #now batch is 1
    def __init__(self,beam):
        self.beam = beam

        self.combfeature_size = 0

        self.opsfeature = opsfeature
        self.lexfeature_size = len(opsfeature)

        tensor = torch.zeros(self.lexfeature_size+self.combfeature_size, 1)
        self.theta = nn.init.xavier_uniform_(tensor).numpy()
        self.num_ = 0

    def load_all(self,Logic_form_treess,xs,ys):
        self.Logic_form_treess = Logic_form_treess
        self.parsed_semss = [[parse_sem(str(tree.label()[0].semantics())) for tree in Logic_form_trees] for Logic_form_trees in tqdm(self.Logic_form_treess)]
        self.recursess = [[recurse(parsed_sem) for parsed_sem in parsed_sems] for parsed_sems in tqdm(self.parsed_semss)]
        self.xs = xs
        self.ys = ys

    def load_one(self,index):
        self.Logic_form_trees = self.Logic_form_treess[index]
        self.parsed_sems = self.parsed_semss[index]
        self.recurses = self.recursess[index]

        self.x = self.xs[index]
        self.y = self.ys[index]
        self.LFs = []
        for i in range(len(self.Logic_form_trees)):
            parsed_sem = self.parsed_sems[i]
            if parsed_sem==False:
                continue
            LF = self.recurses[i]
            if LF!=False:
                self.LFs.append((self.Logic_form_trees[i],LF,parsed_sem))
        self.right_LFs = []
        self.false_LFs = []
        self.right_Logic_Forms = []
        self.wrong_logic_Forms = []
        self.right_sems = []
        self.wrong_sems = []
        for (L,LF,sem__) in self.LFs:
            try:
                if LF(self.x)==self.y:
                    self.right_LFs.append(LF)
                    self.right_Logic_Forms.append(L)
                    self.right_sems.append(sem__)
                else:
                    self.false_LFs.append(LF)
                    self.wrong_logic_Forms.append(L)
                    self.wrong_sems.append(sem__)
            except:
                pass

    def get_feature(self,logic_form_tree):
        feature_vector = np.zeros([self.combfeature_size+self.lexfeature_size,1],np.float32)

        op_feature = []
        if isinstance(logic_form_tree,str):
            sem = parse_sem(logic_form_tree)
        else:
            sem = parse_sem(str(logic_form_tree.label()[0].semantics()))
        if sem!=False:
            collect_features(sem,op_feature)

        for opfeature in op_feature:
            if opfeature[0]!=None and opfeature[1]!=None:
                opf = ''.join(list(opfeature))
                if opf in self.opsfeature:
                    feature_vector[self.combfeature_size + self.opsfeature[opf], 0] += 1.0
        return feature_vector

    def compute_lf_prob(self,l):
        numerator = np.exp(np.dot(self.get_feature(l).transpose(),self.theta))[0,0]
        return numerator/self.denominator_Ls

    def compute_gradients(self):
        if len(self.right_Logic_Forms)!=0:
            self.denominator_Ls = sum([np.exp(np.dot(self.get_feature(l).transpose(), self.theta))[0,0] for l in self.Logic_form_trees])
            self.right_Ls_exp_sum = sum([np.exp(np.dot(self.get_feature(l).transpose(), self.theta))[0,0] for l in self.right_Logic_Forms])
            E_1 = sum([self.compute_lf_prob(l)*self.get_feature(l) for l in self.right_Logic_Forms])/(self.right_Ls_exp_sum/self.denominator_Ls)
            E_2 = sum([self.compute_lf_prob(l)*self.get_feature(l) for l in self.Logic_form_trees])
            self.grad = E_1-E_2
        else:
            self.grad = np.zeros([self.lexfeature_size+self.combfeature_size, 1])



    def optimize(self,beam=False):
        global_steps = 0
        for epoch in tqdm(range(epoch_num)):
            if beam:
                self.Logic_form_treess = []
                for sent_tokenizeds in tqdm(self.sent_tokenizedss):
                    Logic_form_trees = []
                    for sent_tokenized in sent_tokenizeds:
                        Logic_form_trees.extend(self.beam_parse(sent_tokenized))
                    Logic_form_trees = list(set(Logic_form_trees))
                    sorted(Logic_form_trees, key=lambda x: np.dot(self.get_feature(x).transpose(), self.theta)[0,0])
                    Logic_form_trees = Logic_form_trees[:min(len(Logic_form_trees), beam_width)]
                    self.Logic_form_treess.append(Logic_form_trees)
                self.parsed_semss = [[parse_sem(tree) for tree in Logic_form_trees] for Logic_form_trees in tqdm(self.Logic_form_treess)]
                self.recursess = [[recurse(parsed_sem) for parsed_sem in parsed_sems] for parsed_sems in tqdm(self.parsed_semss)]

            self.test_on_train(beam)
            self.dump_exp(beam)
            grad = 0
            for index in range(len(self.ys)):
                self.load_one(index)
                self.compute_gradients()
                grad+=self.grad

            self.theta = self.theta + lr / (1 + decay_rate * global_steps) * grad
            global_steps += 1




    def test_on_train(self,beam=False):
        all = 0
        Found = 0
        for index in range(len(self.ys)):
            Ls = self.Logic_form_treess[index]
            if len(Ls)==0:
                continue
            self.load_one(index)
            all+=1
            scores = np.array([np.exp(np.dot(self.get_feature(l).transpose(),self.theta))[0,0] for l in Ls],np.float32)
            select_L = Ls[np.argmax(scores)]
            if beam:
                parsed_sem = parse_sem(select_L)
            else:
                parsed_sem = parse_sem(str(select_L.label()[0].semantics()))
            if parsed_sem == False:
                continue
            LF = recurse(parsed_sem)
            if LF != False:
                try:
                    if LF(self.xs[index])==self.ys[index]:
                        Found+=1
                except:
                    continue
            else:
                continue
        print('Acc',str(Found/all))

    def dump_exp(self,beam=False):
        all = 0
        Found = 0
        dump_L_list = []
        for index in range(len(self.ys)):
            Ls = self.Logic_form_treess[index]
            if len(Ls)==0:
                continue
            self.load_one(index)
            all += 1
            scores = np.array([np.exp(np.dot(self.get_feature(l).transpose(), self.theta))[0, 0] for l in Ls],
                              np.float32)
            select_L = Ls[np.argmax(scores)]
            if beam:
                sem = ''+select_L
            else:
                sem = str(select_L.label()[0].semantics())
            parsed_sem = parse_sem(sem)
            if parsed_sem == False:
                continue
            LF = recurse(parsed_sem)
            if LF != False:
                try:
                    if LF(self.xs[index]) == self.ys[index]:
                        Found += 1
                    dump_L_list.append((index,sem))
                except:
                    continue
            else:
                continue
        with open('exp_dump_raw.pkl','wb') as f:
            pickle.dump(dump_L_list,f)



    def get_lexion(self,lexicon):
        self.raw_lexicon = lexicon

    def load_all_beam(self,sent_tokenizedss,xs,ys):
        self.sent_tokenizedss = sent_tokenizedss
        self.xs= xs
        self.ys = ys

    def beam_parse(self,one_sent_tokenize):
        try:
            self.beam_lexicon = copy.deepcopy(self.raw_lexicon)
            CYK_form = [[[token] for token in one_sent_tokenize]]
            CYK_sem = [[]]
            for layer in range(1,len(one_sent_tokenize)):
                layer_form = []
                layer_sem = []
                lex = lexicon.fromstring(self.beam_lexicon, True)
                parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
                for col in range(0,len(one_sent_tokenize)-layer):
                    form = []
                    sem_temp = []
                    word_index = 0
                    st = col+0
                    ed = st+layer
                    for splt in range(st,ed):
                        words_L = CYK_form[splt-st][st]
                        words_R = CYK_form[ed-splt-1][splt+1]
                        for word_0 in words_L:
                            for word_1 in words_R:
                                try:
                                    for parse in parser.parse([word_0, word_1]):
                                        (token, op) = parse.label()
                                        categ = token.categ()
                                        sem = token.semantics()
                                        word_name = '$Layer{}_Horizon{}_{}'.format(str(layer), str(col),str(word_index))
                                        word_index+=1
                                        entry = "\n\t\t"+word_name+' => '+str(categ)+" {"+str(sem)+"}"
                                        if str(sem)+'_'+str(categ) not in sem_temp:
                                            form.append((parse,word_name,entry,str(sem)))
                                            sem_temp.append(str(sem)+'_'+str(categ))
                                except:
                                    pass
                    form = sorted(form,key=lambda s:np.dot(self.get_feature(s[0]).transpose(),self.theta)[0,0],reverse=True)
                    form = form[:min(len(form),beam_width)]
                    add_form = []
                    for elem in form:
                        parse, word_name, entry,sem_ = elem
                        add_form.append(word_name)
                        self.beam_lexicon = self.beam_lexicon+entry
                        layer_sem.append(sem_)
                    layer_form.append(add_form)
                CYK_form.append(layer_form)
                CYK_sem.append(layer_sem)
            return CYK_sem[-1]
        except:
            return []







class FromExp2LF():
    def __init__(self,beam=False):
        self.beam = beam
        self.parser = Parser(beam)
        self.raw_lexicon = raw_lexicon
    def read_exps(self):
        if os.path.exists(exp_file_dump):
            with open(exp_file_dump, "rb") as f:
                self.new_exps = pickle.load(f)
            return 0
        print('generating '+exp_file)
        nlp = StanfordCoreNLP(corenlp_model_path)
        with open(exp_file,"r") as f:
            exps = json.load(f)
        self.new_exps = []
        for exp in exps:
            exp['sent'] = generate_phrase(exp['sent'],nlp)
            self.new_exps.append(exp)
        nlp.close()
        with open(exp_file_dump,"wb") as f:
            pickle.dump(self.new_exps,f)

    def expand_lexicon(self):
        sents_tokenized = []
        for exp in self.new_exps:
            sent = exp['exp']
            sent_tokenized = print_tokenized(pre_process_sent(sent))
            sents_tokenized.append(sent_tokenized[0])
        quote_words = new_predicate(sents_tokenized)
        self.raw_lexicon = add_new_predicate(quote_words,self.raw_lexicon)

    def parse_all(self):
        if os.path.exists('xs.pkl'):
            print('reading')
            with open('xs.pkl','rb') as f:
                xs = pickle.load(f)
            with open('ys.pkl','rb') as f:
                ys = pickle.load(f)
            with open('lfss.pkl', 'rb') as f:
                Logic_formss = pickle.load(f)
            return Logic_formss,xs,ys
        xs = []
        ys = []
        idx = []
        Logic_formss = []
        lex = lexicon.fromstring(self.raw_lexicon, True)
        parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
        id = -1
        sent_tokenizedss = []
        for exp in tqdm(self.new_exps):
            id+=1
            logic_forms = []

            sent = exp['exp']
            sent_tokenized = print_tokenized(pre_process_sent(sent))
            for index_1 in range(len(sent_tokenized)):
                while [] in sent_tokenized[index_1]:
                    sent_tokenized[index_1].remove([])
                for index_2 in range(len(sent_tokenized[index_1])):
                    if sent_tokenized[index_1][index_2].startswith("'") or sent_tokenized[index_1][index_2].startswith("\""):
                        sent_tokenized[index_1][index_2] = sent_tokenized[index_1][index_2].replace(' ','SPACE')
                        for com in comma_index:
                            sent_tokenized[index_1][index_2] = sent_tokenized[index_1][index_2].replace(com, 'COMMA'+str(comma_index[com]))
                        sent_tokenized[index_1][index_2] = sent_tokenized[index_1][index_2].replace("\"","'")
            if self.beam:
                sent_tokenizedss.append(sent_tokenized)
                xs.append(exp['sent'])
                ys.append(True)
                continue
            for e in sent_tokenized:
                try:
                    parses = list(parser.parse(e))
                    logic_forms += parses
                except:
                    continue
            if len(logic_forms)!=0:
                Logic_formss.append(logic_forms)
                ys.append(True)
                xs.append(exp['sent'])
                idx.append(id)
        if self.beam:
            print(len(xs))
            return sent_tokenizedss,xs,ys
        assert len(xs)==len(ys) and len(xs)==len(Logic_formss)
        print(len(xs))
        print(idx)
        with open('xs.pkl','wb') as f:
            pickle.dump(xs,f)
        with open('ys.pkl','wb') as f:
            pickle.dump(ys,f)
        with open('lfss.pkl', 'wb') as f:
            pickle.dump(Logic_formss, f)

        return Logic_formss, xs, ys

    def Train(self):
        self.read_exps()
        print('reading finish')
        self.expand_lexicon()
        print('expanding finish')
        self.parser.get_lexion(self.raw_lexicon)
        if self.beam:
            sent_tokenizedss, xs, ys = self.parse_all()
            self.parser.load_all_beam(sent_tokenizedss, xs, ys)
            self.parser.optimize(True)
        else:
            Logic_formss, xs, ys = self.parse_all()
            self.parser.load_all(Logic_formss,xs,ys)
            self.parser.optimize()


if __name__=="__main__":
    train = FromExp2LF(True)
    train.Train()