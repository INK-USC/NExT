import json
import re
import copy
from nltk.ccg import chart, lexicon
from tqdm import tqdm
from collections import Counter
try:
    from CCG.constant import *
except:
    from constant import *

#------map token to predicates---------
def convert_token2predicate(token):
    num_lexicon = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven']
    num_lexicon_dict = {elem:i+1 for i,elem in enumerate(num_lexicon)}
    if token in token2predicate.keys():
        return token2predicate[token]
    elif token.startswith("\"") and token.endswith("\""):
        return [token]
    elif token.startswith("\'") and token.endswith("\'"):
        return [token]
    elif token.isdigit():
        return ["'"+token+"'"]
    elif token in num_lexicon:
        return ["'"+str(num_lexicon_dict[token])+"'"]
    elif token.lower() in token2predicate.keys():
        return token2predicate[token.lower()]
    else:
        return None

def print_tokenized(sent):
    x = [[]]
    for xx in sent:
        if xx.startswith('$'):
            xxx = [xx]
        else:
            xxx = convert_token2predicate(xx)
        if xxx is not None:
            n = []
            for xxxx in xxx:
                for item in x:
                    cp = copy.copy(item)
                    cp.append(xxxx)
                    n.append(cp)
            x = n
    return x

def pre_process_sent(sent):
    r = []
    for phrase in phrases.keys():
        if phrase in sent:
            sent = sent.replace(phrase, phrases[phrase])
    sent = sent.split()

    for i, j in enumerate(sent):
        if 'SUBJ-' in j:
            sent[i] = 'SUBJ'
        elif 'OBJ-' in j:
            sent[i] = 'OBJ'
    insert_index = []
    for i, j in enumerate(sent):
        if j == '\',' or j=='\",':
            continue
        if j.endswith(','):
            sent[i] = j.rstrip(',')
            insert_index.append(i + len(insert_index))
    for i in insert_index:
        sent.insert(i + 1, ',')

    length = len(sent)
    point = 0
    while point <= length-1:
        if sent[point].startswith('\'') or sent[point].startswith('\"'):
            r_d = []
            r_d.append(sent[point])
            if (sent[point].endswith('\'') or sent[point].endswith('\"')) and sent[point] != '\'\'':
                r.append(r_d[0])
                point += 1
            else:
                point += 1
                while not (sent[point].endswith('\'') or sent[point].endswith('\"')):
                    r_d.append(sent[point])
                    point += 1
                r_d.append(sent[point])
                point += 1
                r.append(" ".join(r_d))
        else:
            r.append(sent[point])
            point += 1
    return r

#---add new predicate(quote word)---
def new_predicate(sents_tokenized):  #sents_tokenized is U{sent_tokenized[0]}
    quote_words = []
    for sent in sents_tokenized:
        sent_split = sent
        for word in sent_split:
            if ((word.startswith('"') and word.endswith('"')) or (word.startswith("'") and word.endswith("'"))) and word.replace("\"","'") not in quote_words:
                quote_words.append(word.replace("\"","'"))
    return quote_words

def add_new_predicate(quote_words,raw_lexicon=raw_lexicon):
    for word in quote_words:
        replace_word = word.replace(' ','SPACE')
        for com in comma_index:
            if com in word:
                replace_word = replace_word.replace(com,'COMMA'+str(comma_index[com]))
        if replace_word[1:-1].isdigit():
            raw_lexicon = raw_lexicon + "\n\t\t" + replace_word + " => NP/NP {\\x.'@Num'(" + replace_word + ",x)}" + "\n\t\t" + replace_word + " => N/N {\\x.'@Num'(" + replace_word + ",x)}"
            raw_lexicon = raw_lexicon + "\n\t\t" + replace_word + " => NP {" + replace_word + "}" + "\n\t\t" + replace_word + " => N {" + replace_word + "}"
            raw_lexicon = raw_lexicon + "\n\t\t" + replace_word + " => PP/PP/NP/NP {\\x y F.'@WordCount'('@Num'(" + replace_word + ",x),y,F)}" + "\n\t\t" + replace_word + " => PP/PP/N/N {\\x y F.'@WordCount'('@Num'(" + replace_word + ",x),y,F)}"
        else:
            raw_lexicon = raw_lexicon+"\n\t\t"+replace_word+" => NP {"+replace_word+"}"+"\n\t\t"+replace_word+" => N {"+replace_word+"}"
    return raw_lexicon

#-----give logic form's rule feature-----
def recurse_print(tree,all):
    cates = [str(tree.label()[0].categ())]
    for child in tree:
        (token, op) = child.label()
        cates.append(str(token.categ()))
    assert len(cates)<=3
    if cates not in all:
        all.append(cates)
    for child in tree:
        if child.label()[1]!='Leaf':
            recurse_print(child,all)

#---from str to logic form
def parse_sem(sem):
    sem_split = re.split(',|(\()',sem)
    while None in sem_split:
        sem_split.remove(None)
    new_sem = []
    for i,w in enumerate(sem_split):
        if w=='(':
            new_sem.insert(-1,'(')
        else:
            new_sem.append(sem_split[i])
    sem_ = ''
    for i,s in enumerate(new_sem):
        sem_replace = s.replace('SPACE', ' ')
        for com in comma_index:
            sem_replace = sem_replace.replace('COMMA' + str(comma_index[com]), com)
        if sem_replace.startswith("'"):
            if sem_replace.endswith(")"):
                posi = len(sem_replace)-1
                while sem_replace[posi]==")":
                    posi-=1
                assert sem_replace[posi]=="\'"
            else:
                posi = len(sem_replace)-1
                assert sem_replace[posi] == "\'"

            sem_replace = sem_replace[0]+sem_replace[1:posi].replace('\'','\\\'')+sem_replace[posi:]
        if new_sem[i-1]=='(':
            sem_=sem_+sem_replace
        else:
            sem_ = sem_+','+sem_replace

    try:
        em = ('.root',eval(sem_[1:]))
        return em
    except:
        return False


#----from logic form to labeling function
def recurse(sem):
    try:
        if isinstance(sem, tuple):
            op = ops[sem[0]]
            args = [recurse(arg) for arg in sem[1:]]
            if False in args:
                return False
            return op(*args) if args else op
        else:
            if sem in syntax:
                return syntax[sem]
            else:
                return sem
    except:
        return False

def recurse_soft(sem):
    try:
        if isinstance(sem, tuple):
            op = ops_soft[sem[0]]
            args = [recurse_soft(arg) for arg in sem[1:]]
            if False in args:
                return False
            return op(*args) if args else op
        else:
            if sem in syntax:
                return syntax[sem]
            else:
                return sem
    except:
        return False

#----Preprocess for verification sentence
def generate_phrase(sentence,nlp):
    split_sentence = sentence.split()
    subj = None
    obj = None
    for i in range(len(split_sentence)):
        if split_sentence[i].startswith('OBJ-'):
            obj = ''.join(split_sentence[i])
            split_sentence[i] = 'OBJ'
        if split_sentence[i].startswith('SUBJ-'):
            subj = ''.join(split_sentence[i])
            split_sentence[i] = 'SUBJ'
    assert subj!=None and obj!=None
    sentence_new = token2str(split_sentence)
    combine = list(zip(*nlp.ner(sentence_new)))
    ners = list(combine[1])
    split_sent = list(combine[0])

    obj_lis = []
    subj_lis = []
    for i, word in enumerate(split_sent):
        if word=='OBJ':
            obj_lis.append(i)
        if word=='SUBJ':
            subj_lis.append(i)
    if obj_lis[0] < subj_lis[0]:
        posi = [[obj_lis[0], obj_lis[-1]], [subj_lis[0], subj_lis[-1]]]
    else:
        posi = [[subj_lis[0], subj_lis[-1]], [obj_lis[0], obj_lis[-1]]]

    new_split = split_sent[0:posi[0][0] + 1] + split_sent[posi[0][1] + 1:posi[1][0] + 1]
    new_ners = ners[0:posi[0][0] + 1] + ners[posi[0][1] + 1:posi[1][0] + 1]
    assert 'SUBJ' in new_split and 'OBJ' in new_split
    for i in range(len(new_split)):
        if new_split[i]=='SUBJ':
            new_split[i]=subj
        if new_split[i]=='OBJ':
            new_split[i]=obj

    if posi[1][1] < len(split_sent) - 1:
        new_split += split_sent[posi[1][1] + 1:]
        new_ners += ners[posi[1][1] + 1:]

    obj_posi = -1
    subj_posi = -1
    for i,word in enumerate(new_split):
        if word.startswith('SUBJ-'):
            subj_posi = i
        if word.startswith('OBJ-'):
            obj_posi = i

    return Phrase(new_split,new_ners,(subj_posi,obj_posi),sentence)

class Phrase():
    def __init__(self,split_token,ner,posi,sentence,idx=None):
        self.sentence = sentence
        self.token = split_token
        self.ner = ner
        self.subj_posi,self.obj_posi = posi
        self.subj = self.token[self.subj_posi]
        self.obj = self.token[self.obj_posi]
        self.idx =  idx
    def get_mid(self):
        st = min(self.subj_posi,self.obj_posi)+1
        ed = max(self.subj_posi,self.obj_posi)
        midphrase = token2str(self.token[st:ed])
        midner = self.ner[st:ed]
        return {'word':midphrase,'NER':midner,'tokens':self.token[st:ed],'position':(st,ed)}

    def get_other_posi(self,LoR,XoY):
        assert LoR == 'Left' or LoR == 'Right' or LoR=='Range'
        assert XoY == 'X' or XoY == 'Y'

        if XoY == 'X':
            split_posi = self.subj_posi
        else:
            split_posi = self.obj_posi

        if LoR=='Left':
            phrase_ = token2str(self.token[:split_posi])
            phrase_ner = self.ner[:split_posi]
            phrase_tokens = self.token[:split_posi]
            posi = (0,split_posi)
        elif LoR=='Right':
            phrase_ = token2str(self.token[split_posi+1:])
            phrase_ner = self.ner[split_posi+1:]
            phrase_tokens = self.token[split_posi+1:]
            posi = (split_posi+1,len(self.token))
        else:
            phrase_ = self.sentence
            phrase_ner = self.ner
            phrase_tokens = self.token
            return {'word':phrase_,'NER':phrase_ner,'tokens':phrase_tokens,'POSI':split_posi}

        return {'word':phrase_,'NER':phrase_ner,'tokens':phrase_tokens,'position':posi}

    def with_(self,XoY,SoE,substring):
        assert XoY == 'X' or XoY == 'Y'
        assert SoE == 'starts' or SoE == 'ends'

        if XoY=='X':
            word = self.subj
        else:
            word = self.obj

        if SoE=='starts':
            return word.startswith(substring)
        else:
            return word.endswith(substring)
#--------some tools-------------


#---token2str----
def token2str(tokens):
    string = ''
    for token in tokens:
        string = string+" "+token
    return string[1:]


def find_text(string,substring):
    raw_index = string.find(substring)
    if raw_index==-1:
        return []
    else:
        index_list = []
        index = 0
        while raw_index!=-1:
            index = index+raw_index
            index_list.append(index)
            index +=1
            if index==len(string):
                break
            raw_index = string[index:].find(substring)
        return index_list

#-----feature extractor(ABANDON NOW!!!!!) DO NOT USE!-------

def rule_features():
    lis = []
    pass_num = 0
    with open('exp_data_new.json',"r") as f:
        exps = json.load(f)
    for exp_num in tqdm(range(len(exps))):
        exp = exps[exp_num]
        try:
            e = exp['exp']
            new_sent = pre_process_sent(e)
            sent_tokenized = print_tokenized(new_sent)
            raw_lexicon = add_new_predicate(sent_tokenized)
            lex = lexicon.fromstring(raw_lexicon, True)
            parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
            for parse in parser.parse(sent_tokenized[0].split()):
                recurse_print(parse,lis)
        except:
            pass_num+=1
            pass
    with open('combrules.json',"w") as f:
        json.dump(lis,f)



def collect_features(semantics, features):
    if isinstance(semantics, tuple):

        for child in semantics[1:]:
            if isinstance(child, tuple) and child[0] != semantics[0]:
                features.append((semantics[0],child[0]))
            collect_features(child, features)


def operator_precedence_features():
    features = []
    pass_num = 0

    with open('exp_data_turk_new.json',"r") as f:
        exps = json.load(f)
    for exp_num in tqdm(range(len(exps))):
        exp = exps[exp_num]
        if exp_num==int(len(exps)/4) or exp_num==int(len(exps)/2) or exp_num==int(len(exps)/4*3):
            print(len(features))
        try:
            e = exp['exp']
            new_sent = pre_process_sent(e)
            sent_tokenized = print_tokenized(new_sent)
            quote_words =  new_predicate(sent_tokenized)
            raw_lexicon = add_new_predicate(quote_words)
            lex = lexicon.fromstring(raw_lexicon, True)
            parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
            for sent in sent_tokenized:
                for i,parse in enumerate(list(parser.parse(sent))):
                    sem = parse_sem(str(parse.label()[0].semantics()))
                    if sem!=False:
                        collect_features(sem, features)
        except:
            pass_num+=1
    counter = Counter(features)
    prt = list(list(zip(*list(counter.most_common())))[0])
    print(prt)
    print(len(prt))
    print(pass_num,len(exps))
