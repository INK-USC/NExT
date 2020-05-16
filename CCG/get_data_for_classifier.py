import pickle
try:
    from CCG.utils import *
except:
    from utils import *
import json
import numpy as np
with open('exp_dump_raw.pkl','rb') as f:
    exp_dump = pickle.load(f)
with open('exp_data_turk.pkl','rb') as f:
    exp_read = pickle.load(f)

exps_used_for_TAC = []

for exp in exp_dump:
    sem = exp[1]
    exp_raw = exp_read[exp[0]]
    try:
        rel = exp_raw['relation']
    except:
        rel = exp_raw['rel']
    new_sent = pre_process_sent(exp_raw['exp'])
    sent_tokenized = print_tokenized(new_sent)[0]
    new_words = []
    for word in sent_tokenized:
        if ((word.startswith("'") and word.endswith("'")) or (word.startswith("\"") and word.endswith("\""))) and not word[1:-1].isdigit() and word[1:-1] not in new_words:
            new_words.append(word[1:-1])
    tacred_tokens = exp_raw['sent'].token
    exps_used_for_TAC.append((tacred_tokens,new_words,exp_raw['exp'],rel,sem))

def posi_add(lis,sublis,pad=None):
    posi = []
    i = 0
    while i <len(lis):
        if lis[i:i+len(sublis)]==sublis:
            posi.extend([1]*len(sublis))
            i+=len(sublis)
        else:
            posi.append(0)
            i+=1
    if pad!=None:
        posi.extend([0]*(pad-len(posi)))
    return posi


#--------get pretrain------------
PT_toks = []
PT_label = []
for elem in exps_used_for_TAC:
    new_words = elem[1]
    toks = elem[0]
    for word in new_words:
        PT_toks.append((word.split(),toks))
        PT_label.append(posi_add(toks,word.split(),110))

PT_label = np.array(PT_label,np.int32)
PT_label.dump('TK_label.npy')
with open('TK_tok_exp.json','w') as f:
    json.dump(PT_toks,f)

#---------fin get pretrain----------


def get_pattern(lis,sublis):
    posi = []
    num_appear = 0
    i = 0
    while i <len(lis):
        if lis[i:i+len(sublis)]==sublis:
            posi.extend([1]*len(sublis))
            i+=len(sublis)
            num_appear+=1
        else:
            posi.append(0)
            i+=1
    if num_appear>1:
        return [0]*len(lis)
    return posi

dump_json = []
for elem in tqdm(exps_used_for_TAC):
    while '' in elem[1]:
        elem[1].remove('')
    lis = np.array([0]*len(elem[0]),np.int32)
    for word in elem[1]:
        arr = np.array(get_pattern(elem[0],word.split()),np.int32)
        lis+=arr
    pat = []
    lis = list(lis)
    for i,judge in enumerate(lis):
        if judge>=1 or elem[0][i].startswith('SUBJ-') or elem[0][i].startswith('OBJ-'):
            pat.append(elem[0][i])
    dump_json.append((elem[3], elem[4], [' '.join(pat)],elem[1]))


with open('explanations.json','w') as f:
    json.dump(dump_json,f)


with open('explanations.json','r') as f:
    data = json.load(f)
pats = []
exp2pat = {}
for i,d in enumerate(data):
    if len(d[2])==1:
        exp2pat[int(i)]=len(pats)
        pats.append([d[0],d[2][0]])
with open('pattern.json','w') as f:
    json.dump(pats,f)
with open('exp2pat.json','w') as f:
    json.dump(exp2pat,f)