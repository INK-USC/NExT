import json
import random
import numpy as np
data = json.load(open('train.json', 'r'))

def random_pred(seq1, seq2):
    sum = 0
    for i in seq1:
        sum += i
    assert sum == 1
    assert len(seq1) == len(seq2)
    pp = random.randint(1, 100)
    flag = 0
    for i,j in enumerate(seq1):
        for k in range(len(seq1) - i - 1):
            seq1[len(seq1) - i -1] += seq1[k]
    for i,j in enumerate(seq1):
        seq1[i] *= 100
    for j, i in enumerate(seq1):
        if pp >= i:
            continue
        else:
            flag = j
            break
    return seq2[j]

tokens = []
max_token_len = 0
for item in data:
    if max_token_len < len(item['token']):
        max_token_len = len(item['token'])

low_freq_words = []
words = []
all_words = {}

'''
all_words = json.load(open('dict', 'r'))
num = 0
for i in all_words:
    if all_words[i] <= 10:
        num += 1
'''
ll = []
a = 0
b = 0
c = 0

for i, item in enumerate(data):
    lll = []
    sent = item['token']
    subj_s = item['subj_start']
    obj_s = item['obj_start']
    subj_e = item['subj_end']
    obj_e = item['obj_end']
    subj_type = item['subj_type']
    obj_type = item['obj_type']
    subj = 'SUBJ-'+subj_type
    obj = 'OBJ-'+obj_type
    for posi in range(subj_s,subj_e+1):
        sent[posi] = subj
    for posi in range(obj_s,obj_e+1):
        sent[posi] = obj
    st_1 = min(subj_s,obj_s)
    ed_1 = min(subj_e,obj_e)
    st_2 = max(subj_s,obj_s)
    ed_2 = max(subj_e,obj_e)
    len_mid = len(sent[ed_1:st_2])
    sent = sent[:st_1]+sent[ed_1:st_2]+sent[ed_2:]

    seq_len = len(sent)
    pp = random.randint(1, 100)
    pattern_num = random_pred([0.2, 0.3, 0.3, 0.2], [1, 2, 3, 4])
    if pattern_num == 1:
        a += 1
    elif pattern_num == 2:
        b += 1
    else:
        c += 1
    pattern = []
    if seq_len <= pattern_num:
        continue
    starting = random.randint(0, seq_len-pattern_num)
    for j in range(pattern_num):
        lll.append(starting+j)
        pattern.append(sent[starting+j])
    for j in range(len(sent) - pattern_num + 1):
        flag = 1
        for k in range(pattern_num):
            if sent[j+k] not in pattern:
                flag = 0
                break
        if flag == 1:
            for k in range(pattern_num):
                if j+k not in lll:
                    lll.append(j+k)

    tokens.append((pattern, sent))
    ll.append(lll)
print(a)
print(b)
print(c)
pattern_mask = np.zeros((len(ll), 110), dtype = int)
for i in range(len(tokens)):
    for j in ll[i]:
        pattern_mask[i][j] = 1

np.save("PT_pattern_mask.npy", pattern_mask)
json.dump(tokens, open('PT_toks.json', 'w'), indent = 4)
