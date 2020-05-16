import json
import pickle
from CCG.utils import Phrase

def handle(data,j):
    ner = data['stanford_ner']
    token = data['token']

    subj_start = data['subj_start']
    subj_end = data['subj_end']
    obj_start = data['obj_start']
    obj_end = data['obj_end']
    subj_type = data['subj_type']
    obj_type = data['obj_type']
    subj = 'SUBJ-'+subj_type
    obj = 'OBJ-'+obj_type

    for posi in range(subj_start,subj_end+1):
        token[posi] = subj
    for posi in range(obj_start,obj_end+1):
        token[posi] = obj

    st_1 = min(subj_start,obj_start)
    ed_1 = min(subj_end,obj_end)
    st_2 = max(subj_start,obj_start)
    ed_2 = max(subj_end,obj_end)


    len_mid = len(token[ed_1:st_2])
    token = token[:st_1]+token[ed_1:st_2]+token[ed_2:]
    sentence = ' '.join(token)
    ner = ner[:st_1]+ner[ed_1:st_2]+ner[ed_2:]
    if st_1==subj_start:
        new_subj_posi = st_1
        new_obj_posi = st_1+len_mid
    else:
        new_obj_posi = st_1
        new_subj_posi = st_1+len_mid

    assert token[new_subj_posi].startswith('SUBJ-') and token[new_obj_posi].startswith('OBJ-')
    return (data['relation'],Phrase(token,ner,(new_subj_posi,new_obj_posi),sentence,j))



if __name__=="__main__":
    files = ['train.json','dev.json','test.json']
    dump_files= ['train.pkl','dev.pkl','test.pkl']
    for i,file in enumerate(files):
        dump_data = []
        with open(file,"r") as f:
            data = json.load(f)
        for j,sent in enumerate(data):
            dump_data.append(handle(sent,j))
        with open(dump_files[i],'wb') as f:
            pickle.dump(dump_data,f)