# from CCG.constant import *
from functools import reduce
import tensorflow as tf

unmatch_count_score = 0.5
unmatch_count_dist = 7
unmatch_match_score = 0.5
unmatch_match_dist = 7


Selection = ['NER']
compare_soft={                   #eq mt(more than) lt(less than) nmt(no more than) nlt(no less than)
    'eq':lambda a,b:tf.math.maximum(tf.cast(tf.equal(a,b),tf.float32),tf.cast(tf.logical_and(tf.greater_equal(a,b-unmatch_count_dist),tf.less_equal(a,b+unmatch_count_dist)),tf.float32)*unmatch_count_score),
    'mt':lambda a,b:tf.math.maximum(tf.cast(tf.greater(a,b),tf.float32),tf.cast(tf.greater(a,b-unmatch_count_dist),tf.float32)*unmatch_count_score),
    'lt':lambda a,b:tf.math.maximum(tf.cast(tf.less(a,b),tf.float32),tf.cast(tf.less(a,b+unmatch_count_dist),tf.float32)*unmatch_count_score),
    'nmt':lambda a,b:tf.math.maximum(tf.cast(tf.less_equal(a,b),tf.float32),tf.cast(tf.less_equal(a,b+unmatch_count_dist),tf.float32)*unmatch_count_score),
    'nlt':lambda a,b:tf.math.maximum(tf.cast(tf.greater_equal(a,b),tf.float32),tf.cast(tf.greater_equal(a,b-unmatch_count_dist),tf.float32)*unmatch_count_score),
}

#c: Tensor [seqlen seqlen 2 2] : sentence,ner,subj_posi,obj_posi,subj,obj
def get_mid(attr,mask_mat,c):
    subj_posi = c[:,-4:-3]
    obj_posi = c[:,-3:-2]
    seqlen = (c.get_shape()[1]-4)//2
    tokens = c[:,0:seqlen]
    ner = c[:,seqlen:seqlen*2]
    st_posi = tf.math.minimum(subj_posi,obj_posi)
    ed_posi = subj_posi+obj_posi-st_posi
    mask = tf.gather_nd(mask_mat,tf.concat([st_posi+1,ed_posi-1],axis=1))
    ner = ner*mask
    tokens = tokens*mask
    res = tf.cond(attr=='NER',lambda:ner,lambda:tokens)
    return res

def get_other_posi(POSI,attr,arg,mask_mat,c):
    assert POSI=='Left' or POSI=='Right'
    batch_size = c.get_shape()[0]
    subj_posi = c[:, -4:-3]
    obj_posi = c[:, -3:-2]
    seqlen = (c.get_shape()[1] - 4) // 2
    tokens = c[:, 0:seqlen]
    ner = c[:, seqlen:seqlen * 2]
    posi = tf.cond(arg=='ArgY',lambda:obj_posi,lambda:subj_posi)

    mask = tf.cond(POSI=='Left',lambda:tf.gather_nd(mask_mat,tf.concat([0*posi,posi-1],axis=1)),
                   lambda:tf.gather_nd(mask_mat,tf.concat([posi+1,(0*posi+1)*(seqlen-1)],axis=1)))

    new_ner = mask*ner
    new_tokens = mask*tokens
    res = tf.cond(attr=='NER',lambda:new_ner,lambda:new_tokens)
    return res

def get_range(attr,arg,range_,mask_mat,c):
    subj_posi = c[:, -4:-3]
    obj_posi = c[:, -3:-2]
    seqlen = (c.get_shape()[1] - 4) // 2
    tokens = c[:, 0:seqlen]
    ner = c[:, seqlen:seqlen * 2]
    posi = tf.cond(arg == 'ArgY', lambda: obj_posi, lambda: subj_posi)
    res = tf.cond(attr == 'NER', lambda: ner, lambda: tokens)
    mask = tf.gather_nd(mask_mat,tf.concat([tf.math.maximum(0,posi-range_),tf.math.minimum(seqlen-1,posi+range_)],axis=1))
    res = res*mask
    return res

def In(w,seq):
    assert w in NER2ID
    idx = NER2ID[w]
    eq = tf.equal(seq,idx)
    sum = tf.reshape(tf.reduce_sum(eq,axis=1),[1,-1])
    return tf.cast(sum,tf.bool)


def merge_soft(x,y):
    if type(x)!= tuple:
        x = [x]
    if type(y)!= tuple:
        y = [y]
    x = list(x)
    y = list(y)
    return tuple(x+y)

#function for $Is
def IsFunc_soft(ws,ps,label_mat,keyword_dict,mask_mat,c):
    if isinstance(ps,tuple):
        bool_list  = []
        for p in ps:
            if isinstance(ws, tuple):
                if ws[0] in Selection:
                    bool_list.append(p(ws)(label_mat,keyword_dict,mask_mat)(c))
                else:
                    bool_list.append(tf.maximum(sum([p(w)(label_mat,keyword_dict,mask_mat)(c) for w in ws])-len(ws)+1,0.0))
            else:
                bool_list.append(p(ws)(label_mat,keyword_dict,mask_mat)(c))
        return tf.maximum(sum(bool_list)-len(bool_list)+1,0.0)

    if isinstance(ws,tuple):
        if ws[0] in Selection:
            return ps(ws)(label_mat,keyword_dict,mask_mat)(c)
        else:
            return tf.maximum(sum([ps(w)(label_mat,keyword_dict,mask_mat)(c) for w in ws])-len(ws)+1,0.0)
    else:
        return ps(ws)(label_mat,keyword_dict,mask_mat)(c)

#function for @Left and @Right
def at_POSI_soft(POSI,ws,arg,label_mat,keyword_dict,mask_mat,c,option=None):
    if isinstance(ws,tuple):
        if ws[0] in Selection:
            assert POSI=='Left' or POSI=='Right' or POSI=='Range'
            if POSI=='Left' or POSI=='Right':
                score_raw = tf.cast(In(ws[1],get_other_posi(POSI,ws[0],arg,mask_mat,c)),tf.float32)
                return score_raw
            else:
                score_raw =  tf.cast(In(ws[1], get_range(ws[0], arg, option['range'],mask_mat,c)), tf.float32)
                return score_raw
        else:
            bool_list = []
            for w in ws:
                bool_list.append(at_POSI_0_soft(POSI,arg,w,label_mat,keyword_dict,mask_mat,c,option))
            return tf.maximum(sum(bool_list)-len(bool_list)+1,0.0)
    else:
        return at_POSI_0_soft(POSI,arg,ws,label_mat,keyword_dict,mask_mat,c,option)

#function for @Left0 and @Right0
def at_POSI_0_soft(POSI,arg,w,label_mat,keyword_dict,mask_mat,c,option=None):
    if arg not in ['ArgX','ArgY']:
        w,arg = arg,w
        if POSI == 'Left':
            POSI = 'Right'
        elif POSI == 'Right':
            POSI = 'Left'

    if isinstance(w,tuple) and w[0] not in Selection:
        return tf.maximum(sum([at_POSI_0_soft(POSI,arg,ww,label_mat,keyword_dict,mask_mat,c,option) for ww in w])-len(w)+1,0.0)


    if option==None:
        option = {'attr': 'word', 'range': -1, 'numAppear':1,'cmp':'nlt','onlyCount':False}  #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        assert POSI == 'Left' or POSI == 'Right' or POSI == 'Range'
        if POSI == 'Left' or POSI == 'Right':
            score_raw = tf.cast(In(ws[1], get_other_posi(POSI, ws[0], arg, mask_mat,c)), tf.float32)
            return score_raw
        else:
            score_raw = tf.cast(In(ws[1], get_range(ws[0], arg, option['range'], mask_mat,c)), tf.float32)
            return score_raw
    else:                                                                                   #For now,option['attr']==tokens if and only if 'right before' is used or onlyCount==True, otherwise ==word
        if option == 'Direct':
            w_split = w.split()
            while '' in w_split:
                w_split.remove('')
            range_ = len(w_split)

            subj_posi = c[:, -4:-3]
            obj_posi = c[:, -3:-2]
            seqlen = (c.get_shape()[1] - 4) // 2

            if arg=='ArgY':
                arg_posi = obj_posi
            else:
                arg_posi = subj_posi

            if POSI == 'Left':
                st = tf.math.maximum(0, arg_posi - range_)
                position = tf.concat([st,arg_posi-1],axis=1)
                unmatch_position = tf.concat([tf.math.maximum(0,arg_posi - range_ - unmatch_match_dist), arg_posi - 1], axis=1)
            elif POSI == 'Right':
                st = arg_posi
                ed = tf.math.minimum(seqlen - 1, arg_posi+range_)
                position = tf.concat([st+1,ed],axis=1)
                unmatch_position = tf.concat([st + 1, tf.math.minimum(seqlen - 1,st+range_+unmatch_match_dist)], axis=1)
            else:
                raise ValueError
            if w in ['ArgX','ArgY']:
                if w=='ArgX':
                    tar = c[:,-4:-3]
                    src = c[:,-3:-2]
                else:
                    src = c[:,-4:-3]
                    tar = c[:,-3:-2]
                if POSI=='Left':
                    score_raw = tf.reshape(tf.cast(tf.equal(src-tar,1),tf.float32),[1,-1])
                    return score_raw
                else:
                    score_raw = tf.reshape(tf.cast(tf.equal(tar-src, 1),tf.float32),[1,-1])
                    return score_raw
            else:
                mask_raw = tf.cast(tf.gather_nd(mask_mat, position), tf.float32)
                mask_raw = mask_raw+(tf.cast(tf.gather_nd(mask_mat, unmatch_position), tf.float32)-mask_raw)*unmatch_match_score
                return tf.reshape(tf.reduce_max(label_mat[:, :, keyword_dict[w]] * (mask_raw),axis=1), [1, -1])

        if option['range']==-1:
            assert POSI=='Left' or POSI=='Right'
            batch_size = c.get_shape()[0]
            subj_posi = c[:, -4:-3]
            obj_posi = c[:, -3:-2]
            seqlen = (c.get_shape()[1] - 4) // 2
            if arg == 'ArgY':
                arg_posi = obj_posi
            else:
                arg_posi = subj_posi

            if POSI=='Left':
                position = tf.concat([0*arg_posi, arg_posi - 1], axis=1)
                unmatch_position = position
            else:
                position = tf.concat([arg_posi + 1, (0*arg_posi+1) * (seqlen - 1)],axis=1)
                unmatch_position = position

        else:                                                                               #For now, if range!=-1 then attr == 'tokens'
            subj_posi = c[:, -4:-3]
            obj_posi = c[:, -3:-2]
            seqlen = (c.get_shape()[1] - 4) // 2
            if arg == 'ArgY':
                arg_posi = obj_posi
            else:
                arg_posi = subj_posi

            range_ = option['range']

            if POSI == 'Left':
                st = tf.math.maximum(0, arg_posi - range_)
                position = tf.concat([st, arg_posi-1],axis=1)
                unmatch_position = tf.concat([tf.math.maximum(0, arg_posi - range_ - unmatch_match_dist), arg_posi - 1],axis=1)
            elif POSI=='Right':
                st = arg_posi
                ed = tf.math.minimum(seqlen - 1, arg_posi+range_)
                position = tf.concat([arg_posi+1, ed],axis=1)
                unmatch_position = tf.concat([st + 1, tf.math.minimum(seqlen - 1, st + range_ + unmatch_match_dist)],axis=1)
            else:
                st = tf.math.maximum(0,arg_posi-range_)
                ed = tf.math.minimum(seqlen-1,arg_posi+range_)
                position=tf.concat([st,ed],axis=1)
                unmatch_position = tf.concat([tf.math.maximum(0, arg_posi - range_ - unmatch_match_dist), tf.math.minimum(seqlen - 1, st + range_ + unmatch_match_dist)], axis=1)

        if option['onlyCount']:
            score_raw = tf.reshape(tf.cast(compare_soft[option['cmp']](position[:,1]-position[:,0],option['numAppear']),tf.float32),[1,-1])
            return score_raw
        else:
            assert option['cmp']=='nlt' and option['numAppear']==1
            if w in ['ArgX','ArgY']:
                if w=='ArgX':
                    tar = c[:,-4]
                else:
                    tar = c[:,-3]
                score_raw = tf.reshape(tf.cast(tf.logical_and(tar>=position[:,0],tar<=position[:,1]),tf.float32),[1,-1])
                return score_raw
            else:
                mask_raw = tf.cast(tf.gather_nd(mask_mat, position),tf.float32)
                mask_raw = mask_raw+(tf.cast(tf.gather_nd(mask_mat, unmatch_position),tf.float32)-mask_raw)*unmatch_match_score
                return tf.reshape(tf.reduce_max(label_mat[:, :, keyword_dict[w]] * (mask_raw),axis=1), [1, -1])



#function for @Between

def at_between_soft(w,label_mat,keyword_dict,mask_mat,c,option=None):
    if option==None:
        option = {'attr': 'word', 'numAppear':1,'cmp':'nlt','onlyCount':False}                #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        score_raw = tf.cast(In(w[1],get_mid(w[0],mask_mat,c)),tf.float32)
        return score_raw
    else:                                                                                   #For now,option['attr']==tokens if and only if  onlyCount==True, otherwise ==word
        if option['onlyCount']:
            score_raw =  tf.reshape(tf.cast(compare_soft[option['cmp']](tf.abs(c[:,-4]-c[:,-3])-1,option['numAppear']),tf.float32),[1,-1])
            return score_raw
        else:
            assert option['cmp'] == 'nlt' and option['numAppear'] == 1
            l_posi = tf.math.minimum(c[:,-4],c[:,-3])
            g_posi = c[:,-4]+c[:,-3]-l_posi
            if w in ['ArgX','ArgY']:
                if w=='ArgX':
                    tar = c[:,-4]
                else:
                    tar = c[:,-3]
                score_raw = tf.reshape(tf.cast(tf.logical_and(tar>l_posi,tar<g_posi),tf.float32),[1,-1])
                return score_raw
            else:
                l_posi = tf.reshape(l_posi,[-1,1])+1
                g_posi = tf.reshape(g_posi, [-1, 1])-1
                position = tf.concat([l_posi, g_posi],axis=1)
                mask_raw = tf.cast(tf.gather_nd(mask_mat, position),tf.float32)
                seqlen = (c.get_shape()[1] - 4) // 2
                return tf.reshape(tf.reduce_max(label_mat[:, :, keyword_dict[w]] * (mask_raw),axis=1), [1, -1])



#function for counting
def at_lessthan_soft(funcx,nouny,w,label_mat,keyword_dict,mask_mat,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'lt','onlyCount':onlyCount})(label_mat,keyword_dict,mask_mat)(c)                #There are less than 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(label_mat,keyword_dict,mask_mat)(c)      #the word 'x' is less than 3 words before OBJ

def at_atmost_soft(funcx,nouny,w,label_mat,keyword_dict,mask_mat,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'nmt','onlyCount':onlyCount})(label_mat,keyword_dict,mask_mat)(c)                #There are at most 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(label_mat,keyword_dict,mask_mat)(c)      #the word 'x' is at most 3 words before OBJ

def at_atleast_soft(funcx,nouny,w,label_mat,keyword_dict,mask_mat,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'nlt','onlyCount':onlyCount})(label_mat,keyword_dict,mask_mat)(c)             #There are at least 3 words before OBJ
    else:
        return funcx(w,{'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(label_mat,keyword_dict,mask_mat)(c)      #the word 'x' is no less than 3 words before OBJ

def at_morethan_soft(funcx,nouny,w,label_mat,keyword_dict,mask_mat,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'mt','onlyCount':onlyCount})(label_mat,keyword_dict,mask_mat)(c)                #There are more than 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(label_mat,keyword_dict,mask_mat)(c)    #the word 'x' is more than 3 words before OBJ


#function for @In0

def at_In0_soft(arg,w,label_mat,keyword_dict,mask_mat,c):
    assert arg=='Sentence'                                                                                                          #Right?
    if isinstance(w,tuple):
        seqlen = (c.get_shape()[1] - 4) // 2
        score_raw = tf.cast(In(w,c[:,seqlen:seqlen*2]),tf.float32)
        return score_raw
    else:
        seqlen = (c.get_shape()[1] - 4) // 2
        return tf.reshape(tf.reduce_max(label_mat[:, :, keyword_dict[w]], axis=1), [1, -1])
        # return w in c.sentence

def at_WordCount_soft(nounNum,nouny,F,label_mat,keyword_dict,mask_mat,c):
    if isinstance(nouny,tuple):
        return tf.maximum(tf.maximum(sum([F(noun, option={'attr': 'word', 'range': -1, 'numAppear': 1, 'cmp': 'nlt', 'onlyCount': False})(label_mat,keyword_dict,mask_mat)(c) for noun in nouny])-len(nouny)+1,0.0)+F(nouny[0],option={'attr': 'tokens','range': -1,'numAppear':sum([len(noun.split()) for noun in nouny]),'cmp': 'eq','onlyCount': True})(label_mat,keyword_dict,mask_mat)(c)-1,0.0)
    else:
        return tf.maximum(F(nouny, option={'attr':'word','range':-1,'numAppear':1,'cmp':'nlt','onlyCount':False})(label_mat,keyword_dict,mask_mat)(c)+F(nouny, option={'attr':'tokens','range':-1,'numAppear':len(nouny.split()),'cmp':'eq','onlyCount':True})(label_mat,keyword_dict,mask_mat)(c)-1,0.0)

ops_soft={
    ".root": lambda xs: lambda label_mat,keyword_dict,mask_mat:lambda c: tf.maximum(sum([x(label_mat,keyword_dict,mask_mat)(c) for x in xs])-len(xs)+1,0.0) if type(xs) == tuple else xs(label_mat,keyword_dict,mask_mat)(c),
    "@Word": lambda x: x,
    "@Is": lambda ws, p: lambda label_mat,keyword_dict,mask_mat:lambda c: IsFunc_soft(ws, p,label_mat,keyword_dict,mask_mat, c),
    "@between": lambda a: lambda w, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: at_between_soft(w, label_mat,keyword_dict,mask_mat,c, option),
    "@And": lambda x, y: merge_soft(x, y),
    "@Num": lambda x, y: {'attr': y, "num": int(x)},
    "@LessThan": lambda funcx, nouny: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: at_lessthan_soft(funcx, nouny, w, label_mat,keyword_dict,mask_mat,c),
    "@AtMost": lambda funcx, nouny: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: at_atmost_soft(funcx, nouny, w, label_mat,keyword_dict,mask_mat,c),
    "@AtLeast": lambda funcx, nouny: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: at_atleast_soft(funcx, nouny, w, label_mat,keyword_dict,mask_mat,c),
    "@MoreThan": lambda funcx, nouny: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: at_morethan_soft(funcx, nouny, w, label_mat,keyword_dict,mask_mat,c),
    "@WordCount": lambda nounNum, nouny, F:lambda useless: lambda label_mat,keyword_dict,mask_mat:lambda c: at_WordCount_soft(nounNum,nouny,F,label_mat,keyword_dict,mask_mat,c),

    "@NumberOf": lambda x, f: [x, f],
    "@LessThan1": lambda nounynum: lambda x: lambda label_mat,keyword_dict,mask_mat:lambda c: at_lessthan_soft(x[1], {'attr': x[0], "num": int(nounynum)}, 'There',label_mat,keyword_dict,mask_mat,c),
    "@AtMost1": lambda nounynum: lambda x: lambda label_mat,keyword_dict,mask_mat:lambda c: at_atmost_soft(x[1], {'attr': x[0], "num": int(nounynum)}, 'There', label_mat,keyword_dict,mask_mat,c),
    "@AtLeast1": lambda nounynum: lambda x: lambda label_mat,keyword_dict,mask_mat:lambda c: at_atleast_soft(x[1], {'attr': x[0], "num": int(nounynum)}, 'There',label_mat,keyword_dict,mask_mat,c),
    "@MoreThan1": lambda nounynum: lambda x: lambda label_mat,keyword_dict,mask_mat:lambda c: at_morethan_soft(x[1], {'attr': x[0], "num": int(nounynum)}, 'There',label_mat,keyword_dict,mask_mat,c),

    "@In0": lambda arg: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: at_In0_soft(arg, w, label_mat,keyword_dict,mask_mat,c),

    "@By": lambda x, f, z: lambda label_mat,keyword_dict,mask_mat:lambda c: f(x, {'attr': z['attr'], 'range': z['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': False})(label_mat,keyword_dict,mask_mat)(c),

    "@Left0": lambda arg: lambda w, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: at_POSI_0_soft('Left', arg, w, label_mat,keyword_dict,mask_mat,c, option),

    "@Right0": lambda arg: lambda w, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: at_POSI_0_soft('Right', arg, w, label_mat,keyword_dict,mask_mat,c, option),

    "@Range0": lambda arg: lambda w, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: at_POSI_0_soft('Range', arg, w, label_mat,keyword_dict,mask_mat,c, option),

    "@Left": lambda arg, ws, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: at_POSI_soft('Left', ws, arg, label_mat,keyword_dict,mask_mat,c, option),
    "@Right": lambda arg, ws, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: at_POSI_soft('Right', ws, arg, label_mat,keyword_dict,mask_mat,c, option),

    "@Direct": lambda func: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: func(w, 'Direct')(label_mat,keyword_dict,mask_mat)(c)
}