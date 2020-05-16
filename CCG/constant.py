try:
    from CCG.soft_constant import *
except:
    pass

decay_rate = 0
epoch_num = 5
lr = 0.05
beam_width = 100

exp_file = 'all_exp_add.json'
exp_file_dump = 'exp_data_turk.pkl'
corenlp_model_path = '../stanford-corenlp-full-2018-10-05'


# Parsing parameters
predicate2idx =  {'$Arg': 0, '$The': 1, '$True': 2, '$False': 3, '$And': 4, '$Or': 5, '$Not': 6, '$All': 7, '$Any': 8, '$None': 9, '$Is': 10, '$Exists': 11, '$Int': 12, '$Direct': 13, '$Last': 14, '$AtLeastOne': 15, '$Because': 16, '$Equals': 17, '$NotEquals': 18, '$LessThan': 19, '$AtMost': 20, '$AtLeast': 21, '$MoreThan': 22, '$In': 23, '$Contains': 24, '$Separator': 25, '$Possessive': 26, '$Count': 27, '$Punctuation': 28, '$Tuple': 29, '$ArgXListAnd': 30, '$EachOther': 31, '$Token': 32, '$Word': 33, '$Char': 34, '$Upper': 35, '$Lower': 36, '$Capital': 37, '$StartsWith': 38, '$EndsWith': 39, '$Left': 40, '$Right': 41, '$Within': 42, '$Apart': 43, '$Sentence': 44, '$Between': 45, '$PersonNER': 46, '$LocationNER': 47, '$DateNER': 48, '$NumberNER': 49, '$OrganizationNER': 50, '$NorpNER': 51, '$ArgX': 52, '$ArgY': 53, '$There': 54, '$By': 55, '$Which': 56, '$Numberof': 57, '$Of': 58, '$Link': 59, '$SandWich': 60}
predicate2token =  {'$Arg': ['arg', 'argument'], '$The': ['the', 'The'], '$True': ['true', 'correct'], '$False': ['false', 'incorrect', 'wrong'], '$And': ['and', 'but'], '$Or': ['or', 'nor'], '$Not': ['not', 'n"t'], '$All': ['all', 'both'], '$Any': ['any', 'a', 'one of'], '$None': ['none', 'not any', 'neither', 'no'], '$Is': ['is', 'are', 'be', 'comes', 'come', 'appears', 'appear', 'as', 'occurs', 'occur', 'is stated', 'is found', 'said', 'is identified', 'are identified', 'is used', 'is placed'], '$Exists': ['exist', 'exists'], '$Int': ['no'], '$Direct': ['immediately', 'right', 'directly'], '$Last': ['last', 'final', 'ending'], '$AtLeastOne': ['a', 'another'], '$Because': ['because', 'since', 'if'], '$Equals': ['equal', 'equals', '=', '==', 'same as', 'same', 'identical', 'exactly'], '$NotEquals': ['different than', 'different'], '$LessThan': ['less than', 'smaller than', '<'], '$AtMost': ['at most', 'no larger than', 'less than or equal', 'within', 'no more than', '<='], '$AtLeast': ['at least', 'no less than', 'no smaller than', 'greater than or equal', '>='], '$MoreThan': ['more than', 'greater than', 'larger than', '>'], '$In': ['is in', 'in'], '$Contains': ['contains', 'contain', 'containing', 'include', 'includes', 'says', 'states', 'mentions', 'mentioned', 'referred', 'refers', 'is referring to'], '$Separator': [',', '/'], '$Possessive': ['"s'], '$Count': ['number', 'length', 'count'], '$Punctuation': [], '$Tuple': ['pair', 'tuple'], '$ArgXListAnd': ['they', 'them', 'entities', 'they', 'them'], '$EachOther': ['eachother', 'each other'], '$Token': ['token'], '$Word': ['word', 'words', 'term', 'terms', 'token', 'tokens', 'phrase', 'phrases', 'string'], '$Char': ['character', 'characters', 'letter', 'letters'], '$Upper': ['upper', 'uppercase', 'upper case', 'all caps', 'all capitalized'], '$Lower': ['lower', 'lowercase', 'lower case'], '$Capital': ['capital', 'capitals', 'capitalized'], '$StartsWith': ['starts with', 'start with', 'starting with'], '$EndsWith': ['ends with', 'end with', 'ending with'], '$Left': ['to the left of', 'left', 'in front of', 'before', 'precedes', 'preceding', 'followed by'], '$Right': ['to the right of', 'right', 'behind', 'after', 'preceded by', 'follows', 'following'], '$Within': ['within', 'next'], '$Apart': ['apart', 'away'], '$Sentence': ['sentence', 'text', 'it'], '$Between': ['between', 'in between', 'sandwiched', 'enclosed', 'Between', 'admist', 'in the middle of'], '$PersonNER': ['person', 'people'], '$LocationNER': ['location', 'locations', 'place', 'places'], '$DateNER': ['date', 'dates'], '$NumberNER': ['number', 'numbers'], '$OrganizationNER': ['organization', 'organizations', 'company', 'companies', 'agency', 'agencies', 'institution', 'institutions'], '$NorpNER': ['political', 'politician', 'religious'], '$ArgX': ['x', 'X', '<S>', 'SUBJ', 'SUBJ-ORGANIZATION', 'subject'], '$ArgY': ['y', 'Y', '<O>', 'OBJ', 'OBJ-ORGANIZATION', 'object'], '$There': ['there', 'There'], '$By': ['by'], '$Which': ['which'], '$Numberof': ['the number of'], '$Of': ['of'], '$Link': ['links', 'link', 'connects', 'connect'], '$SandWich': ['sandwich', 'sandwiches']}
token2predicate = {'arg': ['$Arg'], 'argument': ['$Arg'], 'the': ['$The'], 'The': ['$The'], 'true': ['$True'], 'correct': ['$True'], 'false': ['$False'], 'incorrect': ['$False'], 'wrong': ['$False'], 'and': ['$And'], 'but': ['$And'], 'or': ['$Or'], 'nor': ['$Or'], 'not': ['$Not'], 'n"t': ['$Not'], 'all': ['$All'], 'both': ['$All'], 'any': ['$Any'], 'a': ['$Any', '$AtLeastOne'], 'one of': ['$Any'], 'none': ['$None'], 'not any': ['$None'], 'neither': ['$None'], 'no': ['$None', '$Int'], 'is': ['$Is'], 'are': ['$Is'], 'be': ['$Is'], 'comes': ['$Is'], 'come': ['$Is'], 'appears': ['$Is'], 'appear': ['$Is'], 'as': ['$Is'], 'occurs': ['$Is'], 'occur': ['$Is'], 'is stated': ['$Is'], 'is found': ['$Is'], 'said': ['$Is'], 'is identified': ['$Is'], 'are identified': ['$Is'], 'is used': ['$Is'], 'is placed': ['$Is'], 'exist': ['$Exists'], 'exists': ['$Exists'], 'immediately': ['$Direct'], 'right': ['$Direct', '$Right'], 'directly': ['$Direct'], 'last': ['$Last'], 'final': ['$Last'], 'ending': ['$Last'], 'another': ['$AtLeastOne'], 'because': ['$Because'], 'since': ['$Because'], 'if': ['$Because'], 'equal': ['$Equals'], 'equals': ['$Equals'], '=': ['$Equals'], '==': ['$Equals'], 'same as': ['$Equals'], 'same': ['$Equals'], 'identical': ['$Equals'], 'exactly': ['$Equals'], 'different than': ['$NotEquals'], 'different': ['$NotEquals'], 'less than': ['$LessThan'], 'smaller than': ['$LessThan'], '<': ['$LessThan'], 'at most': ['$AtMost'], 'no larger than': ['$AtMost'], 'less than or equal': ['$AtMost'], 'within': ['$AtMost', '$Within'], 'no more than': ['$AtMost'], '<=': ['$AtMost'], 'at least': ['$AtLeast'], 'no less than': ['$AtLeast'], 'no smaller than': ['$AtLeast'], 'greater than or equal': ['$AtLeast'], '>=': ['$AtLeast'], 'more than': ['$MoreThan'], 'greater than': ['$MoreThan'], 'larger than': ['$MoreThan'], '>': ['$MoreThan'], 'is in': ['$In'], 'in': ['$In'], 'contains': ['$Contains'], 'contain': ['$Contains'], 'containing': ['$Contains'], 'include': ['$Contains'], 'includes': ['$Contains'], 'says': ['$Contains'], 'states': ['$Contains'], 'mentions': ['$Contains'], 'mentioned': ['$Contains'], 'referred': ['$Contains'], 'refers': ['$Contains'], 'is referring to': ['$Contains'], ',': ['$Separator', []], '/': ['$Separator'], '"s': ['$Possessive'], 'number': ['$Count', '$NumberNER'], 'length': ['$Count'], 'count': ['$Count'], 'pair': ['$Tuple'], 'tuple': ['$Tuple'], 'they': ['$ArgXListAnd', '$ArgXListAnd'], 'them': ['$ArgXListAnd', '$ArgXListAnd'], 'entities': ['$ArgXListAnd'], 'eachother': ['$EachOther'], 'each other': ['$EachOther'], 'token': ['$Token', '$Word'], 'word': ['$Word'], 'words': ['$Word'], 'term': ['$Word'], 'terms': ['$Word'], 'tokens': ['$Word'], 'phrase': ['$Word'], 'phrases': ['$Word'], 'string': ['$Word'], 'character': ['$Char'], 'characters': ['$Char'], 'letter': ['$Char'], 'letters': ['$Char'], 'upper': ['$Upper'], 'uppercase': ['$Upper'], 'upper case': ['$Upper'], 'all caps': ['$Upper'], 'all capitalized': ['$Upper'], 'lower': ['$Lower'], 'lowercase': ['$Lower'], 'lower case': ['$Lower'], 'capital': ['$Capital'], 'capitals': ['$Capital'], 'capitalized': ['$Capital'], 'starts with': ['$StartsWith'], 'start with': ['$StartsWith'], 'starting with': ['$StartsWith'], 'ends with': ['$EndsWith'], 'end with': ['$EndsWith'], 'ending with': ['$EndsWith'], 'to the left of': ['$Left'], 'left': ['$Left'], 'in front of': ['$Left'], 'before': ['$Left'], 'precedes': ['$Left'], 'preceding': ['$Left'], 'followed by': ['$Left'], 'to the right of': ['$Right'], 'behind': ['$Right'], 'after': ['$Right'], 'preceded by': ['$Right'], 'follows': ['$Right'], 'following': ['$Right'], 'next': ['$Within'], 'apart': ['$Apart'], 'away': ['$Apart'], 'sentence': ['$Sentence'], 'text': ['$Sentence'], 'it': ['$Sentence'], 'between': ['$Between'], 'in between': ['$Between'], 'sandwiched': ['$Between'], 'enclosed': ['$Between'], 'Between': ['$Between'], 'admist': ['$Between'], 'in the middle of': ['$Between'], 'person': ['$PersonNER'], 'people': ['$PersonNER'], 'location': ['$LocationNER'], 'locations': ['$LocationNER'], 'place': ['$LocationNER'], 'places': ['$LocationNER'], 'date': ['$DateNER'], 'dates': ['$DateNER'], 'numbers': ['$NumberNER'], 'organization': ['$OrganizationNER'], 'organizations': ['$OrganizationNER'], 'company': ['$OrganizationNER'], 'companies': ['$OrganizationNER'], 'agency': ['$OrganizationNER'], 'agencies': ['$OrganizationNER'], 'institution': ['$OrganizationNER'], 'institutions': ['$OrganizationNER'], 'political': ['$NorpNER'], 'politician': ['$NorpNER'], 'religious': ['$NorpNER'], 'x': ['$ArgX'], 'X': ['$ArgX'], '<S>': ['$ArgX'], 'SUBJ': ['$ArgX'], 'SUBJ-ORGANIZATION': ['$ArgX'], 'subject': ['$ArgX'], 'y': ['$ArgY'], 'Y': ['$ArgY'], '<O>': ['$ArgY'], 'OBJ': ['$ArgY'], 'OBJ-ORGANIZATION': ['$ArgY'], 'object': ['$ArgY'], 'there': ['$There'], 'There': ['$There'], 'by': ['$By'], 'which': ['$Which'], 'the number of': ['$Numberof'], 'of': ['$Of'], 'links': ['$Link'], 'link': ['$Link'], 'connects': ['$Link'], 'connect': ['$Link'], 'sandwich': ['$SandWich'], 'sandwiches': ['$SandWich']}
phrases =  {'one of': '$Any', 'not any': '$None', 'is stated': '$Is', 'is found': '$Is', 'is identified': '$Is', 'are identified': '$Is', 'is used': '$Is', 'is placed': '$Is', 'same as': '$Equals', 'different than': '$NotEquals', 'less than': '$LessThan', 'smaller than': '$LessThan', 'at most': '$AtMost', 'no larger than': '$AtMost', 'less than or equal': '$AtMost', 'no more than': '$AtMost', 'at least': '$AtLeast', 'no less than': '$AtLeast', 'no smaller than': '$AtLeast', 'greater than or equal': '$AtLeast', 'more than': '$MoreThan', 'greater than': '$MoreThan', 'larger than': '$MoreThan', 'is in': '$In', 'is referring to': '$Contains', 'each other': '$EachOther', 'upper case': '$Upper', 'all caps': '$Upper', 'all capitalized': '$Upper', 'lower case': '$Lower', 'starts with': '$StartsWith', 'start with': '$StartsWith', 'starting with': '$StartsWith', 'ends with': '$EndsWith', 'end with': '$EndsWith', 'ending with': '$EndsWith', 'to the left of': '$Left', 'in front of': '$Left', 'followed by': '$Left', 'to the right of': '$Right', 'preceded by': '$Right', 'in between': '$Between', 'in the middle of': '$Between', 'the number of': '$Numberof'}

comma_index = {',':0,'-LRB-':1,'-RRB-':2,'.':3,'-':4}


#feature
opsfeature = {'@And@Is': 0, '@Is@Word': 1, '@Is@Direct': 2, '@Direct@Left0': 3, '.root@And': 4, '@Is@between': 5, '@Direct@Right0': 6, '@Is@And': 7, '@And@Word': 8, '@between@And': 9, '@Word@And': 10, '@Left0@Word': 11, '@Is@Left0': 12, '@Right0@Word': 13, '@between@Word': 14, '.root@Is': 15, '@Is@Right0': 16, '@Right0@And': 17, '@Left0@And': 18, '@Is@LessThan': 19, '@LessThan@between': 20, '@LessThan@Num': 21, '@And@Right0': 22, '@And@between': 23, '.root@between': 24, '@between@Is': 25, '.root@Left': 26, '@Left@Word': 27, '@And@Left0': 28}



compare={                   #eq mt(more than) lt(less than) nmt(no more than) nlt(no less than)
    'eq':lambda a,b:a==b,
    'mt':lambda a,b:a>b,
    'lt':lambda a,b:a<b,
    'nmt':lambda a,b:a<=b,
    'nlt':lambda a,b:a>=b
}

NER2ID =  {'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6,'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}
TYPE2ID = {'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9,
                 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}


entity_type = {
    'per:title': 'PERSON TITLE',
    'org:top_members/employees': 'ORGANIZATION PERSON',
    'per:employee_of': 'PERSON ORGANIZATION',
    'org:alternate_names': 'ORGANIZATION ORGANIZATION',
    'org:country_of_headquarters': 'ORGANIZATION COUNTRY',
    'per:countries_of_residence': 'PERSON COUNTRY',
    'org:city_of_headquarters': 'ORGANIZATION CITY',
    'per:cities_of_residence': 'PERSON CITY',
    'per:age': 'PERSON NUMBER',
    'per:stateorprovinces_of_residence': 'PERSON STATE_OR_PROVINCE',
    'per:origin': 'PERSON NATIONALITY',
    'org:subsidiaries': 'ORGANIZATION ORGANIZATION',
    'org:parents': 'ORGANIZATION ORGANIZATION',
    'per:spouse': 'PERSON PERSON',
    'org:stateorprovince_of_headquarters': 'ORGANIZATION STATE_OR_PROVINCE',
    'per:children': 'PERSON PERSON',
    'per:other_family': 'PERSON PERSON',
    'per:alternate_names': 'PERSON PERSON',
    'org:members': 'ORGANIZATION ORGANIZATION',
    'per:siblings': 'PERSON PERSON',
    'per:schools_attended': 'PERSON ORGANIZATION',
    'per:parents': 'PERSON PERSON',
    'per:date_of_death': 'PERSON DATE',
    'org:member_of': 'ORGANIZATION ORGANIZATION',
    'org:founded_by': 'ORGANIZATION PERSON',
    'org:website': 'ORGANIZATION URL',
    'per:cause_of_death': 'PERSON CAUSE_OF_DEATH',
    'org:political/religious_affiliation': 'ORGANIZATION RELIGION',
    'org:founded': 'ORGANIZATION DATE',
    'per:city_of_death': 'PERSON CITY',
    'org:shareholders': 'ORGANIZATION PERSON',
    'org:number_of_employees/members': 'ORGANIZATION NUMBER',
    'per:date_of_birth': 'PERSON DATE',
    'per:city_of_birth': 'PERSON CITY',
    'per:charges': 'PERSON CRIMINAL_CHARGE',
    'per:stateorprovince_of_death': 'PERSON STATE_OR_PROVINCE',
    'per:religion': 'PERSON RELIGION',
    'per:stateorprovince_of_birth': 'PERSON STATE_OR_PROVINCE',
    'per:country_of_birth': 'PERSON COUNTRY',
    'org:dissolved': 'ORGANIZATION DATE',
    'per:country_of_death': 'PERSON COUNTRY',
}


#------function of ops (Algorithm)--------
'''
ws: (word,word,word,....)  ('NER',NERTYPE) word 
w: word ('NER',NERTYPE) 
c: candidate  
p: function 
arg: ArgX or ArgY
a: (Word1,Word2)
'''
#List of attributes we need to support besides 'word' and 'tokens'
Selection = ['NER']


def count_sublist(lis,sublis):
    cnt = 0
    if len(sublis)>len(lis):
        return cnt
    len_sub = len(sublis)
    for st in range(len(lis)-len(sublis)+1):
        if lis[st:st+len_sub]==sublis:
            cnt+=1
    return cnt


#function for $And
def merge(x,y):
    if type(x)!= tuple:
        x = [x]
    if type(y)!= tuple:
        y = [y]
    x = list(x)
    y = list(y)
    return tuple(x+y)

#function for $Is
def IsFunc(ws,ps,c):
    if isinstance(ps,tuple):
        bool_list  = []
        for p in ps:
            if isinstance(ws, tuple):
                if ws[0] in Selection:
                    bool_list.append(p(ws)(c))
                else:
                    bool_list.append(all([p(w)(c) for w in ws]))
            else:
                bool_list.append(p(ws)(c))
        return all(bool_list)

    if isinstance(ws,tuple):
        if ws[0] in Selection:
            return ps(ws)(c)
        else:
            return all([ps(w)(c) for w in ws])
    else:
        return ps(ws)(c)

#function for @Left and @Right
def at_POSI(POSI,ws,arg,c,option=None):
    if isinstance(ws,tuple):
        if ws[0] in Selection:
            return ws[1] in c.get_other_posi(POSI,arg[-1])[ws[0]]
        else:
            bool_list = []
            for w in ws:
                bool_list.append(at_POSI_0(POSI,arg,w,c,option))
            return all(bool_list)
    else:
        return at_POSI_0(POSI,arg,ws,c,option)

#function for @Left0 and @Right0
def at_POSI_0(POSI,arg,w,c,option=None):
    if arg not in ['ArgX','ArgY']:
        w,arg = arg,w
        if POSI == 'Left':
            POSI = 'Right'
        elif POSI == 'Right':
            POSI = 'Left'

    if isinstance(w,tuple) and w[0] not in Selection:
        return all([at_POSI_0(POSI,arg,ww,c,option) for ww in w])

    if w=='ArgY':
        w = c.obj
    elif w=='ArgX':
        w = c.subj

    if option==None:
        option = {'attr': 'word', 'range': -1, 'numAppear':1,'cmp':'nlt','onlyCount':False}  #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        return w[1] in c.get_other_posi(POSI, arg[-1])[w[0]]
    else:                                                                                   #For now,option['attr']==tokens if and only if 'right before' is used or onlyCount==True, otherwise ==word
        w = w.lower()
        w_split = w.split()
        while '' in w_split:
            w_split.remove('')
        if option == 'Direct':
            range_ = len(w_split)
            info = [token.lower() for token in c.get_other_posi(POSI, arg[-1])['tokens']]
            if POSI == 'Left':
                st = max(0, len(info) - range_)
                info = info[st:]
            elif POSI == 'Right':
                ed = min(len(info) - 1, range_-1)
                info = info[:ed + 1]
            else:
                raise ValueError
            # print(info)
            if info==w_split:
                return True
            else:
                return False
        if option['range']==-1:
            info = [token.lower() for token in c.get_other_posi(POSI, arg[-1])['tokens']]
        else:                                                                               #For now, if range!=-1 then attr == 'tokens'
            info = [token.lower() for token in c.get_other_posi(POSI, arg[-1])[option['attr']]]
            range_ = option['range']
            if POSI == 'Left':
                st = max(0, len(info) - range_)
                info = info[st:]
            elif POSI=='Right':
                ed = min(len(info) - 1, range_-1)
                info = info[:ed + 1]
            else:
                count_posi = c.get_other_posi(POSI, arg[-1])['POSI']
                st = max(0,count_posi-range_)
                ed = min(len(info),count_posi+1+range_)
                info = info[st:ed+1]
        if option['onlyCount']:
            return compare[option['cmp']](len(info),option['numAppear'])
        else:
            return compare[option['cmp']](count_sublist(info,w_split),option['numAppear'])


#function for @Between

def at_between(w,c,option=None,a=None):
    if w=='ArgY':
        w = c.obj
    elif w=='ArgX':
        w = c.subj
    if option==None:
        option = {'attr': 'word', 'numAppear':1,'cmp':'nlt','onlyCount':False}                #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        return w[1] in c.get_mid()[w[0]]
    else:                                                                                   #For now,option['attr']==tokens if and only if  onlyCount==True, otherwise ==word
        w = w.lower()
        w_split = w.split()
        while '' in w_split:
            w_split.remove('')
        info = [token.lower() for token in c.get_mid()['tokens']]
        if option['onlyCount']:
            return compare[option['cmp']](len(info),option['numAppear'])
        else:
            # print(info, w,info.count(w),type(info.count(w)),option['numAppear'],option['eq'],compare[option['eq']](info.count(w),option['numAppear']))
            return compare[option['cmp']](count_sublist(info,w_split),option['numAppear'])


#function for counting
def at_lessthan(funcx,nouny,w,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'lt','onlyCount':onlyCount})(c)                #There are less than 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(c)      #the word 'x' is less than 3 words before OBJ

def at_atmost(funcx,nouny,w,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'nmt','onlyCount':onlyCount})(c)                #There are at most 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(c)      #the word 'x' is at most 3 words before OBJ

def at_atleast(funcx,nouny,w,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'nlt','onlyCount':onlyCount})(c)             #There are at least 3 words before OBJ
    else:
        return funcx(w,{'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(c)      #the word 'x' is no less than 3 words before OBJ

def at_morethan(funcx,nouny,w,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'mt','onlyCount':onlyCount})(c)                #There are more than 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(c)    #the word 'x' is more than 3 words before OBJ


#function for @In0

def at_In0(arg,w,c):
    assert arg=='Sentence'
    if isinstance(w,tuple):
        return w in c.ner
    else:
        w = w.lower().split()
        info = [token.lower() for token in c.token]
        return count_sublist(info,w)>0

def at_WordCount(nounNum,nouny,F,c):
    if isinstance(nouny,tuple):
        return all([F(noun, option={'attr': 'word', 'range': -1, 'numAppear': 1, 'cmp': 'nlt', 'onlyCount': False})(c) for noun in nouny]) and F(nouny[0],option={'attr': 'tokens','range': -1,'numAppear':sum([len(noun.split()) for noun in nouny]),'cmp': 'eq','onlyCount': True})(c)
    else:
        return F(nouny,option={'attr':'word','range':-1,'numAppear':1,'cmp':'nlt','onlyCount':False})(c) and F(nouny,option={'attr':'tokens','range':-1,'numAppear':len(nouny.split()),'cmp':'eq','onlyCount':True})(c)

ops={
    ".root":lambda xs:lambda c:all([x(c) for x in xs]) if type(xs)==tuple else xs(c),
    "@Word":lambda x:x,
    "@Is":lambda ws,p: lambda c: IsFunc(ws,p,c),
    "@between": lambda a: lambda w,option=None:lambda c:at_between(w,c,option,a),
    "@And": lambda x,y:merge(x,y),
    "@Num": lambda x,y:{'attr':y,'num':int(x)},
    "@LessThan":lambda funcx,nouny:lambda w:lambda c:at_lessthan(funcx,nouny,w,c),
    "@AtMost":lambda funcx,nouny:lambda w:lambda c:at_atmost(funcx,nouny,w,c),
    "@AtLeast":lambda funcx,nouny:lambda w:lambda c:at_atleast(funcx,nouny,w,c),
    "@MoreThan":lambda funcx,nouny:lambda w:lambda c:at_morethan(funcx,nouny,w,c),
    "@WordCount":lambda nounNum,nouny,F:lambda useless:lambda c:at_WordCount(nounNum,nouny,F,c),

    "@NumberOf":lambda x,f:[x,f],
    "@LessThan1":lambda nounynum:lambda x:lambda c:at_lessthan(x[1],{'attr':x[0],"num":int(nounynum)},'There',c),
    "@AtMost1":lambda nounynum:lambda x:lambda c:at_atmost(x[1],{'attr':x[0],"num":int(nounynum)},'There',c),
    "@AtLeast1":lambda nounynum:lambda x:lambda c:at_atleast(x[1],{'attr':x[0],"num":int(nounynum)},'There',c),
    "@MoreThan1":lambda nounynum:lambda x:lambda c:at_morethan(x[1],{'attr':x[0],"num":int(nounynum)},'There',c),


    "@In0":lambda arg: lambda w:lambda c:at_In0(arg,w,c),
    #By
    "@By":lambda x,f,z:lambda c: f(x,{'attr': z['attr'], 'range': z['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': False})(c),
    # is xx Arg
    "@Left0":lambda arg: lambda w,option=None:lambda c:at_POSI_0('Left',arg,w,c,option),
    "@Right0":lambda arg: lambda w,option=None:lambda c:at_POSI_0('Right',arg,w,c,option),
    "@Range0":lambda arg: lambda w,option=None:lambda c:at_POSI_0('Range',arg,w,c,option),

    "@Left":lambda arg, ws,option=None:lambda c:at_POSI('Left',ws,arg,c,option),
    "@Right":lambda arg, ws,option=None:lambda c:at_POSI('Right',ws,arg,c,option),

    "@Direct":lambda func:lambda w:lambda c: func(w,'Direct')(c),

    "@StartsWith": lambda x,y: lambda c: c.with_(x[-1],'starts',y),
    "@EndsWith": lambda x,y: lambda c:c.with_(x[-1],'ends',y),
}


syntax = {
    "@LOC": ('NER','LOCATION'),
    "@Date":('NER','DATE'),
    "@Num":('NER','NUMBER'),
    "@Org":('NER','ORGANIZATION'),
    "@Norp":('NER','Norp'),
    '@PER':('NER','PERSON')
}

lexicon_head = ''':- S,NP,N,PP
        VP :: S\\NP
        Det :: NP/N
        Adj :: N/N'''

raw_lexicon =''':- S,NP,N,PP
        VP :: S\\NP
        Det :: NP/N
        Adj :: N/N
        arg => NP {None}
        #$True => (S\\VP)/PP {None}
        #$False => (S\\VP)/PP {None}
        $And => var\\.,var/.,var {\\x y.'@And'(x,y)}
        $Or => var\\.,var/.,var {\\x y.'@Or'(x,y)}
        $Not => (S\\NP)\\(S\\NP) {None}
        $Not => (S\\NP)/(S\\NP) {None}
        $All => NP/N {None}
        $All => NP {None}
        $All => NP/NP {None}
        $Any => NP/N {None}
        $None => N {None}
        #$Is => (S\\NP)/NP {\\y x.'@Is'(x,y)}
        #$Is => (S\\NP)/(S\\NP) {\\y x.'@Is'(x,y)}
        $Is => (S\\NP)/PP {\\y x.'@Is'(x,y)}   # word 'a' occurs between <S> and <O>
        $Is => (S\\NP)\\PP {\\y x.'@Is'(x,y)}  # between <S> and <O> occurs word 'a'
        $Is => (S\\PP)\\NP {\\x y.'@Is'(x,y)}  # between <S> and <O> word 'a' occurs
        $Exists => S\\NP/PP {\\y x.'@Is'(x,y)}
        #$Exists => S\\NP {None}
        $Int => Adj {None} #There are no words between <S> and <O>
        $AtLeastOne => NP/N {None}
        #$Equals => (S\\NP)/NP {None}
        #$NotEquals => (S\\NP)/NP {None}
        
        
        $LessThan => PP/PP/N {\\x y.'@LessThan'(y,x)} #There are less than 3 words between <S> and <O>   
        $AtMost => PP/PP/N {\\x y.'@AtMost'(y,x)} #There are at most 3 words between <S> and <O>
        $AtLeast => PP/PP/N {\\x y.'@AtLeast'(y,x)} #same as above
        $MoreThan => PP/PP/N {\\x y.'@MoreThan'(y,x)} #same as above
        
        $LessThan => PP/N {\\x.'@LessThan1'(y,x)} #number of words between X and Y is less than 7.
        $AtMost => PP/N {\\x.'@AtMost1'(y,x)} 
        $AtLeast => PP/N {\\x.'@AtLeast1'(y,x)}   #same as above
        $MoreThan => PP/N {\\x.'@MoreThan1'(y,x)} #same as above

        #$In => S\\NP/NP {None} 
        $In => PP/NP {\\x.'@In0'(x)} 
        $Contains => S\\NP/NP {None} #The sentence contains two words
        $Separator => var\\.,var/.,var {\\x y.'@And'(x,y)} #connection between two words
        #$Processive => NP/N\\N {None}
        #$Count => N {None}
        #$Tuple => N {None}
        #$ArgXListAnd => NP {None}
        $EachOther => N {None}
        $Token => N {\\x.'@Word'(x)}
        $Word => NP/N {\\x.'@Word'(x)}
        $Word => NP/NP {\\x.'@Word'(x)}
        
        $Word => N {'tokens'} #There are no more than 3 words between <S> and <O>
        $Word => NP {'tokens'} #There are no more than 3 words between <S> and <O>
        
        $Char => N {None} #same as above
        #$Lower => Adj {None}
        #$Capital => Adj {None}
        $StartsWith => S\\NP/NP {\\y x.'@StartsWith'(x,y)}
        $EndsWith => S\\NP/NP {\\y x.'@EndsWith'(x,y)}
        $Left => PP/NP {\\x.'@Left0'(x)} # the word 'a' is before <S>
        $Left => (S\\NP)/NP {\\y x.'@Left'(y,x)}  #Precedes
        $Right => PP/NP {\\x.'@Right0'(x)}# the word 'a' ia after <S>
        $Right => (S\\NP)/NP {\\y x.'@Right'(y,x)} 
        #$Within => ((S\\NP)\\(S\\NP))/NP {None} # the word 'a' is within 2 words after <S>
        #$Within => (NP\\NP)/NP {None}
        $Within => PP/PP/N {\\x y.'@AtMost'(y,x)} #Does Within has other meaning.
        $Sentence => NP {'Sentence'}
        
        $Between => (S/S)/NP {\\x y.'@between'(x,y)}
        $Between => S/NP {\\x.'@between'(x)}
        $Between => PP/NP {\\x.'@between'(x)}
        $Between => (NP\\NP)/NP {\\x y.'@between'(x,y)}
        
        $PersonNER => NP {'@PER'}
        $LocationNER => NP {'@LOC'}
        $DateNER => NP {'@Date'}
        $NumberNER => NP {'@Num'}
        $OrganizationNER => NP {'@Org'}
        $NorpNER => NP {'@Norp'}
        $ArgX => NP {'ArgX'}
        $ArgY => NP {'ArgY'}
        #$will => S\\NP/VP {None}
        #$Which => (NP\\NP)/(S/NP) {None}
        #$might => S\\NP/VP {None}
        $that => NP/N {None}
        #$that => (N\\N)/(S/NP) {None} #same as which
        $Apart => (S/PP)\\NP {None}
        $Direct => PP/PP {\\x.'@Direct'(x)} # the word 'a' is right before <S>   
        $Direct => (S\\NP)/PP {\\y x.'@Is'(x,'@Direct'(y))}
        $Last => Adj {None}
        $There => NP {'There'}
        $By => S\\NP\\PP/NP {\\z f x.'@By'(x,f,z)} #precedes sth by 10 chatacters       
        $By => (S\\NP)\\PP/(PP/PP) {\\F x y.'@Is'(y,F(x))} #precedes sth by no more than10 chatacters        
        $By => PP\\PP/(PP/PP) {\\F x. F(x)} #occurs before by no...
        
        $Numberof => NP/PP/NP {\\x F.'@NumberOf'(x,F)}
        
        $Of => PP/NP {\\x.'@Range0'(x)} # the word 'x' is at most 3 words of Y     
        $Of => NP/NP {\\x.x} #these two are designed to solve problems like $Is $Left $Of and $Is $Left
        $Of => N/N {\\x.x}
        $Char => NP/N {None}
        $ArgX => N {'ArgX'}
        $ArgY => N {'ArgY'}
        $Link => (S\\NP)/NP {\\x y.'@Is'(y,'@between'(x))}
        $SandWich => (S\\NP)/NP {\\x y.'@Is'(x,'@between'(y))}
        $The => N/N {\\x.x}
        $The =>NP/NP {\\x.x}
        '''
