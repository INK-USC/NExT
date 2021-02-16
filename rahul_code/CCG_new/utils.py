import re
import string
from CCG_new import constants
from CCG_new import util_classes
import pdb

def _find_quoted_phrases(explanation):
    """
        Checks for the existence of a quoted phrase within an explanation
        Three types of quotes are accepted
        
        Arguments:
            explanation (str) : explanation text for a labeling decision
        
        Returns:
            arr : an array of quoted phrases or an empty array
    """
    possible_queries = re.findall('"[^"]+"', explanation)
    if len(possible_queries):
        possible_queries = [query for query in possible_queries]
        return possible_queries
    
    possible_queries = re.findall("'[^']+'", explanation)
    if len(possible_queries):
        possible_queries = [query for query in possible_queries]
        return possible_queries

    possible_queries = re.findall("`[^`]+`", explanation)
    if len(possible_queries):
        possible_queries = [query for query in possible_queries]
        return possible_queries
    
    return []

def segment_explanation(explanation):
    """
        Segments an explanation into portions that have a quoted phrase and those that don't
        Prepends the quoted word segments with an asterisk to indicate the existences of a quoted phrase

        Arguments:
            explanation (str) : raw explanation text
        
        Returns:
            arr : segmented explanation
    """
    possible_queries = _find_quoted_phrases(explanation)
    pieces = []
    last_end_position = 0
    for query in possible_queries:
        start_position = explanation.find(query)
        piece = explanation[last_end_position:start_position]
        last_end_position = start_position + len(query)
        pieces.append(piece)
        pieces.append("*"+query)
    
    if last_end_position < len(explanation):
        pieces.append(explanation[last_end_position:])
    
    return pieces

def clean_text(text, lower=True, collapse_punc=True, commas=True, switch_amp=True, exclusive=True, whitespace=True):
    """
        Function that cleans text in the following way:
            1. Lowecases the text
            2. Replaces all !.? with a a single ., keeps any ... in the text
            3. Replaces multiple commas with a single comma (same for ":")
            4. Removes all non [lowercase characters, \., \,, digit, \', \", >, <, =, \/]
            5. Removes unnecessary whitespace within text 
        
        Used for Cleaning Explanations.

        Arguments:
            text           (str) : text to be cleaned
            lower         (bool) : lowercase text or not
            collapse_punc (bool) : whether to collapse punctuation to a single .
            commas        (bool) : whether to get rid of unnecessary commas
            switch_amp    (bool) : whether to switch & into 'and'
            exclusive     (bool) : remove all characters not mentioned above
            whitespace    (bool) : whether to get rid of unnecessary whitespace
        
        Returns:
            str : cleaned text
    """
    if lower:
        text = text.lower()
    if collapse_punc:
        text = re.sub(r'\?+', '.', text)
        text = re.sub(r'\!+', '.', text)
        text = re.sub(r'(\.){2}', ".", text)
        text = re.sub(r'(\.){4,}', ".", text)
    if commas:
        text = re.sub(r",{2,}", ",", text)
    if switch_amp:
        text = re.sub(r'&', 'and', text)
    if exclusive:
        text = re.sub(r"[^a-z .,0-9'\"><=\/]", "", text)
    if whitespace:
        text = " ".join(text.split()).strip()

    return text

def clean_explanation(explanation):
    """
        Function that cleans an explanation text, but also makes sure to leave text within quotes uncleaned

        Arguments:
            explanation (str) : explanation to be cleaned
        
        Returns:
            str : cleaned text
    """
    segments = segment_explanation(explanation)
    for i, segment in enumerate(segments):
        if len(segment):
            if segment[0] != "*":
                segments[i] = clean_text(segment)
            else:
                segments[i] = "\"" + segment[2:len(segment)-1].lower() + "\""
    
    cleaned_explanation = " ".join(segments).strip()

    return cleaned_explanation

def convert_chunk_to_terminal(chunk):
    """
        Given a chunk we have three possible ways to convert it to a terminal in our lexicon.
        1. Check for chunk in our lexicon and convert it
        2. If a chunk matches text from the original statement, indicated in the explanation by " ", 
           then consider that its own terminal
        3. If its a number, we put quotes around it
        4. If its a numeric string we convert its base 10 form and put quotes around it
        5. Else this chunk will not be matched to a terminal

        Arguments:
            chunk (str) : chunk that needs to be converted
        
        Returns:
            arr|None : array of terminals that this chunk can converted to
                       if no good match is found None is returned
    """
    num_lexicon = ['one','two','three','four','five','six','seven','eight','nine', 'ten']
    num_lexicon_dict = {elem:str(i+1) for i,elem in enumerate(num_lexicon)}
    if chunk in constants.TOKEN_TERMINAL_MAP:
        return constants.TOKEN_TERMINAL_MAP[chunk]
    elif chunk.startswith("\"") and chunk.endswith("\"") or chunk in constants.PHRASE_VALUES:
        return [chunk]
    elif chunk.isdigit():
        return ["\""+chunk+"\""]
    elif chunk in num_lexicon:
        return ["\""+num_lexicon_dict[chunk]+"\""]
    else:
        return None

def chunk_explanation(cleaned_explanation, nlp):
    """
        Taking a cleaned explanation, we then chunk it to allow it to tokens to be converted
        into predicates from our grammar's lexicon. We use spaCy to tokenize the model, but also:
            1. check for certain phrases and convert them automatically into our lexicon space
            2. capture phrases in quotes and keep them as one "token"
        
        Arguments:
            cleaned_explanation (str) : output of clean_explanation
            nlp         (spaCy model) : an instantiated spaCy model
        
        Returns:
            arr : chunks (str) that can be converted into predicates of our grammar's vocab
    """
    for i, phrase in enumerate(constants.PHRASE_ARRAY):
        if phrase in cleaned_explanation:
            temp_sub = constants.PHRASE + str(i)
            cleaned_explanation = re.sub(phrase, temp_sub, cleaned_explanation)
    
    doc = nlp(cleaned_explanation)
    initial_tokens = [token.text for token in doc]
    final_tokens = []
    cur_token = ""
    inside = False
    for i, token in enumerate(initial_tokens):
        if token.startswith("\"") and not inside:
            if token.endswith("\"") and len(token) > 1:
                final_tokens.append(token)
            else:
                cur_token = token
                inside = True
        elif token.endswith("\""):
            if token != "\"":
                cur_token += " "
            cur_token += token
            final_tokens.append(cur_token)
            inside = False
        elif inside:
            if cur_token != "\"":
                cur_token += " "
            cur_token += token
        else:
            final_tokens.append(token)
    
    for i, token in enumerate(final_tokens):
        if token.startswith(constants.PHRASE):
            phrase = constants.PHRASE_ARRAY[int(token.split(constants.PHRASE)[1])]
            final_tokens[i] = constants.PHRASE_TERMINAL_MAP[phrase]
    return final_tokens

def clean_and_chunk(explanation, nlp):
    return chunk_explanation(clean_explanation(explanation), nlp)

def prepare_token_for_rule_addition(token, reverse=False):
    """
        Certain tokens spawn new rules in the CCG grammar, before the rule can be created though
        we must normalize the token. We do this to ensure the parsing of the grammar by the CCGChartParser.
        We also reverse the normalization when needed, but collapse certain patterns to a single comma.

        Arguments:
            token    (str) : token to be normalized
            reverse (bool) : reverse the normalization
        
        Returns:
            str : (un)normalized token
    """
    punctuations = string.punctuation.replace("\"", "")
    if reverse:
        token = token.replace(constants.SPACE, ' ')
        for i in range(len(punctuations)-1, -1, -1):
            replacement = constants.PUNCT + str(i)
            token = token.replace(replacement, punctuations[i])
        return token
    
    token = token.replace(' ', constants.SPACE)
    for i, punc in enumerate(punctuations):
        replacement = constants.PUNCT + str(i)
        token = token.replace(punc, replacement)
    return token

def add_rules_to_grammar(tokens, grammar_string):
    """
        Certain tokens require rules to be addeed to a base CCG grammar, depending on the token
        this function adds those rules to the grammar.

        Arguments:
            tokens         (arr) : tokens that require new rules to be added to the grammar
            grammar_string (str) : string representation of the base grammar to add to
    """
    grammar = grammar_string
    for token in tokens:
        raw_token = token[1:len(token)-1]
        token = prepare_token_for_rule_addition(token)
        if raw_token.isdigit():
            grammar = grammar + "\t" + token + " => NP/NP {\\x.'@Num'(" + token + ",x)}" + "\n\t" + token + " => N/N {\\x.'@Num'(" + token + ",x)}"+"\n"
            grammar = grammar + "\t" + token + " => NP {" + token + "}" + "\n\t\t" + token + " => N {" + token + "}"+"\n"
            grammar = grammar + "\t" + token + " => PP/PP/NP/NP {\\x y F.'@WordCount'('@Num'(" + token + ",x),y,F)}" + "\n\t" + token + " => PP/PP/N/N {\\x y F.'@WordCount'('@Num'(" + token + ",x),y,F)}"+"\n"
        else:
            grammar = grammar + "\t" + token + " => NP {"+token+"}"+"\n\t"+token+" => N {"+token+"}"+"\n"
    return grammar.strip()

def generate_phrase(sentence, nlp):
    """
        Generate a useful wrapper object for each sentence in the data

        Arguments:
            sentence    (str) : sentence to generate wrapper for
            nlp (spaCy model) : pre-loaded spaCy model to use for NER detection
        
        Returns
            Phrase : useful wrapper object
    """
    if "SUBJ" in sentence and "OBJ" in sentence:
        subj_type = re.search(r"SUBJ-[A-Z_'s,]+", sentence).group(0).split("-")[1].strip()
        subj_type = subj_type.replace("'s", "")
        subj_type = subj_type.replace(",", "")
        obj_type = re.search(r"OBJ-[A-Z_'s,]+", sentence).group(0).split("-")[1].strip()
        obj_type = obj_type.replace("'s", "")
        obj_type = obj_type.replace(",", "")
        sentence = re.sub(r"SUBJ-[A-Z_]+", "SUBJ", sentence)
        sentence = re.sub(r"SUBJ-[A-Z_'s]+", "SUBJ's", sentence)
        sentence = re.sub(r"SUBJ-[A-Z_]+,", "SUBJ,", sentence)
        sentence = re.sub(r"OBJ-[A-Z_]+", "OBJ", sentence)
        sentence = re.sub(r"OBJ-[A-Z_'s]+", "OBJ's ", sentence)
        sentence = re.sub(r"OBJ-[A-Z_]+,", "OBJ,", sentence)

    doc = nlp(sentence)
    ners = [token.ent_type_ if token.text not in ["SUBJ", "OBJ"] else "" for token in doc]
    tokens = [token.text for token in doc]
    subj_posi = None
    obj_posi = None
    indices_to_pop = []
    for i, token in enumerate(tokens):
        if token == "SUBJ":
            if subj_posi:
                indices_to_pop.append(i)
            else:
                subj_posi = i
                ners[i] = subj_type
        elif token == "OBJ":
            if obj_posi:
                indices_to_pop.append(i)
            else:
                obj_posi = i
                ners[i] = obj_type

    if len(indices_to_pop):
        indices_to_pop.reverse()
        for index in indices_to_pop:
            ners.pop(index)
            tokens.pop(index)
    
    return util_classes.Phrase(tokens, ners, subj_posi, obj_posi)

def create_semantic_repr(parse_tree):
    """
        Given a valid parse tree, we extract the semtantic string representation of the tree.
        In order to execute the functions that make up the semantic string, we create a hierarchical
        representation of the string, so that functions that rely on the return value of functions
        lower down the in the representation can be evaluated.

        In doing this, we loose the original lexical heirachy of the parse tree.
        Differs from this approach https://homes.cs.washington.edu/~lsz/papers/zc-uai05.pdf
        Meaning that we can't create features based on lexical structure, but only semantic structure

        If the semantic string representation of the tree makes sense when considering the ordering
        of the functions making up the string, then we output the hierarchical tuple, else we return
        false. False here indicates that while a valid parse tree was attainable, the semantics of the
        parse do not make sense.

        Arguments:
            parse_tree (nltk.tree.Tree (result of CCGChartParser.parse())) : tree to convert to semantic
                                                                             representation
        
        Returns:
            tuple | false : if valid semantically, we output a tuple describing the semantics, else false
    """
    full_semantics = str(parse_tree.label()[0].semantics())
    # Possible clause delimiter
    clauses = re.split(',|(\\()',full_semantics)
    delete_index = []
    for i in range(len(clauses)-1, -1, -1):
        if clauses[i] == None:
            delete_index.append(i)
    for i in delete_index:
        del clauses[i]
    
    # Switch poisition of ( and Word before it
    switched_semantics = []
    for i, token in enumerate(clauses):
        if token=='(':
            switched_semantics.insert(-1,'(')
        else:
            switched_semantics.append(token)
    # Converting semantic string into a multi-level tuple, ex: (item, tuple) would be a two level tuple
    # This representation allows for the conversion from semantic representation to labeling function
    hierarchical_semantics = ""
    for i, clause in enumerate(switched_semantics):
        prepped_clause = clause
        if prepped_clause.startswith("\""):
            prepped_clause = prepare_token_for_rule_addition(prepped_clause, reverse=True)
            if prepped_clause.endswith(")"):
                posi = len(prepped_clause)-1
                while prepped_clause[posi]==")":
                    posi-=1
                assert prepped_clause[posi]=="\""
            else:
                posi = len(prepped_clause)-1
                assert prepped_clause[posi] == "\""
            # print(prepped_clause)
            prepped_clause = prepped_clause[0] + \
                             prepped_clause[1:posi].replace('\'','\\\'') + \
                             prepped_clause[posi:]
            # print(prepped_clause[1:posi])
            # print(prepped_clause)

        if switched_semantics[i-1] != "(" and len(hierarchical_semantics):
            hierarchical_semantics += ","

        hierarchical_semantics += prepped_clause
    # if the ordering of the semantics in this semantic representation is acceptable per the functions
    # the semantics map to, then we will be able to create the desired multi-label tuple
    # else we return False
    # print(hierarchical_semantics)
    try:
        hierarchical_tuple = ('.root', eval(hierarchical_semantics))
    # print("cool")
    # print(hierarchical_tuple)
        return hierarchical_tuple
    except:
        return False

def create_labeling_function(semantic_repr, level=0):
    """
        Creates a labeling function (lambda function) from a hierarchical tuple representation
        of the semantics of a parse tree. The labeling function takes in a Phrase object and then
        evaluates whether the labeling function applies to this Phrase object.

        Arguments:
            semantic_repr (tuple) : hierarchical tuple representation
        
        Returns:
            function | false : if a function is creatable via the tuple, it is created, else false
    """
    try:
    # print("level {}".format(level))
        if isinstance(semantic_repr, tuple):
            # print("function name {}".format(semantic_repr[0]))
            op = constants.STRICT_MATCHING_OPS[semantic_repr[0]]
            # print("function {}".format(op))
            args = [create_labeling_function(arg, level=level+1) for arg in semantic_repr[1:]]
            # print("level {}".format(level))
            # print("args {}".format(args))
            if False in args:
                return False
            return op(*args) if args else op
        else:
            # print("semantic_repr {}".format(semantic_repr))
            if semantic_repr in constants.NER_TERMINAL_TO_EXECUTION_TUPLE:
                return constants.NER_TERMINAL_TO_EXECUTION_TUPLE[semantic_repr]
            else:
                return semantic_repr
    except:
        return False

def create_soft_labeling_function(semantic_repr, level=0):
    """
        Creates a labeling function (lambda function) from a hierarchical tuple representation
        of the semantics of a parse tree. The labeling function takes in a Phrase object and then
        evaluates whether the labeling function applies to this Phrase object.

        Arguments:
            semantic_repr (tuple) : hierarchical tuple representation
        
        Returns:
            function | false : if a function is creatable via the tuple, it is created, else false
    """
    try:
        if isinstance(semantic_repr, tuple):
            op = constants.SOFT_MATCHING_OPS[semantic_repr[0]]
            args = [create_soft_labeling_function(arg) for arg in semantic_repr[1:]]
            if False in args:
                return False
            return op(*args) if args else op
        else:
            if semantic_repr in constants.NER_TERMINAL_TO_EXECUTION_TUPLE:
                return constants.NER_TERMINAL_TO_EXECUTION_TUPLE[semantic_repr]
            else:
                return semantic_repr
    except:
        return False