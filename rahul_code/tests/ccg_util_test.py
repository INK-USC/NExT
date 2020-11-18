import spacy
import sys
sys.path.append("../")
from CCG_new import utils

nlp = spacy.load("en_core_web_sm")

def test_normalize_quotes():
    inputs = ["'abc'", "`bcd`", "\"`'bdc'`\""]
    outputs = ["\"abc\"", "\"bcd\"", "\"bdc\""]
    for i, inp in enumerate(inputs):
        assert utils.normalize_quotes(inp) == outputs[i]
    
def test_clean_text():
    text = "Rahul"
    assert utils.clean_text(text) == "rahul"
    assert utils.clean_text(text, lower=False, exclusive=False) == "Rahul"

    text = "Rahul, what if I want to keep punctuation!?!?!?."
    assert utils.clean_text(text) == "rahul, what if i want to keep punctuation."
    assert utils.clean_text(text, collapse_punc=False, exclusive=False) == "rahul, what if i want to keep punctuation!?!?!?."
    
    text = "RAHUL,,,,, why did you leave!?!?!?!."
    assert utils.clean_text(text) == "rahul, why did you leave."
    assert utils.clean_text(text, commas=False) == "rahul,,,,, why did you leave."

    text = "Rahul, maybe its a good idea to drink more water & less coffee."
    assert utils.clean_text(text) == "rahul, maybe its a good idea to drink more water and less coffee."
    assert utils.clean_text(text, switch_amp=False, exclusive=False) == "rahul, maybe its a good idea to drink more water & less coffee."

    text = "RahUL, look it's a tweet #hashtags @username could be USEFUL, but we also loose tokens like $%^"
    assert utils.clean_text(text) ==  "rahul, look it's a tweet hashtags username could be useful, but we also loose tokens like"
    assert utils.clean_text(text, exclusive=False) == "rahul, look it's a tweet #hashtags @username could be useful, but we also loose tokens like $%^"
    
    text = "Rahul    sometimes there is like    weird   whitespace!"
    assert utils.clean_text(text) == "rahul sometimes there is like weird whitespace."
    assert utils.clean_text(text, whitespace=False) == "rahul    sometimes there is like    weird   whitespace."

def test_clean_explanation():
    explanation = "The tweet contains the phrase 'cheery smile'"
    assert utils.clean_explanation(explanation) == "the tweet contains the phrase \"cheery smile\""

    explanation = "The tweet contains the phrase 'Cheery smile'"
    assert utils.clean_explanation(explanation) == "the tweet contains the phrase \"cheery smile\""

    explanation = "The tweet contains some funky characters like '$$$!!!@@#()#)$*$' & BETWEEN SUBJ AND OBJ THERE are four Words"
    assert utils.clean_explanation(explanation) == "the tweet contains some funky characters like \"$$$!!!@@#()#)$*$\" and between subj and obj there are four words"
    
def test_convert_chunk_to_terminal():
    terminal_chunks = ["not", "n\"t", "n't", "number", "/"]
    terminals = [["$Not"], ["$Not"], ["$Not"], ['$Count', '$NumberNER'], ['$Separator']]
    for i, chunk in enumerate(terminal_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == terminals[i]
    
    quote_chunks = ["\"this is a quote chunk\"", "\"and this is another one\""]
    for i, chunk in enumerate(quote_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == [chunk]
    
    digit_chunks = ["4", "3", "10", "100"]
    digit_quotes = [["\"4\""], ["\"3\""], ["\"10\""], ["\"100\""]]
    for i, chunk in enumerate(digit_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == digit_quotes[i]
    
    num_chunks = ["one", "two", "ten"]
    num_quotes = [["\"1\""], ["\"2\""], ["\"10\""]]
    for i, chunk in enumerate(num_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == num_quotes[i]
    
    not_recognized_chunks = ["glue", "blue", "shoe", "boo boo", "gaboo"]
    for i, chunk in enumerate(not_recognized_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == None    

def test_chunk_explanation():
    explanation = "The words ', is a' appear right before OBJ and the word 'citizen' is right after OBJ"
    expected_chunking = ['the', 'words', '", is a"', 'appear', 'right', 'before', 'obj', 'and', 'the', 'word', '"citizen"', 'is', 'right', 'after', 'obj']
    cleaned_explanation = utils.clean_explanation(explanation)
    chunked_explanation = utils.chunk_explanation(cleaned_explanation, nlp)
    assert chunked_explanation == expected_chunking

    explanation = "The words 'year old' are to the right of OBJ and 'is' is right before OBJ"         
    expected_chunking = ['the', 'words', '"year old"', "are", "$Right", "obj", "and", '"is"', "is", "right", "before", "obj"]
    cleaned_explanation = utils.clean_explanation(explanation)
    chunked_explanation = utils.chunk_explanation(cleaned_explanation, nlp)
    assert chunked_explanation == expected_chunking

    explanation = "The tweet contains the phrase 'when the GTA Online Biker DLC comes out'"
    expected_chunking = ['the', 'tweet', 'contains', 'the', 'phrase', '"when the gta online biker dlc comes out"']
    cleaned_explanation = utils.clean_explanation(explanation)
    chunked_explanation = utils.chunk_explanation(cleaned_explanation, nlp)
    assert chunked_explanation == expected_chunking

def test_prepare_token_for_rule_addition():
    quote_token = "\"this could be important!!:)\""
    expected_prep = "\"thisSPACEcouldSPACEbeSPACEimportantPUNCT0PUNCT0PUNCT14PUNCT7\""
    prepped_token = utils.prepare_token_for_rule_addition(quote_token)
    assert prepped_token == expected_prep
    
    reverse_token = utils.prepare_token_for_rule_addition(expected_prep, reverse=True)
    assert reverse_token == quote_token

def test_generate_phrase():
    sentence = "His wife, OBJ-PERSON, often accompanied him on SUBJ-PERSON SUBJ-PERSON expeditions, as she did in 1947, when she became the first woman to climb Mount McKinley"
    phrase_tokens = ['His', 'wife', ',', 'OBJ-PERSON', ',', 'often', 'accompanied', 'him', 'on', 'SUBJ-PERSON', 'expeditions', ',', 'as', 'she', 'did', 'in', '1947', ',', 'when', 'she', 'became', 'the', 'first', 'woman', 'to', 'climb', 'Mount', 'McKinley']
    # the last two NER labels are wrong, but are the output of spaCy's NER tagger.
    phrase_ners = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'DATE', '', '', '', '', '', 'ORDINAL', '', '', '', 'PERSON', 'PERSON']
    phrase_subj_posi = 9
    phrase_obj_posi = 3
    phrase = utils.generate_phrase(sentence, nlp)
    assert len(phrase.tokens) == len(phrase.ners)
    assert phrase.tokens == phrase_tokens
    assert phrase.ners == phrase_ners
    assert phrase.subj_posi == phrase_subj_posi
    assert phrase.obj_posi == phrase_obj_posi

    sentence = "SUBJ-PERSON's mother OBJ-PERSON was a singer in the dance group Soul II Soul, which had hits in the 1980s and 1990s."
    phrase_tokens = ['SUBJ-PERSON', "'s", 'mother', 'OBJ-PERSON', 'was', 'a', 'singer', 'in', 'the', 'dance', 'group', 'Soul', 'II', 'Soul', ',', 'which', 'had', 'hits', 'in', 'the', '1980s', 'and', '1990s', '.']
    phrase_ners = ['', '', '', '', '', '', '', '', '', '', '', 'PRODUCT', 'PRODUCT', 'PRODUCT', '', '', '', '', '', '', 'DATE', '', 'DATE', '']
    phrase_subj_posi = 0
    phrase_obj_posi = 3
    phrase = utils.generate_phrase(sentence, nlp)
    assert len(phrase.tokens) == len(phrase.ners)
    assert phrase.tokens == phrase_tokens
    assert phrase.ners == phrase_ners
    assert phrase.subj_posi == phrase_subj_posi
    assert phrase.obj_posi == phrase_obj_posi

    sentence = "GAMEDAY VS BUFORD TODAY AT 5:30 AT HOME ! ! ! NEVER BEEN SO EXCITED #revenge"
    phrase_tokens = ['GAMEDAY', 'VS', 'BUFORD', 'TODAY', 'AT', '5:30', 'AT', 'HOME', '!', '!', '!', 'NEVER', 'BEEN', 'SO', 'EXCITED', '#', 'revenge']
    phrase_ners = ['ORG', '', '', '', '', 'TIME', '', '', '', '', '', '', '', '', '', '', '']
    phrase_subj_posi = None
    phrase_obj_posi = None
    phrase = utils.generate_phrase(sentence, nlp)
    assert len(phrase.tokens) == len(phrase.ners)
    assert phrase.tokens == phrase_tokens
    assert phrase.ners == phrase_ners
    assert phrase.subj_posi == phrase_subj_posi
    assert phrase.obj_posi == phrase_obj_posi
