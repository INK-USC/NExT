import sys
sys.path.append("../training/")
import random
import util_functions as func

random_state = 42
random.seed(random_state)

def test_tokenize():
    text = "HEY I've got some funKy things! \nIsn't it Funny!!     "
    tokens = ['hey', 'i', "'ve", 'got', 'some', 'funky', 'things', '!', 'is', "n't", 'it', 'funny', '!', '!']

    assert func.tokenize(text) == tokens

def test_build_vocab():
    train = ["Here are some strings", 
             "Some not so interesting strings!", 
             "Let's make ThEm MORe Intersting?",
             "Yes you can :)",
             "Coolio <::>"]
    
    embedding_name = "glove.6B.50d"

    vocab = func.build_vocab(train, embedding_name, random_state, False)

    tokens_in_order = ['<unk>', '<pad>', '<bos>', '<eos>', ':', 'some', 'strings', '!', "'s", ':)', '<', '>',
                       '?', 'are', 'can', 'coolio', 'here', 'interesting', 'intersting', 'let', 'make',
                       'more', 'not', 'so', 'them', 'yes', 'you']
    
    assert "torchtext.vocab.Vocab" in str(type(vocab))
    assert vocab.itos == tokens_in_order

def test_convert_text_to_tokens():
    train = ["Here are some strings", 
             "Some not so interesting strings!", 
             "Let's make ThEm MORe Intersting?",
             "Yes you can :)",
             "Coolio <::>"]
    
    embedding_name = "glove.6B.50d"

    vocab = func.build_vocab(train, embedding_name, random_state, False)

    sample_data = ["Let's make ThEm MORe Intersting?",
                   "Some not so interesting strings!", 
                   "However, this one is going to have lots of <unk>s"]
    
    tokenized_data = [[19, 8, 20, 24, 21, 18, 12], 
                      [5, 22, 23, 17, 6, 7], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 11, 0]]
    
    assert func.convert_text_to_tokens(sample_data, vocab, func.tokenize) == tokenized_data

def test_build_pretraining_triples():
    train = ["Here are some strings", 
             "Some not so interesting strings!", 
             "Let's make ThEm MORe Intersting?",
             "Yes you can :)",
             "Coolio <::>"]
    
    embedding_name = "glove.6B.50d"

    vocab = func.build_vocab(train, embedding_name, random_state, False)

    sample_data = ["Let's make ThEm MORe Intersting?",
                   "Some not so interesting strings!", 
                   "Are you sure, I'd like to be coolio!?"]
    
    act_tokenized_data = [[2, 19, 8, 20, 24, 21, 18, 12, 3],
                          [2, 5, 22, 23, 17, 6, 7, 3],
                          [2, 13, 26, 0, 0, 0, 0, 0, 0, 0, 15, 7, 12, 3]]
    
    act_queries = [[19], [22, 23, 17], [0, 0]]

    act_labels = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    token_seqs, queries, labels = func.build_pretraining_triples(sample_data, vocab, func.tokenize)

    assert act_tokenized_data == token_seqs
    assert act_queries == queries
    assert act_labels == labels

def test_extract_queries_from_explanations():
    text = 'First type of "query"'

    assert ['"query"'] == func.extract_queries_from_explanations(text)

    text = 'Another "type" of "query"'

    assert ['"type"', '"query"'] == func.extract_queries_from_explanations(text)

    text = 'Ideally all explanations will only use "double quote\'s", so we can avoid issues with "\'"'

    assert ['"double quote\'s"', '"\'"'] == func.extract_queries_from_explanations(text)

    text = "An explanation can use 'single quotes'"

    assert ["'single quotes'"] == func.extract_queries_from_explanations(text)

    text = "However, there can be some problems with 'apostrophes like, 's'"

    assert ["'apostrophes like, '"] == func.extract_queries_from_explanations(text)

    text = "We can even handle ''double single quotes too''"

    assert ["'double single quotes too'"] == func.extract_queries_from_explanations(text)

    text = "Though do \"not\" mix 'quotes'"

    assert ['"not"'] == func.extract_queries_from_explanations(text)

    text = "Finally we also handle `backticks as quotes`"

    assert ["`backticks as quotes`"] == func.extract_queries_from_explanations(text)

    text = "No quotes here though, so should be empty"

    assert [] == func.extract_queries_from_explanations(text)