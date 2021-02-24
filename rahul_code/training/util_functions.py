import spacy
from torchtext.data import Field, Example, Dataset
import pickle
import re
import torch


nlp = spacy.load("en_core_web_sm")

def find_array_start_position(big_array, small_array):
    """
        Find the starting index of a sub_array inside of a larger array

        Returns -1 if the small_array is not contrained within the larger array

        Arguments:
            big_array   (arr) : the larger array to search through
            small_array (arr) : the smaller array to find
        
        Returns:
            int : start position of small_array if small_array is within big_array, else -1
    """
    small_array_len = len(small_array)
    cut_off = len(big_array) - small_array_len
    for i, elem in enumerate(big_array):
        if i <= cut_off:
            if elem == small_array[0]:
                if big_array[i:i+small_array_len] == small_array:
                    return i
        else:
            break

    return -1

def generate_save_string(embedding_name, random_state=-1, sample=-1.0):
    """
        To allow for multiple datasets to exist at once, we add this string to identify which dataset a run
        script should load.

        Arguments:
            embedding_name (str) : name of pre-trained embedding to use
                                   (possible names can be found in possible_embeddings)
            random_state   (int) : random state used to split data into train, dev, test (if applicable)
            sample       (float) : percentage of possible data used for training
    """
    return "_".join([embedding_name, str(random_state), str(sample)])

def clean_text(text):
    text = text.strip()
    text = text.lower()
    text = text.replace('\n', '')

    return text

def tokenize(sentence, tokenizer=nlp):
    """
        Simple tokenizer function that is needed to build a vocabulary
        Uses spaCy's en model to tokenize

        Arguments:
            sentence          (str) : input to be tokenized
            tokenizer (spaCy Model) : spaCy model to use for tokenization purposes

        Returns:
            arr : list of tokens
    """
    sentence = clean_text(sentence)
    return [tok.text for tok in tokenizer.tokenizer(sentence)]

def build_vocab(train, embedding_name, save_string="", save=True):
    """
        Note: Using the Field class will be deprecated soon by TorchText, however at the time of writing the
              new methodology for creating a vocabulary has not been released.
        
        Function that takes in training data and builds a TorchText Vocabulary object, which couples two
        important datastructures:
            1. Mapping from text token to token_id
            2. Mapping from token_id to vector

        Function expects a pre-computed set of vectors to be used in the mapping from token_id to vector

        Arguments:
            train          (arr) : array of text sequences that make up one's training data
            embedding_name (str) : name of pre-trained embedding to use 
                                   (possible names can be found in possible_embeddings)
            save_string    (str) : string to indicate some of the hyper-params used to create the vocab
        Returns:
            torchtext.vocab : vocab object
    """
    # text_field = Field(tokenize=tokenize, init_token = '<bos>', eos_token='<eos>')
    text_field = Field(tokenize=tokenize)
    fields = [("text", text_field)]
    train_examples = []
    for text in train:
        train_examples.append(Example.fromlist([text], fields))
    
    train_dataset = Dataset(train_examples, fields=fields)
    
    text_field.build_vocab(train_dataset, vectors=embedding_name)
    vocab = text_field.vocab

    print("Finished building vocab of size {}".format(str(len(vocab))))

    if save:
        file_name = "../data/pre_train_data/vocab_{}.p".format(save_string)

        with open(file_name, "wb") as f:
            pickle.dump(vocab, f)

    return vocab

def convert_text_to_tokens(data, vocab, tokenize_fn):
    """
        Converts sequences of text to sequences of token ids per the provided vocabulary

        Arguments:
            data              (arr) : sequences of text
            vocab (torchtext.vocab) : vocabulary object
            tokenize_fun (function) : function to use to break up text into tokens

        Returns:
            arr : array of arrays, each inner array is a token_id representation of the text passed in

    """
    word_seqs = [tokenize_fn(seq) for seq in data]
    token_seqs = [[vocab[word] for word in word_seq] for word_seq in word_seqs]

    return token_seqs

def extract_queries_from_explanations(explanation):
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
        possible_queries = [query[1:len(query)-1] for query in possible_queries]
        return possible_queries
    
    possible_queries = re.findall("'[^']+'", explanation)
    if len(possible_queries):
        possible_queries = [query[1:len(query)-1] for query in possible_queries]
        return possible_queries

    possible_queries = re.findall("`[^`]+`", explanation)
    if len(possible_queries):
        possible_queries = [query[1:len(query)-1] for query in possible_queries]
        return possible_queries

    return []

def similarity_loss_function(pos_scores, neg_scores):
    """
        L_sim in the NExT Paper

        Arguments:
            pos_scores (torch.tensor) : per query the max value of (tau - cos(q_li_j, q_li_k))^2
                                        dims: (n,)
            neg_scores (torch.tensor) : per query the max value of (cos(q_li_j, q_lk_m))^2
                                        dims: (n,)
        
        Returns:
            torch.tensor : average of the sum of scores per query, dims: (1,)
    """
    return torch.mean(pos_scores + neg_scores)