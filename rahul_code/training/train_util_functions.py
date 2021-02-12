import pickle
import json
import random
from training.util_functions import generate_save_string, convert_text_to_tokens, tokenize, extract_queries_from_explanations, clean_text
from training.constants import SPACY_NERS, SPACY_TO_TACRED, TACRED_NERS, TACRED_ENTITY_TYPES, PARSER_TRAIN_SAMPLE, UNMATCH_TYPE_SCORE, TACRED_LABEL_MAP, DEV_F1_SAMPLE
from training.util_classes import BaseVariableLengthDataset, UnlabeledTrainingDataset, TrainingDataset
import sys
sys.path.append("../")
from CCG_new.soft_grammar_functions import NER_LABEL_SPACE
from CCG_new.parser import CCGParserTrainer
from CCG_new.utils import generate_phrase
import spacy
import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
import dill
import pdb
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def batch_type_restrict(relation, phrase_inputs):
    entity_types = TACRED_ENTITY_TYPES[relation]
    entity_ids = (NER_LABEL_SPACE[entity_types[0]], NER_LABEL_SPACE[entity_types[1]])
    restrict_subj = torch.eq(torch.reshape(phrase_inputs[:,-2],[1,-1]), entity_ids[0]).float()
    restrict_obj = torch.eq(torch.reshape(phrase_inputs[:,-1],[1,-1]), entity_ids[1]).float()
    restrict_subj = restrict_subj+(1.0-restrict_subj)*UNMATCH_TYPE_SCORE
    restrict_obj = restrict_obj+(1.0-restrict_obj)*UNMATCH_TYPE_SCORE
    
    return restrict_subj*restrict_obj

def build_phrase_input(phrases, pad_idx):
    tokens = [phrase.tokens for phrase in phrases]
    ners = [phrase.ners for phrase in phrases]
    subj_posis = torch.tensor([phrase.subj_posi for phrase in phrases]).unsqueeze(1)
    obj_posis =  torch.tensor([phrase.obj_posi for phrase in phrases]).unsqueeze(1)
    subj =  torch.tensor([phrase.ners[phrase.subj_posi] for phrase in phrases]).unsqueeze(1)
    obj = torch.tensor([phrase.ners[phrase.obj_posi] for phrase in phrases]).unsqueeze(1)

    ner_pad = NER_LABEL_SPACE["<PAD>"]

    tokens = BaseVariableLengthDataset.variable_length_batch_as_tensors(tokens, pad_idx)
    ners = BaseVariableLengthDataset.variable_length_batch_as_tensors(ners, ner_pad)

    assert tokens.shape == ners.shape

    phrase_input = torch.cat([tokens, ners, subj_posis, obj_posis, subj, obj], dim=1)

    return phrase_input

def build_mask_mat_for_batch(seq_length):
    mask_mat = torch.zeros((seq_length, seq_length, seq_length))
    for i in range(seq_length):
        for j in range(seq_length):
            mask_mat[i,j,i:j+1] = 1
    mask_mat = mask_mat.float()

    return mask_mat

def set_ner_label_space(labels):
    if len(NER_LABEL_SPACE) > 0:
        largest_key = max(list(NER_LABEL_SPACE.values()))
    else:
        largest_key = -1
    current_key = largest_key + 1
    for label in labels:
        if label not in NER_LABEL_SPACE:
            NER_LABEL_SPACE[label] = current_key
            current_key += 1

# def _prepapre_label(label, label_map):
#     label_vec = [0.0]*len(label_map)
#     label_vec[label_map[label]] = 1.0

#     return label_vec

def _prepare_labels(labels, dataset="tacred"):
    converted_labels = []
    if dataset == "tacred":
        for label in labels:
            converted_labels.append(TACRED_LABEL_MAP[label])
    return converted_labels

def _prepare_unlabeled_data(unlabeled_data_phrases, vocab, special_words):
    seq_tokens = []
    seq_phrases = []
    for phrase in unlabeled_data_phrases:
        tokens = [token.lower() for token in phrase.tokens]
        ners = phrase.ners
        ners = [SPACY_TO_TACRED[ner] if ner in SPACY_TO_TACRED else ner for ner in ners]
        tokens = [vocab[token] if token not in special_words else special_words[token] for token in tokens]
        ners = [NER_LABEL_SPACE[ner] for ner in ners]
        seq_tokens.append(tokens)
        phrase.update_tokens_and_ners(tokens, ners)
        seq_phrases.append(phrase)
    
    return seq_tokens, seq_phrases

def set_tacred_ner_label_space():
    temp = SPACY_NERS[:]
    for i, entry in enumerate(temp):
        if entry in SPACY_TO_TACRED:
            temp[i] = SPACY_TO_TACRED[entry]
    temp.append("<PAD>")
    set_ner_label_space(temp)
    set_ner_label_space(TACRED_NERS)

def create_parser(parser_training_data, explanation_path, dataset="tacred"):
    parser_trainer = None
    
    if dataset == "tacred":
        parser_trainer = CCGParserTrainer("re", explanation_path, "", True, parser_training_data)
    
    parser_trainer.train()
    parser = parser_trainer.get_parser()

    return parser

def match_training_data(labeling_functions, train):

    phrases = [generate_phrase(entry, nlp) for entry in train]

    matched_data_tuples = []
    unlabeled_data_phrases = []

    for phrase in phrases:
        not_matched = True
        for function in labeling_functions:
            try:
                if function(phrase):
                    matched_data_tuples.append((phrase.sentence, labeling_functions[function]))
                    not_matched = False
                    break
            except:
                continue
        if not_matched: 
            unlabeled_data_phrases.append(phrase)

    return matched_data_tuples, unlabeled_data_phrases

def build_unlabeled_dataset(unlabeled_data_phrases, vocab, save_string, special_words):
    pad_idx = vocab["<pad>"]
    seq_tokens, seq_phrases = _prepare_unlabeled_data(unlabeled_data_phrases, vocab, special_words)
    dataset = UnlabeledTrainingDataset(seq_tokens, seq_phrases, pad_idx)

    print("Finished building unlabeled dataset of size: {}".format(str(len(seq_tokens))))

    file_name = "../data/training_data/unlabeled_data_{}.p".format(save_string)

    with open(file_name, "wb") as f:
        pickle.dump(dataset, f)

def build_labeled_dataset(sentences, labels, vocab, save_string, split, special_words, dataset="tacred"):
    pad_idx = vocab["<pad>"]
    sentences = [clean_text(sentence) for sentence in sentences]
    seq_tokens = convert_text_to_tokens(sentences, vocab, tokenize, special_words)
    label_ids = _prepare_labels(labels, dataset)

    dataset = TrainingDataset(seq_tokens, label_ids, pad_idx)

    print("Finished building {} dataset of size: {}".format(split, str(len(seq_tokens))))

    file_name = "../data/training_data/{}_data_{}.p".format(split, save_string)

    with open(file_name, "wb") as f:
        pickle.dump(dataset, f)

def build_word_to_idx(raw_explanations, vocab, save_string):
    quoted_words = []
    for i, expl in enumerate(raw_explanations):
        queries = extract_queries_from_explanations(expl)
        for query in queries:
            query = clean_text(query)
            quoted_words.append(query)
    
    tokenized_queries = convert_text_to_tokens(quoted_words, vocab, tokenize)

    print("Finished tokenizing actual queries, count: {}".format(str(len(tokenized_queries))))

    file_name = "../data/training_data/query_tokens_{}.p".format(save_string)
    with open(file_name, "wb") as f:
        pickle.dump(tokenized_queries, f)

    quoted_words_to_index = {}
    for i, quoted_word in enumerate(quoted_words):
        quoted_words_to_index[quoted_word] = i
    
    file_name = "../data/training_data/word2idx_{}.p".format(save_string)
    with open(file_name, "wb") as f:
        pickle.dump(quoted_words_to_index, f)

def build_datasets_from_splits(train_path, dev_path, test_path, vocab_path, explanation_path, save_string,
                               label_filter=None, sample_rate=-1.0, dataset="tacred"):
    
    with open(train_path) as f:
        train = json.load(f)
        train = [entry["text"] for entry in train]
    
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    if dataset == "tacred":
        subj_idx = len(vocab)
        obj_idx = subj_idx + 1
        special_words = {"subj" : subj_idx, "obj" : obj_idx}

    parser_training_data = random.sample(train, min(PARSER_TRAIN_SAMPLE, len(train)))
    
    parser = create_parser(parser_training_data, explanation_path, dataset)

    # with open("parser.p", "rb") as f:
    #     parser = dill.load(f)
    
    strict_labeling_functions = parser.labeling_functions
    
    train_sample = None
    if sample_rate > 0:
        sample_number = int(len(train) * sample_rate)
        train_sample = random.sample(train, sample_number)

    if train_sample:
        matched_data_tuples, unlabeled_data_phrases = match_training_data(strict_labeling_functions, train_sample)
    else:
        matched_data_tuples, unlabeled_data_phrases = match_training_data(strict_labeling_functions, train)

    # with open("matched_data_tuples.p", "rb") as f:
    #     matched_data_tuples = pickle.load(f)
    
    # with open("unlabeled_data.p", "rb") as f:
    #     unlabeled_data_phrases = pickle.load(f)
    
    build_unlabeled_dataset(unlabeled_data_phrases, vocab, save_string, special_words)

    build_labeled_dataset([tup[0] for tup in matched_data_tuples], 
                          [tup[1] for tup in matched_data_tuples],
                          vocab, save_string, "matched",  special_words, dataset)

    with open(dev_path) as f:
        dev = json.load(f)
    
    build_labeled_dataset([ent["text"] for ent in dev], 
                          [ent["label"] for ent in dev],
                          vocab, save_string, "dev", special_words, dataset)
    
    with open(test_path) as f:
        test = json.load(f)
    
    build_labeled_dataset([ent["text"] for ent in test], 
                          [ent["label"] for ent in test],
                          vocab, save_string, "test", special_words, dataset)

    filtered_raw_explanations = parser.filtered_raw_explanations

    build_word_to_idx(filtered_raw_explanations, vocab, save_string)

    soft_matching_functions = parser.soft_labeling_functions

    function_labels = _prepare_labels([entry[1] for entry in soft_matching_functions], dataset)

    file_name = "../data/training_data/labeling_functions_{}.p".format(save_string)
    with open(file_name, "wb") as f:
        dill.dump({"function_pairs" : soft_matching_functions,
                     "labels" : function_labels}, f)

def _apply_no_relation_label(values, preds, label_map, no_relation_key, threshold, entropy=True):
    if entropy:
        no_relation_mask = values > threshold
    else:
        no_relation_mask = values < threshold

    final_preds = [label_map[no_relation_key] if mask else preds[i] for i, mask in enumerate(no_relation_mask)]

    return final_preds

def evaluate_next_clf(data_path, model, label_map, no_relation_thresholds=None,
                      batch_size=128, no_relation_key="no_relation"):

    if len(no_relation_key) == 0:
        label_space = [i for i in range(0,41)]
    else:
        label_space = [i for i in range(0,42)]

    with open(data_path, "rb") as f:
        eval_dataset = pickle.load(f)
    
    # deactivate dropout layers
    model.eval()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if no_relation_thresholds == None:
        no_relation_thresholds = prep_and_tune_no_relation_threshold(model, eval_dataset, device, batch_size,\
                                                                    label_map, no_relation_key)
    
    entropy_threshold, max_value_threshold = no_relation_thresholds

    total_ent_f1_score, total_val_f1_score = 0, 0
    total_class_probs = []
    batch_count = 0
    for step, batch in enumerate(tqdm(eval_dataset.as_batches(batch_size=batch_size, shuffle=False))):
        batch = [r.to(device) for r in batch]
        tokens, batch_labels = batch
        with torch.no_grad():
            preds = model.forward(tokens)

            class_probs = nn.functional.softmax(preds, dim=1)
            max_probs = torch.max(class_probs, dim=1).values.cpu().numpy()
            entropy = torch.sum(class_probs * -1.0 * torch.log(class_probs), axis=1).cpu().numpy()
            class_preds = torch.argmax(class_probs, dim=1).cpu().numpy()
            
            entropy_final_class_preds = _apply_no_relation_label(max_probs, class_preds, label_map,\
                                                                 no_relation_key, entropy_threshold)

            max_value_final_class_preds = _apply_no_relation_label(max_probs, class_preds, label_map,\
                                                                   no_relation_key, max_value_threshold, False)
            
            f1_labels = batch_labels.cpu().numpy()
            entropy_f1_score = metrics.f1_score(f1_labels, entropy_final_class_preds, labels=label_space, average='macro')
            max_value_f1_score = metrics.f1_score(f1_labels, max_value_final_class_preds, labels=label_space, average='macro')

            total_ent_f1_score += entropy_f1_score
            total_val_f1_score += max_value_f1_score
            batch_count += 1
            total_class_probs.append(class_probs.cpu().numpy())
    
    avg_ent_f1_score = total_ent_f1_score / batch_count
    avg_val_f1_score = total_val_f1_score / batch_count
    total_class_probs = np.concatenate(total_class_probs, axis=0)

    return avg_ent_f1_score, avg_val_f1_score, total_class_probs, no_relation_thresholds

def prep_and_tune_no_relation_threshold(model, eval_dataset, device, batch_size, label_map, no_relation_key):
    
    if len(no_relation_key) == 0:
        return int(-1.0 * np.log(1/len(label_map)) / step) + 1, 0.0
    
    entropy_values = []
    max_prob_values = []
    predict_labels = []
    labels = []
    
    sample_number = min(DEV_F1_SAMPLE, eval_dataset.length)

    model.eval()

    for step, batch in enumerate(tqdm(eval_dataset.as_batches(batch_size=batch_size, shuffle=False, sample=sample_number))):
        tokens, batch_labels = batch
        tokens = tokens.to(device)

        with torch.no_grad():
            preds = model.forward(tokens) # b x c
            class_probs = nn.functional.softmax(preds, dim=1)
            max_probs = torch.max(class_probs, dim=1).values
            entropy = torch.sum(class_probs * -1.0 * torch.log(class_probs), axis=1)
            class_preds = torch.argmax(class_probs, dim=1)
            
            entropy_values.append(entropy.cpu().numpy())
            max_prob_values.append(max_probs.cpu().numpy())
            predict_labels.append(class_preds.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    
    entropy_values = np.concatenate(entropy_values).ravel()
    max_prob_values = np.concatenate(max_prob_values).ravel()
    predict_labels = np.concatenate(predict_labels).ravel()
    labels = np.concatenate(labels).ravel()

    no_relation_threshold_entropy, _ = tune_no_relation_threshold(entropy_values, predict_labels, labels,\
                                                                  label_map, no_relation_key)

    no_relation_threshold_max_value, _ = tune_no_relation_threshold(max_prob_values, predict_labels, labels,\
                                                                    label_map, no_relation_key, False)

    return no_relation_threshold_entropy, no_relation_threshold_max_value
    
def tune_no_relation_threshold(values, preds, labels, label_map, no_relation_key="no_relation", entropy=True):
    step = 0.01
    if entropy:
        max_entropy_cut_off = int(-1.0 * np.log(1/len(label_map)) / step) + 1
        thresholds = [0.01 * i for i in range(1, max_entropy_cut_off)]
    else:
        thresholds = [0.01 * i for i in range(1, 99)]
    best_f1 = 0
    best_threshold = -1
    for threshold in thresholds:
        final_preds = _apply_no_relation_label(values, preds, label_map, no_relation_key, threshold, entropy)
        f1_score = metrics.f1_score(labels, final_preds, average='macro')

        if f1_score > best_f1:
            best_threshold = threshold
            best_f1 = f1_score
    
    return best_threshold, best_f1