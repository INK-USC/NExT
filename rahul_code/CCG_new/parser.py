import numpy as np
import json
from CCG_new import constants
from CCG_new import utils
from CCG_new import util_classes as classes
import os
import pickle
import torch.nn as nn
import torch
import copy
from collections import namedtuple
from nltk.ccg import chart, lexicon
import spacy
import random
import numpy as np

nlp = spacy.load("en_core_web_sm")

class TrainedCCGParser():
    """
        A wrapper around an NLTK CCG Chart Parser
        Prepares the data for the parser and builds the grammar for the parser as well
        Includes a trainable theta vector so that the machine learns which parses of an explanation is
        most likely given the downstream task.

        Attributes:
            use_beam                      (bool) : whether to use beam search when figuring out best possible
                                                   parse theta vector is used here to score parses
            operation_index_map           (dict) : key - path in Logic Form Tree, value - id for path
            theta                       (vector) : trainable theta vector to evaluate the best possible parse
                                                   for a logical form
            loaded_data                   (list) : data that needs to be parsed, each item is a DataPoint object
            grammar                        (str) : multi-line string describing the grammar rules
            standard_ccg_parser (CCGChartParser) : NLTK CCGChart Parser
    """
    def __init__(self, low_end_filter_count=0, high_end_filter_pct=0.5, use_beam=False):
        self.loaded_data = None
        self.grammar = None
        self.standard_ccg_parser = None
        self.labeling_functions = None
        self.low_end_filter_count = low_end_filter_count
        self.high_end_filter_pct = high_end_filter_pct

    def load_data(self, data):
        self.loaded_data = data
    
    def create_and_set_grammar(self, init_grammar=constants.RAW_GRAMMAR):
        """
            Function that takes initial fixed grammar and adds some loaded_data specific rules to the grammar
            Rules associated with words found in explanations from loaded_data are added to the grammar
            After rules are added the class property grammar is set.

            Arguments:
                init_grammar (str) : initial grammar to use
        """
        quote_words = {}
        for i, triple in enumerate(self.loaded_data):
            explanation = triple.raw_explanation
            if len(explanation):
                chunks = utils.clean_and_chunk(explanation, nlp)
                self.loaded_data[i].chunked_explanation = chunks
                for chunk in chunks:
                    terminals = utils.convert_chunk_to_terminal(chunk)
                    if terminals:
                        if terminals[0].startswith("\"") and terminals[0].endswith("\""):
                            quote_words[terminals[0]] = 1
        
        self.grammar = utils.add_rules_to_grammar(quote_words, init_grammar)

    def set_standard_parser(self, rule_set=chart.DefaultRuleSet):
        lex = lexicon.fromstring(self.grammar, True)
        self.standard_ccg_parser = chart.CCGChartParser(lex, rule_set)
        
    def tokenize_explanations(self):
        """
            Assuming data has been loaded, and grammar and parser have been created, we start the process of
            parsing explanations into Labeling Functions. Steps performed by this function are:
                1. Explanation gets chunked (if needed)
                2. Explanations get tokenized
        """
        for i, datapoint in enumerate(self.loaded_data):
            if len(datapoint.raw_explanation):
                if datapoint.chunked_explanation:
                    chunks = datapoint.chunked_explanation
                else:
                    explanation = datapoint.raw_explanation
                    chunks = utils.clean_and_chunk(explanation, nlp)
                    self.loaded_data[i].chunked_explanation = chunks
                tokenizations = [[]]
                for chunk in chunks:
                    predicates = utils.convert_chunk_to_terminal(chunk)
                    if predicates:
                        if predicates[0].startswith("\"") and predicates[0].endswith("\""):
                            predicates[0] = utils.prepare_token_for_rule_addition(predicates[0])
                        if len(predicates) == 1:
                            for tokenization in tokenizations:
                                tokenization.append(predicates[0])
                        else:
                            temp_tokenizations = []
                            for tokenization in tokenizations:
                                for possible in predicates:
                                    tokenization_copy = tokenization[:]
                                    tokenization_copy.append(possible)
                                    temp_tokenizations.append(tokenization_copy)
                            tokenizations = temp_tokenizations

                self.loaded_data[i].tokenized_explanations = tokenizations

        
    def build_labeling_rules(self):
        """
            Assuming explanations have already been tokenized, and beam=False, we convert token sequences
            into labeling functions. Several token sequences are often mapped to the same labeling function,
            hence why we store semantic representations and labeling functions as keys in a dictionary. We 
            keep track of counts for training purposes. Steps performed by this function are:
                1. Token Sequences -> Parse Trees
                2. Parse Trees -> Semantic Representation
                3. Semantic Representation -> Labeling Function
        """
        for i, datapoint in enumerate(self.loaded_data):
            if len(datapoint.raw_explanation):
                tokenizations = self.loaded_data[i].tokenized_explanations
                logic_forms = []
                for tokenization in tokenizations:
                    try:
                        parses = list(self.standard_ccg_parser.parse(tokenization))
                        logic_forms += parses

                    except:
                        continue
                semantic_counts = {}
                if len(logic_forms):
                    semantic_counts = {}
                    for tree in logic_forms:
                        semantic_repr = utils.create_semantic_repr(tree)
                        if semantic_repr:
                            if semantic_repr in semantic_counts:
                                semantic_counts[semantic_repr] += 1
                            else:
                                semantic_counts[semantic_repr] = 1
                self.loaded_data[i].semantic_counts = semantic_counts

                labeling_functions = {}
                if len(semantic_counts):
                    for key in semantic_counts:
                        labeling_function = utils.create_labeling_function(key)
                        if labeling_function:
                            if labeling_function(datapoint.sentence): # filtering out labeling functions that don't even apply on their own datapoint
                                labeling_functions[key] = labeling_function                                        

                self.loaded_data[i].labeling_functions = labeling_functions
        
    def matrix_filter(self, unlabeled_data):
        """
        Version of BabbleLabbel's filter bank concept. Label Functions that don't apply to the original
        sentence that the explanation was written about have already been filtered out in build_labeling_rules.

        Filters out all functions that apply to more than high_end_filter_pct of datapoints
        Filters out all functions that don't apply to more than n number of datapoints

        For functions with the same output signature on the datapoints, we pick the first one, and filter out
        the rest.

        The remaining functions are then stored alongside their labels.
        """
        labeling_functions = []
        function_label_map = {}
        for i, datapoint in enumerate(self.loaded_data):
            labeling_functions_dict = datapoint.labeling_functions
            for key in labeling_functions_dict:
                function = labeling_functions_dict[key]
                labeling_functions.append(labeling_functions_dict[key])
                function_label_map[function] = datapoint.label
        
        matrix = [[] for i in range(len(labeling_functions))]

        for i, function in enumerate(labeling_functions):
            for entry in unlabeled_data:
                matrix[i].append(int(function(entry)))
        
        matrix = np.array(matrix, dtype=np.int32)

        row_sums = np.sum(matrix, axis=1)

        total_data_points = len(unlabeled_data)

        functions_to_delete = []
        for i, r_sum in enumerate(row_sums):
            if r_sum/total_data_points > self.high_end_filter_pct or r_sum < self.low_end_filter_count:
                functions_to_delete.append(i)
        
        # print("Total Hits {}".format(sum(row_sums)))
        
        # print("Count Filter {}".format(functions_to_delete))

        matrix = np.delete(matrix, functions_to_delete, 0)
        functions_to_delete.sort(reverse=True)
        for index in functions_to_delete:
            del labeling_functions[index]
        
        hashes = {}
        functions_to_delete = []
        for i, row in enumerate(matrix):
            row_hash = hash(str(row)) # same as babble-labbel
            if row_hash in hashes:
                functions_to_delete.append(i)
                # print("{} conflicted with {}".format(i, hashes[row_hash]))
            else:
                hashes[row_hash] = i

        # print("Hash Filter {}".format(functions_to_delete))
        functions_to_delete.sort(reverse=True)
        for index in functions_to_delete:
            del labeling_functions[index]

        self.labeling_functions = {}
        for function in labeling_functions:
            self.labeling_functions[function] = function_label_map[function]

class CCGParserTrainer():
    """
        Wrapper object to train a TrainedCCGParser object

        Attributes:
            params             (dict) : dictionary holding hyperparameters for training
            parser (TrainedCCGParser) : a TrainedCCGParser instance
    """
    def __init__(self, task, explanation_file, unlabeled_data_file):
        self.params = {}
        self.params["explanation_file"] = explanation_file
        self.params["unlabeled_data_file"] = unlabeled_data_file
        self.params["task"] = task
        # Temporary until data coming in standard
        if task == "ec":
            self.text_key = "tweet"
            self.exp_key = "explanation"
            self.label_key = "label"
        elif task == "re":
            self.text_key = "sent"
            self.exp_key = "exp"
            self.label_key = "relation"
        self.parser = TrainedCCGParser()
        self.unlabeled_data = []
    
    def load_data(self, path):
        with open(path) as f:
            data = json.load(f)
        
        processed_data = []
        for dic in data:
            text = dic[self.text_key]
            explanation = dic[self.exp_key]
            phrase_for_text = utils.generate_phrase(text, nlp)
            label = dic[self.label_key]
            data_point = classes.DataPoint(phrase_for_text, label, explanation)
            processed_data.append(data_point)
        
        self.parser.load_data(processed_data)
    
    def prepare_unlabeled_data(self, path):
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            phrase_for_text = utils.generate_phrase(entry, nlp)
            self.unlabeled_data.append(phrase_for_text)

    def train(self):
        self.load_data(self.params["explanation_file"])
        self.parser.create_and_set_grammar()
        self.parser.tokenize_explanations()
        self.parser.set_standard_parser()
        self.parser.build_labeling_rules()
        self.prepare_unlabeled_data(self.params["unlabeled_data_file"])
        self.parser.matrix_filter(self.unlabeled_data)
    
    def get_parser(self):
        return self.parser
