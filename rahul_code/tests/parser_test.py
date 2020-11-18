import sys
sys.path.append("../")
import json
import pickle
from CCG_new.parser import CCGParserTrainer, TrainedCCGParser
from CCG_new.utils import prepare_token_for_rule_addition

ec_ccg_trainer = CCGParserTrainer(task="ec", explanation_file="data/ec_test_data.json",
                               unlabeled_data_file="data/carer_test_data.json")

def test_trainer_load_data():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    loaded_data = ec_ccg_trainer.parser.loaded_data
    assert len(loaded_data) == 16
    for datapoint in loaded_data:
        assert "DataPoint" in str(type(datapoint))

def test_trainer_prepare_unlabeled_data():
    unlabeled_data_file = ec_ccg_trainer.params["unlabeled_data_file"]
    ec_ccg_trainer.prepare_unlabeled_data(unlabeled_data_file)
    unlabeled_data = ec_ccg_trainer.unlabeled_data
    assert len(unlabeled_data) == 1000
    for phrase in unlabeled_data:
        assert "Phrase" in str(type(phrase))

def test_parser_create_and_set_grammar():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser_grammar = parser.grammar
    with open(explanation_file) as f:
        explanation_data = json.load(f)
    
    for dic in explanation_data:
        token = "\"" + dic["word"].lower() + "\""
        prepped_token = prepare_token_for_rule_addition(dic["word"].lower())
        assert prepped_token in parser_grammar


def test_parser_tokenize_explanations():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    loaded_data = parser.loaded_data
    expected_ec_tokenizations = [
        [['$The', '$Sentence', '$Contains', '$The', '$Word', '"angry"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"increasingSPACEanger"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"beenSPACEsoSPACEexcited"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"whenSPACEtheSPACEgtaSPACEonlineSPACEbikerSPACEdlcSPACEcomesSPACEout"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"bigot"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"howSPACEshit"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"terrorSPACEthreat"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"panic"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"smiling"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"cheerySPACEsmile"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"soSPACEsad"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"mySPACEdepression"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"bitSPACEsurprised"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"soSPACEsudden"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"honesty"']],
         [['$The', '$Word', '"dependentSPACEonSPACEanotherSPACEperson"', '$In', '$The', '$Sentence']]
    ]

    assert len(loaded_data) == 16
    for i, datapoint in enumerate(loaded_data):
        assert datapoint.tokenized_explanations == expected_ec_tokenizations[i]

def test_parser_build_labeling_rules():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    parser.set_standard_parser()
    parser.build_labeling_rules()
    loaded_data = parser.loaded_data

    semantic_counts = [
        {('.root', ('@In0', 'Sentence', ('@Word', 'angry'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'increasing anger'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'been so excited'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'when the gta online biker dlc comes out'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'bigot'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'how shit'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'terror threat'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'panic'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'smiling'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'cheery smile'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'so sad'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'my depression'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'bit surprised'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'so sudden'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'honesty'))): 38},
        {('.root', ('@In0', 'Sentence', ('@Word', 'dependent on another person'))): 28},
    ]

    assert len(loaded_data) == 16

    for i, datapoint in enumerate(loaded_data):
        assert datapoint.semantic_counts == semantic_counts[i]
        # This check should be improved, assumes all semantic reps are labeling functions
        keys = list(datapoint.semantic_counts.keys())
        for key in keys:
            assert key in datapoint.labeling_functions
    
def test_filter_matrix():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    parser.set_standard_parser()
    parser.build_labeling_rules()
    with open("data/carer_test_data_phrase.p", "rb") as f:
        unlabeled_data = pickle.load(f)
    parser.matrix_filter(unlabeled_data)
    # filter count set to zero, so certain explanations don't fire at all on this small sample
    # hence hash filter filters some out
    label_counts = {"anger" : 2, "fear" : 1, "joy" : 1, "sadness" : 1, "surprise" : 1, "trust" : 1}
    final_labeling_functions = parser.labeling_functions
    final_label_counts = {}
    for key in final_labeling_functions:
        assert "function" in str(type(key))
        emotion = final_labeling_functions[key]
        if emotion in final_label_counts:
            final_label_counts[emotion] += 1
        else:
            final_label_counts[emotion] = 1
    
    for key in final_label_counts:
        assert final_label_counts[key] == label_counts[key]
    
    parser.low_end_filter_count = 1
    parser.labeling_functions = None
    parser.matrix_filter(unlabeled_data)
    # filter count set to 1, so certain explanations don't fire at all on this small sample
    # hence count filter filters some out
    # anger has one less count, as hash keeps the first misfiring one
    label_counts = {"anger" : 1, "fear" : 1, "joy" : 1, "sadness" : 1, "surprise" : 1, "trust" : 1}
    final_labeling_functions = parser.labeling_functions
    final_label_counts = {}
    for key in final_labeling_functions:
        assert "function" in str(type(key))
        emotion = final_labeling_functions[key]
        if emotion in final_label_counts:
            final_label_counts[emotion] += 1
        else:
            final_label_counts[emotion] = 1
    
    for key in final_label_counts:
        assert final_label_counts[key] == label_counts[key]
