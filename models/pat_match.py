import json
import numpy as np
import constant
from CCG.constant import entity_type
from CCG.utils import parse_sem,recurse

class Pat_Match(object):
    def __init__(self, config, filt=None):
        self.config = config
        with open(config.pattern_file, "r") as fh:
            self.logic_forms = json.load(fh)
        if filt is not None:
            logic_forms = [logic_form for logic_form in self.logic_forms if constant.LABEL_TO_ID[logic_form[0]] not in filt]
            self.logic_forms = logic_forms
        for j in range(len(self.logic_forms)):
            self.logic_forms[j][1] = recurse(parse_sem(self.logic_forms[j][1]))
            self.logic_forms[j][0] = constant.LABEL_TO_ID[self.logic_forms[j][0]]
        self.entity_type = {constant.LABEL_TO_ID[rel]:text.split() for rel,text in entity_type.items()}
    def match(self, phrases):
        config = self.config
        num_lf = len(self.logic_forms)
        num_text = len(phrases)
        res = np.zeros([num_text, num_lf])
        pred = np.zeros([num_text, config.num_class])

        for i, lf_ in enumerate(self.logic_forms):
            rel, lf,_,_ = lf_
            labeling_function = lf

            for j, phrase in enumerate(phrases):
                if labeling_function(phrase)==1 and phrase.subj.endswith(self.entity_type[rel][0]) and phrase.obj.endswith(self.entity_type[rel][1]):
                    res[j, i] += 1
                    pred[j, rel] += 1

        none_zero = (np.amax(pred, axis=1) > 0).astype(np.int32)
        pred = np.argmax(pred, axis=1)
        pred = pred * none_zero
        return res, pred
