# add by kiro @2020.10.30, to pruning empty srl samples, and expand tokens, lemmas, pos, and ner vocabs
from supar import Parser
import json
import codecs
import sys
from collections import OrderedDict
from stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator import FeatureAnnotator


compound_file = 'data/AMR/amr_2.0_utils/joints.txt'
annotator = FeatureAnnotator('http://localhost:9000', compound_file)


class srl_example:
    def __init__(self, obj=None):
        self.text = obj["text"]
        self.tokens = obj["tokens"]
        self.lemmas = obj["lemmas"]
        self.pos_tags = obj["pos_tags"]
        self.ner_tags = obj["ner_tags"]
        self.dependency_edges = obj["dependency_edges"]
        self.srl = obj["srl"]

    def write_json(self, file_path):
        output = OrderedDict()
        output['text'] = self.text
        output['tokens'] = self.tokens
        output['lemmas'] = self.lemmas
        output['pos_tags'] = self.pos_tags
        output['ner_tags'] = self.ner_tags
        output['dependency_edges'] = self.dependency_edges
        output['srl'] = self.srl
        file_path.write(json.dumps(output) + '\n')


def pruning_srl_samples(filepath):
    sentence_number, saved_number = 0, 0
    with open(filepath, 'r') as f:
        print("read srl file from {}".format(filepath))
        with codecs.open(filepath + '.pruned', encoding='utf8', mode='w') as out_f:
            print("parse and write file to {}".format(filepath + '.pruned'))
            for line in f:  # read line
                sentence_number += 1
                if sentence_number % 1000 == 0:
                    print("processed {} sentences".format(sentence_number))
                # stanford parser parse
                srl_sen = srl_example(line)
                if len(srl_sen.srl) == 0:  # this sample without srl
                    continue
                saved_number += 1
                out_f.write(line)
            print("{} total sentences number {}".format(filepath, sentence_number))
            print("{} total sentences saved {}".format(filepath + '.pruned', saved_number))


if __name__ == "__main__":
    srl_file = sys.argv[1]  # srl file path
    pruning_srl_samples(srl_file)


