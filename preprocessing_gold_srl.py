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
        self.text = obj["sentences"][0]
        self.tokens = None
        self.lemmas = None
        self.pos_tags = None
        self.ner_tags = None
        self.dependency_edges = obj["dependency_edges"]
        self.srl = obj["srl"][0]

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


parser = Parser.load("biaffine-dep-bert-en")


def parser_sentence(srl_sen):
    tokens = srl_sen.tokens  # tokens
    result = parser.predict([tokens], verbose=False, tree=True, proj=True)
    srl_sen.dependency_edges = result.arcs[0]


def pruning_srl_samples(filepath):
    sentence_number, saved_number = 0, 0
    with open(filepath, 'r') as f:
        print("read srl file from {}".format(filepath))
        with codecs.open(filepath + '.features', encoding='utf8', mode='w') as out_f:
            print("parse and write file to {}".format(filepath + '.features'))
            for line in f:  # read line
                sentence_number += 1
                if sentence_number % 1000 == 0:
                    print("processed {} sentences".format(sentence_number))
                # stanford parser parse
                srl_sen = srl_example(json.loads(line))
                if len(srl_sen.srl) == 0:  # this sample without srl
                    continue
                annotation = annotator(srl_sen.text)
                srl_sen.tokens = annotation['tokens']
                assert srl_sen.tokens == srl_sen.text
                # remove sentence that with length > 200 or word char number > 20
                if len(srl_sen.tokens) > 200 or max([len(word) for word in srl_sen.tokens]) > 20:
                    continue
                srl_sen.lemmas = annotation['lemmas']
                srl_sen.pos_tags = annotation['pos_tags']
                srl_sen.ner_tags = annotation['ner_tags']
                parser_sentence(srl_sen)  # parse sentence

                saved_number += 1
                out_f.write(line)
            print("{} total sentences number {}".format(filepath, sentence_number))
            print("{} total sentences saved {}".format(filepath + '.features', saved_number))


if __name__ == "__main__":
    srl_file = sys.argv[1]  # srl file path
    pruning_srl_samples(srl_file)


