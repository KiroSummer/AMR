# add by kiro @2020.10.30
from supar import Parser
import json
import codecs
import sys
from collections import OrderedDict
from stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator import FeatureAnnotator


compound_file = 'data/AMR/amr_2.0_utils/joints.txt'
annotator = FeatureAnnotator('http://localhost:9000', compound_file)


class srl_example:
    def __init__(self, obj):
        self.speakers = obj["speakers"]
        self.doc_key = obj["doc_key"]
        self.sentences = obj["sentences"][0]
        self.srl = obj["srl"][0]
        self.constituents = obj["constituents"]
        self.clusters = obj["clusters"]
        self.tokens = None
        self.lemmas = None
        self.pos_tags = None
        self.ner_tags = None
        self.dependency_edges = None

    def write_json(self, file_path):
        output = OrderedDict()
        output['doc_key'] = self.doc_key
        output['sentence'] = self.sentences
        output['srl'] = self.srl
        output['tokens'] = self.tokens
        output['lemmas'] = self.lemmas
        output['pos_tags'] = self.pos_tags
        output['ner_tags'] = self.ner_tags
        output['dependency_edges'] = self.dependency_edges
        file_path.write(json.dumps(output) + '\n')


class DependencyParser:
    """
    DependencyParser that based on supar, to predict the automatic dependency parsing results for AMR text
    """
    def __init__(self):
        self.parser = Parser.load("biaffine-dep-bert-en")

    def parse(self, filepath):
        """
        Data loading with json format. then dependency parsing.
        """
        sentence_number = 0
        with codecs.open(filepath, encoding="utf8") as f:
            print("read srl file from {}".format(filepath))
            with codecs.open(filepath[:-len(".jsonlines")] + '.dep.json', encoding='utf8', mode='w') as out_f:
                print("parse and write file to {}".format(filepath[:-len(".json")] + '.dep.json'))
                for line in f.readlines():
                    sentence_number += 1
                    if sentence_number % 1000 == 0:
                        print("processed {} sentences".format(sentence_number))
                    sen = json.loads(line)
                    srl_sen = srl_example(sen)
                    self.parser_sentence(srl_sen)  # parse sentence
                    # stanford parser parse
                    annotation = annotator(' '.join(srl_sen.sentences))
                    srl_sen.tokens = annotation['tokens']
                    srl_sen.lemmas = annotation['lemmas']
                    assert len(srl_sen.tokens) == len(srl_sen.lemmas) == len(srl_sen.sentences)
                    srl_sen.pos_tags = annotation['pos_tags']
                    srl_sen.ner_tags = annotation['ner_tags']
                    srl_sen.write_json(out_f)
                print("{} total sentences number {}".format(filepath, sentence_number))

    def parser_sentence(self, srl_sen):
        tokens = srl_sen.sentences
        result = self.parser.predict([tokens], verbose=False, tree=True, proj=True)
        srl_sen.dependency_edges = result.arcs[0]


if __name__ == "__main__":
    srl_file = sys.argv[1]  # srl file path
    parser = DependencyParser()
    parser.parse(srl_file)


