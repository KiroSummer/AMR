import json
import codecs
import sys
from collections import OrderedDict


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


