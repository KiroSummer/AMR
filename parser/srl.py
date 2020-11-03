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


def read_srl_file(filepath, token_vocab, lemma_vocab, pos_tags, ner_tags):
    with open(filepath, 'r') as f:
        sentence_number = 0
        print("read srl file from {}".format(filepath))
        for line in f:  # read line
            sentence_number += 1
            if sentence_number % 1000 == 0:
                print("processed {} sentences".format(sentence_number))
            # stanford parser parse
            srl_sen = srl_example(json.loads(line))
            token_vocab.append(srl_sen.tokens)
            lemma_vocab.append(srl_sen.lemmas)
            pos_tags.append(srl_sen.pos_tags)
            ner_tags.append(srl_sen.ner_tags)
        print("{} total sentences number {}".format(filepath, sentence_number))
