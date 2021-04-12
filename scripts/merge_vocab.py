import os, sys, math

from ..parser.data import DUM, END, CLS, NIL, PAD, UNK


class Vocab(object):
    def __init__(self, name):
        self.name = name
        self._priority = dict()  # for what token -> count mapping ? @kiro

    def read_from_file(self, filename):
        for line in open(filename).readlines():
            token, cnt = line.rstrip('\n').split('\t')
            cnt = int(cnt)
            if token in self._priority:
                self._priority[token] += cnt
            else:
                self._priority[token] = int(cnt)

    def write_to_file(self, filename):
        with open(filename, 'w') as output_file:
            sorted_p = sorted(self._priority.items(), key=lambda item: item[1], reverse=True)
            for x in sorted_p:
                output_file.write(x[0] + '\t' + str(math.ceil(x[1] / 3.0)) + '\n')


if __name__ == "__main__":
    folder1 = sys.argv[1]
    folder2 = sys.argv[2]
    folder3 = sys.argv[3]
    output_dir = sys.argv[4]

    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    vocabs = dict()

    vocabs['tok'] = Vocab('tok_vocab')  # remove the token frequence < 5 @kiro
    vocabs['lem'] = Vocab('lem_vocab')
    vocabs['pos'] = Vocab('pos_vocab')
    vocabs['ner'] = Vocab('ner_vocab')
    vocabs['dep_rel'] = Vocab('dep_rel_vocab')
    vocabs['predictable_concept'] = Vocab('predictable_concept_vocab')
    vocabs['predictable_word'] = Vocab('predictable_word_vocab')  # for AMR-to-Text @kiro
    vocabs['concept'] = Vocab('concept_vocab')
    vocabs['rel'] = Vocab('rel_vocab')
    vocabs['word_char'] = Vocab('word_char_vocab')
    vocabs['lem_char'] = Vocab('lem_char_vocab')
    vocabs['concept_char'] = Vocab('concept_char_vocab')

    def read_from_dir(dir_path):
        for file in os.listdir(dir_path):
            if file.endswith('_vocab'):
                file_prefix = file[:-6]
                print('read from {} / {}'.format(dir_path, file_prefix))
                vocabs[file_prefix].read_from_file(file)

    read_from_dir(folder1)
    read_from_dir(folder2)
    read_from_dir(folder3)

    for vocab in vocabs:
        vocabs[vocab].write_to_file(os.path.join(output_dir, vocabs[vocab].name))

