#!/usr/bin/env python
# coding: utf-8
from collections import Counter
import json, re

from parser.amr import AMR
from parser.AMRGraph import AMRGraph, number_regexp
from parser.AMRGraph import _is_abs_form
from parser.srl import read_srl_file


class AMRIO:

    def __init__(self):
        pass

    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('# ::id '):
                    amr_id = line[len('# ::id '):]
                elif line.startswith('# ::snt '):
                    sentence = line[len('# ::snt '):]
                elif line.startswith('# ::tokens '):
                    tokens = json.loads(line[len('# ::tokens '):])
                elif line.startswith('# ::lemmas '):
                    lemmas = json.loads(line[len('# ::lemmas '):])
                    lemmas = [le if _is_abs_form(le) else le.lower() for le in lemmas]
                elif line.startswith('# ::pos_tags '):
                    pos_tags = json.loads(line[len('# ::pos_tags '):])
                elif line.startswith('# ::ner_tags '):
                    ner_tags = json.loads(line[len('# ::ner_tags '):])
                elif line.startswith('# ::dependency_edges '):
                    dependency_edges = json.loads(line[len('# ::dependency_edges '):])
                elif line.startswith('# ::dependency_rels '):
                    dependency_rels = json.loads(line[len('# ::dependency_rels '):])
                elif line.startswith('# ::abstract_map '):
                    abstract_map = json.loads(line[len('# ::abstract_map '):])
                    graph_line = AMR.get_amr_line(f)  # read the AMR string lines @kiro
                    amr = AMR.parse_AMR_line(graph_line)
                    myamr = AMRGraph(amr)
                    yield tokens, lemmas, pos_tags, ner_tags, dependency_edges, dependency_rels, myamr


class LexicalMap(object):
    # build our lexical mapping (from token/lemma to concept), useful for copy mechanism.
    def __init__(self):
        pass

    # cp_seq, mp_seq, token2idx, idx2token = lex_map.get(lemma, token, vocabs['predictable_concept'])
    def get_concepts(self, lem, tok, vocab=None):
        cp_seq, mp_seq = [], []
        new_tokens = set()
        for le, to in zip(lem, tok):
            cp_seq.append(le + '_')
            mp_seq.append(le)

        if vocab is None:
            return cp_seq, mp_seq

        for cp, mp in zip(cp_seq, mp_seq):
            if vocab.token2idx(cp) == vocab.unk_idx:
                new_tokens.add(cp)
            if vocab.token2idx(mp) == vocab.unk_idx:
                new_tokens.add(mp)
        nxt = vocab.size
        token2idx, idx2token = dict(), dict()
        for x in new_tokens:
            token2idx[x] = nxt
            idx2token[nxt] = x
            nxt += 1
        return cp_seq, mp_seq, token2idx, idx2token


def dynamically_read_file(f, max_sentence_length=50000):
    """
    dynamically read a big amr file
    """
    line = f.readline()
    if not line:
        f.seek(0)  # move the file pointer to the file head
        line = f.readline()
    sample_count = 0
    token, lemma, pos, ner, edges, dep_rels, amrs = [], [], [], [], [], [], []
    while True:
        if not line:  # end of file
            break
        line = line.rstrip()
        if line.startswith('# ::id '):
            amr_id = line[len('# ::id '):]
        elif line.startswith('# ::snt '):
            sentence = line[len('# ::snt '):]
        elif line.startswith('# ::tokens '):
            tokens = json.loads(line[len('# ::tokens '):])
        elif line.startswith('# ::lemmas '):
            lemmas = json.loads(line[len('# ::lemmas '):])
            lemmas = [le if _is_abs_form(le) else le.lower() for le in lemmas]
        elif line.startswith('# ::pos_tags '):
            pos_tags = json.loads(line[len('# ::pos_tags '):])
        elif line.startswith('# ::ner_tags '):
            ner_tags = json.loads(line[len('# ::ner_tags '):])
        elif line.startswith('# ::dependency_edges '):
            dependency_edges = json.loads(line[len('# ::dependency_edges '):])
        elif line.startswith('# ::dependency_rels '):
            dependency_rels = json.loads(line[len('# ::dependency_rels '):])
        elif line.startswith('# ::abstract_map '):
            abstract_map = json.loads(line[len('# ::abstract_map '):])
        # else:
            graph_line = AMR.get_amr_line(f)  # read the AMR string lines @kiro
            amr = AMR.parse_AMR_line(graph_line)
            myamr = AMRGraph(amr)

            sample_count += 1
            token.append(tokens)
            lemma.append(lemmas)
            pos.append(pos_tags)
            ner.append(ner_tags)
            edges.append(dependency_edges)
            dep_rels.append(dependency_rels)
            amrs.append(myamr)
            if sample_count >= max_sentence_length:
                break
        line = f.readline()
    return amrs, token, lemma, pos, ner, edges, dep_rels


def read_file(filename):
    # read preprocessed amr file
    token, lemma, pos, ner, edges, dep_rels, amrs = [], [], [], [], [], [], []
    for _tok, _lem, _pos, _ner, _edges, _dep_rels, _myamr in AMRIO.read(filename):
        token.append(_tok)
        lemma.append(_lem)
        pos.append(_pos)
        ner.append(_ner)
        edges.append(_edges)
        dep_rels.append(_dep_rels)
        amrs.append(_myamr)
    print('read from %s, %d amrs' % (filename, len(token)))
    return amrs, token, lemma, pos, ner, edges, dep_rels


def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n' % (x, y))


import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--srl_data', type=str)  # add srl data
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    amrs, token, lemma, pos, ner, _, dep_rels = read_file(args.train_data)
    lexical_map = LexicalMap()

    # collect concepts and relations
    conc = []
    rel = []
    predictable_conc = []
    for i in range(10):
        # run 10 times random sort to get the priorities of different types of edges
        for amr, lem, tok in zip(amrs, lemma, token):
            concept, edge, not_ok = amr.root_centered_sort()
            lexical_concepts = set()
            cp_seq, mp_seq = lexical_map.get_concepts(lem, tok)
            for lc, lm in zip(cp_seq, mp_seq):
                lexical_concepts.add(lc)
                lexical_concepts.add(lm)

            if i == 0:
                predictable_conc.append([c for c in concept if c not in lexical_concepts])
                conc.append(concept)
            rel.append([e[-1] for e in edge])

    # read srl data
    if args.srl_data != 'no':
        srl_tokens, srl_lemmas, srl_pos, srl_ner, _, srl_srl_tags, _ = read_srl_file(args.srl_data)
        token.extend(srl_tokens)
        lemma.extend(srl_lemmas)
        pos.extend(srl_pos)
        ner.extend(srl_ner)
        srl_vocab = make_vocab(srl_srl_tags)
        write_vocab(srl_vocab, 'srl_vocab')
    # make vocabularies
    token_vocab, token_char_vocab = make_vocab(token, char_level=True)
    lemma_vocab, lemma_char_vocab = make_vocab(lemma, char_level=True)
    pos_vocab = make_vocab(pos)
    ner_vocab = make_vocab(ner)
    dep_rel_vocab = make_vocab(dep_rels)
    conc_vocab, conc_char_vocab = make_vocab(conc, char_level=True)

    predictable_conc_vocab = make_vocab(predictable_conc)
    num_predictable_conc = sum(len(x) for x in predictable_conc)
    num_conc = sum(len(x) for x in conc)
    print('predictable concept coverage', num_predictable_conc, num_conc, num_predictable_conc / num_conc)
    rel_vocab = make_vocab(rel)

    print('make vocabularies')
    write_vocab(token_vocab, 'tok_vocab')
    write_vocab(token_char_vocab, 'word_char_vocab')
    write_vocab(lemma_vocab, 'lem_vocab')
    write_vocab(lemma_char_vocab, 'lem_char_vocab')
    write_vocab(pos_vocab, 'pos_vocab')
    write_vocab(ner_vocab, 'ner_vocab')
    write_vocab(conc_vocab, 'concept_vocab')
    write_vocab(conc_char_vocab, 'concept_char_vocab')
    write_vocab(predictable_conc_vocab, 'predictable_concept_vocab')
    write_vocab(rel_vocab, 'rel_vocab')
    write_vocab(dep_rel_vocab, 'dep_rel_vocab')
