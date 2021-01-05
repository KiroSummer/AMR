import random
import torch
from torch import nn
import numpy as np
from parser.AMRGraph import AMRGraph
from parser.extract import read_file
from parser.srl import read_srl_file

PAD, UNK, DUM, NIL, END, CLS = '<PAD>', '<UNK>', '<DUMMY>', '<NULL>', '<END>', '<CLS>'
GPU_SIZE = 12000  # okay for 8G memory


class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials=None):
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._priority = dict()  # for what token -> count mapping ? @kiro
        num_tot_tokens = 0
        num_vocab_tokens = 0
        for line in open(filename).readlines():
            try:
                token, cnt = line.rstrip('\n').split('\t')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                print(line)
            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_vocab_tokens += cnt
            self._priority[token] = int(cnt)
        self.coverage = num_vocab_tokens / num_tot_tokens
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    def priority(self, x):
        return self._priority.get(x, 0)

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)


def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad] * (max_len - len(x))
        ys.append(y)
    data = np.transpose(np.array(ys))
    return data


def ListsofStringToTensor(xs, vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD] * (max_len - len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([CLS] + z + [END]) + [vocab.padding_idx] * (max_string_len - len(z)))
        ys.append(zs)

    data = np.transpose(np.array(ys), (1, 0, 2))
    return data


def ArraysToTensor(xs):
    "list of numpy array, each has the same demonsionality"
    x = np.array([list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis=0))
    data = np.zeros(shape, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i + 1)] + [slice(0, x) for x in slicing_shape])
        data[slices] = x
    # tensor = torch.from_numpy(data).long()
    return data


def ArraysToTensorWithPadding(xs, padding=0):
    "list of numpy array, each has the same demonsionality"
    x = np.array([list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis=0))
    data = np.zeros(shape, dtype=np.int)
    data.fill(padding)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i + 1)] + [slice(0, x) for x in slicing_shape])
        data[slices] = x
    # tensor = torch.from_numpy(data).long()
    return data


def batchify(data, vocabs, unk_rate=0.):  # batchify the data
    _tok = ListsToTensor([[CLS] + x['tok'] for x in data], vocabs['tok'], unk_rate=unk_rate)
    _lem = ListsToTensor([[CLS] + x['lem'] for x in data], vocabs['lem'], unk_rate=unk_rate)
    _pos = ListsToTensor([[CLS] + x['pos'] for x in data], vocabs['pos'], unk_rate=unk_rate)
    _ner = ListsToTensor([[CLS] + x['ner'] for x in data], vocabs['ner'], unk_rate=unk_rate)
    _edges = ArraysToTensorWithPadding([np.array(x['edge']) for x in data], padding=-1)  # edges to batch Tensor. @kiro
    _word_char = ListsofStringToTensor([[CLS] + x['tok'] for x in data], vocabs['word_char'])

    local_token2idx = [x['token2idx'] for x in data]
    local_idx2token = [x['idx2token'] for x in data]
    _cp_seq = ListsToTensor([x['cp_seq'] for x in data], vocabs['predictable_concept'], local_token2idx)
    _mp_seq = ListsToTensor([x['mp_seq'] for x in data], vocabs['predictable_concept'], local_token2idx)

    concept, edge = [], []
    for x in data:
        amr = x['amr']
        concept_i, edge_i, _ = amr.root_centered_sort(vocabs['rel'].priority)
        concept.append(concept_i)
        edge.append(edge_i)

    augmented_concept = [[DUM] + x + [END] for x in concept]  # for the start and end of the generation process @kiro

    _concept_in = ListsToTensor(augmented_concept, vocabs['concept'], unk_rate=unk_rate)[:-1]
    _concept_char_in = ListsofStringToTensor(augmented_concept, vocabs['concept_char'])[:-1]
    _concept_out = ListsToTensor(augmented_concept, vocabs['predictable_concept'], local_token2idx)[1:]  # out no $start

    out_conc_len, bsz = _concept_out.shape  # out_conc_len, concept_0, concept_1, ..., concept_n-1, <end>
    _rel = np.full((1 + out_conc_len, bsz, out_conc_len), vocabs['rel'].token2idx(PAD))
    # v: [<dummy>, concept_0, ..., concept_l, ..., concept_{n-1}, <end>] u: [<dummy>, concept_0, ..., concept_l, ..., concept_{n-1}]

    for bidx, (x, y) in enumerate(zip(edge, concept)):
        for l, _ in enumerate(y):
            if l > 0:
                # l=1 => pos=l+1=2
                _rel[l + 1, bidx, 1:l + 1] = vocabs['rel'].token2idx(NIL)
        for v, u, r in x:
            r = vocabs['rel'].token2idx(r)
            _rel[v + 1, bidx, u + 1] = r

    ret = {'lem': _lem, 'tok': _tok, 'pos': _pos, 'ner': _ner, 'edge': _edges, 'word_char': _word_char, \
           'copy_seq': np.stack([_cp_seq, _mp_seq], -1), \
           'local_token2idx': local_token2idx, 'local_idx2token': local_idx2token, \
           'concept_in': _concept_in, 'concept_char_in': _concept_char_in, \
           'concept_out': _concept_out, 'rel': _rel}

    bert_tokenizer = vocabs.get('bert_tokenizer', None)
    if bert_tokenizer is not None:
        ret['bert_token'] = ArraysToTensor([x['bert_token'] for x in data])
        ret['token_subword_index'] = ArraysToTensor([x['token_subword_index'] for x in data])
    return ret


class DataLoader(object):
    def __init__(self, vocabs, lex_map, filename, batch_size, for_train):
        self.data = []
        bert_tokenizer = vocabs.get('bert_tokenizer', None)
        for amr, token, lemma, pos, ner, edge in zip(*read_file(filename)):
            if for_train:
                _, _, not_ok = amr.root_centered_sort()
                if not_ok or len(token) == 0:
                    continue
            cp_seq, mp_seq, token2idx, idx2token = lex_map.get_concepts(lemma, token, vocabs['predictable_concept'])
            datum = {'amr': amr, 'tok': token, 'lem': lemma, 'pos': pos, 'ner': ner, 'edge': edge, \
                     'cp_seq': cp_seq, 'mp_seq': mp_seq, \
                     'token2idx': token2idx, 'idx2token': idx2token}
            if bert_tokenizer is not None:
                bert_token, token_subword_index = bert_tokenizer.tokenize(token)
                datum['bert_token'] = bert_token
                datum['token_subword_index'] = token_subword_index

            self.data.append(datum)
        print("Get %d AMRs from %s" % (len(self.data), filename))
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train
        self.unk_rate = 0.

    def set_unk_rate(self, x):
        self.unk_rate = x

    def __iter__(self):
        idx = list(range(len(self.data)))

        if self.train:
            random.shuffle(idx)
            idx.sort(key=lambda x: len(self.data[x]['tok']) + len(self.data[x]['amr'])) 

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(self.data[i]['tok']) + len(self.data[i]['amr'])  # tokens_num = len(toks) + len(concepts)
            data.append(self.data[i])
            if num_tokens >= self.batch_size:
                sz = len(data) * (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
                if sz > GPU_SIZE:
                    # because we only have limited GPU memory
                    batches.append(data[:len(data) // 2])
                    data = data[len(data) // 2:]
                batches.append(data)
                num_tokens, data = 0, []
        if data:
            sz = len(data) * (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
            if sz > GPU_SIZE:
                # because we only have limited GPU memory
                batches.append(data[:len(data) // 2])
                data = data[len(data) // 2:]
            batches.append(data)

        if self.train:  # but the samples in each batch are always the same? @kiro TODO
            random.shuffle(batches)
            print("Total {} training batches.".format(len(batches)))

        for batch in batches:
            yield batchify(batch, self.vocabs, self.unk_rate)


def get_all_gold_predicates(srl):
    if len(srl) > 0:
        predicates, _, _, _ = zip(*srl)
    else:
        predicates = []
    predicates = np.unique(predicates)
    return predicates, len(predicates)


def get_all_gold_arguments(srl):
    arguments = []
    for srl_label in srl:  # remove self-loop V-V
        if srl_label[-1] in ["V", "C-V"]:
            continue
        arguments.append([int(srl_label[1]), int(srl_label[2])])
    if len(arguments) == 0:  # if the sentence has no arguments.
        return [[], []]
    tokenized_arg_starts, tokenized_arg_ends = zip(*arguments)
    return tokenized_arg_starts, tokenized_arg_ends


def tokenize_argument_spans(srl, dictionary):
    srl_span = []
    for srl_label in srl:  # remove self-loop V-V
        if srl_label[-1] in ["V", "C-V"]:
            continue
        srl_span.append([int(srl_label[0]), int(srl_label[1]), int(srl_label[2]),
                         int(dictionary.token2idx(srl_label[3]))]
                        )
    if len(srl_span) == 0:  # if the sentence has no arguments.
        return [[], [], [], []]
    tokenized_predicates, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels = zip(*srl_span)
    return tokenized_predicates, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels


def batchify_srl(data, vocabs, unk_rate=0.):  # batchify the data
    _tok = ListsToTensor([[CLS] + x['tok'] for x in data], vocabs['tok'], unk_rate=unk_rate)
    _lem = ListsToTensor([[CLS] + x['lem'] for x in data], vocabs['lem'], unk_rate=unk_rate)
    _pos = ListsToTensor([[CLS] + x['pos'] for x in data], vocabs['pos'], unk_rate=unk_rate)
    _ner = ListsToTensor([[CLS] + x['ner'] for x in data], vocabs['ner'], unk_rate=unk_rate)
    _edges = ArraysToTensorWithPadding([np.array(x['edge']) for x in data], padding=-1)  # edges to batch Tensor. @kiro
    _word_char = ListsofStringToTensor([[CLS] + x['tok'] for x in data], vocabs['word_char'])

    # srl tuple
    _predicates = [get_all_gold_predicates(x['srl']) for x in data]
    # _argument_starts_ends = [get_all_gold_arguments(x['srl']) for x in data]
    _srl_tuple = [tokenize_argument_spans(x['srl'], vocabs['srl']) for x in data]

    batch_sample_size = len(data)
    max_sample_length = max([len(x['tok']) for x in data])
    max_sample_arg_number = max([len(x['srl']) for x in data])
    max_sample_gold_predicates_number = max([_pred[1] for _pred in _predicates])
    # gold predicates
    padded_gold_predicates = np.zeros([batch_sample_size, max_sample_gold_predicates_number], dtype=np.int)
    padded_num_gold_predicates = np.zeros(batch_sample_size, dtype=np.int)
    # target predicates
    padded_target_predicates = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)

    padded_predicates = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int)
    padded_arg_starts = np.zeros([batch_sample_size, max_sample_arg_number],
                                 dtype=np.int)  # one for padding the arg start
    padded_arg_ends = np.zeros([batch_sample_size, max_sample_arg_number],
                               dtype=np.int)  # zero for padding the arg end
    padded_arg_labels = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int)
    padded_srl_lens = np.zeros(batch_sample_size, dtype=np.int)
    for i in range(batch_sample_size):
        # gold predicates
        padded_gold_predicates[i, : _predicates[i][1]] = _predicates[i][0]
        padded_num_gold_predicates[i] = _predicates[i][1]
        # target predicates
        for p in _predicates[i][0]:
            padded_target_predicates[i, p] = 1
        # output
        sample_arg_number = len(data[i]['srl'])
        padded_predicates[i, :sample_arg_number] = _srl_tuple[i][0]
        padded_arg_starts[i, :sample_arg_number] = _srl_tuple[i][1]
        padded_arg_ends[i, :sample_arg_number] = _srl_tuple[i][2]
        padded_arg_labels[i, :sample_arg_number] = _srl_tuple[i][3]
        padded_srl_lens[i] = sample_arg_number
    _labels = (padded_predicates, padded_arg_starts, padded_arg_ends, padded_arg_labels, padded_srl_lens)
    _gold_preds = (padded_gold_predicates, padded_num_gold_predicates, padded_target_predicates),

    ret = {'lem': _lem, 'tok': _tok, 'pos': _pos, 'ner': _ner, 'edge': _edges, 'word_char': _word_char,
           'gold_preds': _gold_preds, 'srl': _labels}

    bert_tokenizer = vocabs.get('bert_tokenizer', None)
    if bert_tokenizer is not None:
        ret['bert_token'] = ArraysToTensor([x['bert_token'] for x in data])
        ret['token_subword_index'] = ArraysToTensor([x['token_subword_index'] for x in data])
    return ret


class SRLDataLoader(object):
    """
    For Loading SRL data @kiro
    """
    def __init__(self, vocabs, filename, batch_size, for_train):
        self.data = []
        bert_tokenizer = vocabs.get('bert_tokenizer', None)
        for token, lemma, pos, ner, edge, _, srl in zip(*read_srl_file(filename)):
            datum = {'tok': token, 'lem': lemma, 'pos': pos, 'ner': ner, 'edge': edge, 'srl': srl}
            if bert_tokenizer is not None:
                bert_token, token_subword_index = bert_tokenizer.tokenize(token)
                datum['bert_token'] = bert_token
                datum['token_subword_index'] = token_subword_index

            self.data.append(datum)
        print("Get %d SRLs from %s" % (len(self.data), filename))
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train
        self.unk_rate = 0.

    def set_unk_rate(self, x):
        self.unk_rate = x

    def __iter__(self):
        idx = list(range(len(self.data)))

        if self.train:
            random.shuffle(idx)
            idx.sort(key=lambda x: len(self.data[x]['tok']))  # fixed idx? @kiro check TODO

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(self.data[i]['tok'])  # tokens_num = len(toks) + len(concepts)
            data.append(self.data[i])
            if num_tokens >= self.batch_size:
                sz = len(data) * (2 + max(len(x['tok']) for x in data))
                if sz > GPU_SIZE:
                    # because we only have limited GPU memory
                    batches.append(data[:len(data) // 2])
                    data = data[len(data) // 2:]
                batches.append(data)
                num_tokens, data = 0, []
        if data:
            sz = len(data) * (2 + max(len(x['tok']) for x in data))
            if sz > GPU_SIZE:
                # because we only have limited GPU memory
                batches.append(data[:len(data) // 2])
                data = data[len(data) // 2:]
            batches.append(data)

        if self.train:  # but the samples in each batch are always the same? @kiro TODO
            random.shuffle(batches)
            print("Total {} training batches.".format(len(batches)))

        for batch in batches:
            yield batchify_srl(batch, self.vocabs, self.unk_rate)
