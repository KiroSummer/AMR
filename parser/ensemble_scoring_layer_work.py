import torch
import copy

from parser.data import Vocab, DataLoader, DUM, END, CLS, NIL
from parser.parser import Parser
from parser.postprocess import PostProcessor
from parser.extract import LexicalMap
from parser.utils import move_to_device
from parser.bert_utils import BertEncoderTokenizer, BertEncoder
from parser.match import match
import parser.global_variables as global_variables

from threading import Thread
import argparse, os, re, threading, time


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_batch_size', type=int)

    parser.add_argument('--encoder_graph', dest='encoder_graph', action='store_true')
    parser.add_argument('--decoder_graph', dest='decoder_graph', action='store_true')

    parser.add_argument('--use_srl', dest='use_srl', action='store_true')
    parser.add_argument('--use_gold_predicates', dest='use_gold_predicates', action='store_true')
    parser.add_argument('--use_gold_arguments', dest='use_gold_arguments', action='store_true')
    parser.add_argument('--soft_mtl', dest='soft_mtl', action='store_true')
    parser.add_argument('--loss_weights', dest='loss_weights', action='store_true')
    parser.add_argument('--sum_loss', dest='sum_loss', action='store_true')

    parser.add_argument('--beam_size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--max_time_step', type=int)
    parser.add_argument('--output_suffix', type=str)
    parser.add_argument('--device', type=int, default=0)

    return parser.parse_args()


def show_progress(model, dev_data):
    model.eval()
    loss_acm = 0.
    for batch in dev_data:
        batch = move_to_device(batch, model.device)
        concept_loss, arc_loss, rel_loss = model(batch)
        loss = concept_loss + arc_loss + rel_loss
        loss_acm += loss.item()
    print('total loss', loss_acm)
    return loss_acm


class work_thread(Thread):
    def __init__(self, func, args):
        super(work_thread, self).__init__()
        self.func = func
        self.args = args
        self.res = None

    def run(self):
        self.res = self.func(*self.args)

    def get_result(self):
        return self.res


class avg_matrixes():
    def __init__(self, model_num):
        self.matrixes = []
        self.num = model_num
        self.idx = 0

    def append(self, con_LL, arc_LL, rel_LL):
        self.matrixes.append([con_LL, arc_LL, rel_LL])
        self.idx += 1
        assert self.idx <= self.num

    def reset(self):
        self.matrixes = []
        self.idx = 0

    def avg(self):
        assert len(self.matrixes) == 2
        self.res_con_LL = sum([x[0] for x in self.matrixes]) / self.num
        self.res_arc_LL = sum([x[1] for x in self.matrixes]) / self.num
        self.res_rel_LL = sum([x[2] for x in self.matrixes]) / self.num
        self.next_time = time.time() + 5.0

    def return_ans(self):
        return self.res_con_LL, self.res_arc_LL, self.res_rel_LL, self.next_time


def parse_batch(models, batch, beam_size, alpha, max_time_step, args=None):
    res = dict()
    concept_batch = []
    relation_batch = []

    t1 = work_thread(models[0].work, args=(batch, beam_size, max_time_step, 1, args))
    t2 = work_thread(models[1].work, args=(batch, beam_size, max_time_step, 1, args))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    beams = t2.get_result()

    score_batch = []
    for beam in beams:
        best_hyp = beam.get_k_best(1, alpha)[0]
        predicted_concept = [token for token in best_hyp.seq[1:-1]]
        predicted_rel = []
        for i in range(len(predicted_concept)):
            if i == 0:
                continue
            arc = best_hyp.state_dict['arc_ll%d' % i].squeeze_().exp_()[1:]  # head_len
            rel = best_hyp.state_dict['rel_ll%d' % i].squeeze_().exp_()[1:, :]  # head_len x vocab
            for head_id, (arc_prob, rel_prob) in enumerate(zip(arc.tolist(), rel.tolist())):
                predicted_rel.append((i, head_id, arc_prob, rel_prob))
        concept_batch.append(predicted_concept)
        score_batch.append(best_hyp.score)
        relation_batch.append(predicted_rel)
    res['concept'] = concept_batch
    res['score'] = score_batch
    res['relation'] = relation_batch
    return res


def parse_data(models, pp, data, input_file, output_file, args, beam_size=8, alpha=0.6, max_time_step=100):
    tot = 0
    with open(output_file, 'w') as fo:
        for batch in data:
            batch = move_to_device(batch, models[0].device)
            res = parse_batch(models, batch, beam_size, alpha, max_time_step, args=args)
            for concept, relation, score in zip(res['concept'], res['relation'], res['score']):
                fo.write('# ::conc ' + ' '.join(concept) + '\n')
                fo.write('# ::score %.6f\n' % score)
                fo.write(pp.postprocess(concept, relation) + '\n\n')
                tot += 1
    match(output_file, input_file)
    print('write down %d amrs' % tot)


def load_ckpt_without_bert(model, test_model):
    ckpt = torch.load(test_model)['model']
    for k, v in model.state_dict().items():
        if k.startswith('bert_encoder'):
            ckpt[k] = v
    model.load_state_dict(ckpt)


if __name__ == "__main__":

    args = parse_config()

    test_models = []
    print("args.load_path", args.load_path)
    if os.path.isdir(args.load_path):
        for file in os.listdir(args.load_path):
            fname = os.path.join(args.load_path, file)
            if os.path.isfile(fname):
                test_models.append(fname)
        model_args = torch.load(fname)['args']
    else:
        for fname in args.load_path.split('+'):
            test_models.append(fname)
        model_args = torch.load(fname)['args']

    vocabs = dict()

    vocabs['tok'] = Vocab(model_args.tok_vocab, 5, [CLS])
    vocabs['lem'] = Vocab(model_args.lem_vocab, 5, [CLS])
    vocabs['pos'] = Vocab(model_args.pos_vocab, 5, [CLS])
    vocabs['ner'] = Vocab(model_args.ner_vocab, 5, [CLS])
    vocabs['dep_rel'] = Vocab(model_args.dep_rel_vocab, 5, [CLS])
    if args.use_srl:
        vocabs['srl'] = Vocab(model_args.srl_vocab, 50, [NIL])
    vocabs['predictable_concept'] = Vocab(model_args.predictable_concept_vocab, 5, [DUM, END])
    vocabs['concept'] = Vocab(model_args.concept_vocab, 5, [DUM, END])
    vocabs['rel'] = Vocab(model_args.rel_vocab, 50, [NIL])
    vocabs['word_char'] = Vocab(model_args.word_char_vocab, 100, [CLS, END])
    vocabs['concept_char'] = Vocab(model_args.concept_char_vocab, 100, [CLS, END])
    lexical_mapping = LexicalMap()

    bert_encoder = None

    if model_args.with_bert:
        model_args.bert_path = "bert-base-cased"
        bert_tokenizer = BertEncoderTokenizer.from_pretrained(model_args.bert_path, do_lower_case=False)
        bert_encoder = BertEncoder.from_pretrained(model_args.bert_path)
        vocabs['bert_tokenizer'] = bert_tokenizer

    if args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    if model_args.use_srl is True:
        model = Parser(vocabs,
                       model_args.word_char_dim, model_args.word_dim, model_args.pos_dim, model_args.ner_dim,  model_args.dep_rel_dim,
                       model_args.concept_char_dim, model_args.concept_dim,
                       model_args.cnn_filters, model_args.char2word_dim, model_args.char2concept_dim,
                       model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
                       model_args.snt_layers, model_args.graph_layers, model_args.inference_layers, model_args.rel_dim,
                       None, bert_encoder,
                       device, model_args.sum_loss,
                       True, model_args.soft_mtl, model_args.loss_weights,
                       model_args.pred_size, model_args.argu_size, model_args.span_size, vocabs['srl'].size,
                       model_args.ffnn_size, model_args.ffnn_depth, model_args.use_gold_predicates, model_args.use_gold_arguments)
    else:
        model = Parser(vocabs,
                       model_args.word_char_dim, model_args.word_dim, model_args.pos_dim, model_args.ner_dim,  model_args.dep_rel_dim,
                       model_args.concept_char_dim, model_args.concept_dim,
                       model_args.cnn_filters, model_args.char2word_dim, model_args.char2concept_dim,
                       model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
                       model_args.snt_layers, model_args.graph_layers, model_args.inference_layers, model_args.rel_dim,
                       None, bert_encoder,
                       device, model_args.sum_loss,
                       False)
    #
    # model = Parser(vocabs,
    #                model_args.word_char_dim, model_args.word_dim, model_args.pos_dim, model_args.ner_dim,
    #                model_args.concept_char_dim, model_args.concept_dim,
    #                model_args.cnn_filters, model_args.char2word_dim, model_args.char2concept_dim,
    #                model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
    #                model_args.snt_layers, model_args.graph_layers, model_args.inference_layers, model_args.rel_dim,
    #                bert_encoder=bert_encoder, device=device)

    # test_data = DataLoader(vocabs, lexical_mapping, args.test_data, args.test_batch_size, for_train=True)
    another_test_data = DataLoader(vocabs, lexical_mapping, args.test_data, args.test_batch_size, for_train=False)
    models = [copy.deepcopy(model) for _ in range(len(test_models))]
    dict_models_params = []
    for i, test_model in enumerate(test_models):
        print("i", test_model)
        # batch = int(re.search(r'batch([0-9])+', test_model)[0][5:])
        # epoch = int(re.search(r'epoch([0-9])+', test_model)[0][5:])

        load_ckpt_without_bert(models[i], test_model)
        dict_models_params.append(dict(models[i].named_parameters()))

        models[i] = models[i].cuda()
        models[i].eval()

    # loss = show_progress(model, test_data)
    pp = PostProcessor(vocabs['rel'])
    global_variables._init()
    global_variables.set_value('avg', avg_matrixes(2))

    parse_data(models, pp, another_test_data, args.test_data, args.test_data + '.' + args.output_suffix, args,
               args.beam_size, args.alpha, args.max_time_step)
