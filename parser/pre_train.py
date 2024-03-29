import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import argparse, os, random
from parser.data import Vocab, DataLoader, SRLDataLoader, DUM, END, CLS, NIL, DynamicDataLoader
from parser.parser import Parser
from parser.work import show_progress
from parser.extract import LexicalMap
from parser.adam import AdamWeightDecayOptimizer
from parser.utils import move_to_device, MyThread, eval
from parser.bert_utils import BertEncoderTokenizer, BertEncoder
from parser.postprocess import PostProcessor
from parser.work import parse_data


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', type=str)
    parser.add_argument('--tok_vocab', type=str)
    parser.add_argument('--lem_vocab', type=str)
    parser.add_argument('--pos_vocab', type=str)
    parser.add_argument('--ner_vocab', type=str)
    parser.add_argument('--dep_rel_vocab', type=str)
    parser.add_argument('--srl_vocab', type=str)
    parser.add_argument('--concept_vocab', type=str)
    parser.add_argument('--predictable_concept_vocab', type=str)
    parser.add_argument('--predictable_word_vocab', type=str)
    parser.add_argument('--rel_vocab', type=str)
    parser.add_argument('--word_char_vocab', type=str)
    parser.add_argument('--concept_char_vocab', type=str)
    parser.add_argument('--pretrained_file', type=str, default=None)
    parser.add_argument('--with_bert', dest='with_bert', action='store_true')
    parser.add_argument('--bert_path', type=str, default=None)

    parser.add_argument('--encoder_graph', dest='encoder_graph', action='store_true')
    parser.add_argument('--decoder_graph', dest='decoder_graph', action='store_true')
    parser.add_argument('--no_post_process', dest='no_post_process', action='store_true')

    parser.add_argument('--use_srl', dest='use_srl', action='store_true')
    parser.add_argument('--use_gold_predicates', dest='use_gold_predicates', action='store_true')
    parser.add_argument('--use_gold_arguments', dest='use_gold_arguments', action='store_true')
    parser.add_argument('--soft_mtl', dest='soft_mtl', action='store_true')
    parser.add_argument('--loss_weights', dest='loss_weights', action='store_true')
    parser.add_argument('--sum_loss', dest='sum_loss', action='store_true')

    parser.add_argument('--word_char_dim', type=int)
    parser.add_argument('--word_dim', type=int)
    parser.add_argument('--pos_dim', type=int)
    parser.add_argument('--ner_dim', type=int)
    parser.add_argument('--dep_rel_dim', type=int)
    parser.add_argument('--concept_char_dim', type=int)
    parser.add_argument('--concept_dim', type=int)
    parser.add_argument('--rel_dim', type=int)

    parser.add_argument('--cnn_filters', type=int, nargs='+')
    parser.add_argument('--char2word_dim', type=int)
    parser.add_argument('--char2concept_dim', type=int)

    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--snt_layers', type=int)
    parser.add_argument('--graph_layers', type=int)
    parser.add_argument('--inference_layers', type=int)

    parser.add_argument('--pred_size', type=int)
    parser.add_argument('--argu_size', type=int)
    parser.add_argument('--span_size', type=int)
    parser.add_argument('--ffnn_size', type=int)
    parser.add_argument('--ffnn_depth', type=int)

    parser.add_argument('--dropout', type=float)
    parser.add_argument('--unk_rate', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--silver_train_data', type=str)
    parser.add_argument('--silver_data_loss_weight', type=float)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--srl_data', type=str)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--batches_per_update', type=int)
    parser.add_argument('--dev_batch_size', type=int)
    parser.add_argument('--lr_scale', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--eval_every', type=int)

    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=int)

    return parser.parse_args()


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def update_lr(optimizer, lr_scale, embed_size, steps, warmup_steps):
    lr = lr_scale * embed_size ** -0.5 * min(steps ** -0.5, steps * (warmup_steps ** -1.5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def data_proc(data, queue):
    while True:
        for x in data:
            queue.put(x)
        queue.put('EPOCHDONE')


def load_vocabs(args):
    vocabs = dict()
    vocabs['tok'] = Vocab(args.tok_vocab, 5, [CLS])  # remove the token frequence < 5 @kiro
    vocabs['lem'] = Vocab(args.lem_vocab, 5, [CLS])
    vocabs['pos'] = Vocab(args.pos_vocab, 5, [CLS])
    vocabs['ner'] = Vocab(args.ner_vocab, 5, [CLS])
    vocabs['dep_rel'] = Vocab(args.dep_rel_vocab, 5, [CLS])
    if args.use_srl:
        vocabs['srl'] = Vocab(args.srl_vocab, 50, [NIL])
    vocabs['predictable_concept'] = Vocab(args.predictable_concept_vocab, 5, [DUM, END])
    vocabs['predictable_word'] = Vocab(args.predictable_word_vocab, 5, [DUM, END])  # for AMR-to-Text @kiro
    vocabs['concept'] = Vocab(args.concept_vocab, 5, [DUM, END])
    vocabs['rel'] = Vocab(args.rel_vocab, 50, [NIL])
    vocabs['word_char'] = Vocab(args.word_char_vocab, 100, [CLS, END])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 100, [CLS, END])
    lexical_mapping = LexicalMap()
    bert_encoder = None
    if args.with_bert:
        bert_tokenizer = BertEncoderTokenizer.from_pretrained(args.bert_path, do_lower_case=False)
        vocabs['bert_tokenizer'] = bert_tokenizer
    for name in vocabs:
        if name == 'bert_tokenizer':
            continue
        print((name, vocabs[name].size, vocabs[name].coverage))
    return vocabs, lexical_mapping


def main(local_rank, args):
    vocabs, lexical_mapping = load_vocabs(args)
    bert_encoder = None
    if args.with_bert:
        bert_encoder = BertEncoder.from_pretrained(args.bert_path)
        for p in bert_encoder.parameters():  # fix bert @kiro
            p.requires_grad = False

    torch.manual_seed(19940117)
    torch.cuda.manual_seed_all(19940117)
    random.seed(19940117)
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)  # totally read @kiro
    print("#"*25)
    print("Concerned important config details")
    print("use graph encoder?", args.encoder_graph)
    print("use graph decoder?", args.decoder_graph)
    print("use srl for MTL?", args.use_srl)
    print("use_gold_predicates?", args.use_gold_predicates)
    print("use_gold_arguments?", args.use_gold_arguments)
    print("soft mtl?", args.soft_mtl)
    print("sum loss?", args.sum_loss)
    print("loss_weights?", args.loss_weights)
    print("silver_data_loss_weight", args.silver_data_loss_weight)
    print("#"*25)

    model = Parser(vocabs,
                   args.word_char_dim, args.word_dim, args.pos_dim, args.ner_dim, args.dep_rel_dim,
                   args.concept_char_dim, args.concept_dim,
                   args.cnn_filters, args.char2word_dim, args.char2concept_dim,
                   args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout,
                   args.snt_layers, args.graph_layers, args.inference_layers, args.rel_dim,
                   args.pretrained_file, bert_encoder,
                   device, args.sum_loss,
                   False)
    print(Parser)

    if args.world_size > 1:
        torch.manual_seed(19940117 + dist.get_rank())
        torch.cuda.manual_seed_all(19940117 + dist.get_rank())
        random.seed(19940117 + dist.get_rank())

    model = model.cuda(local_rank)
    dev_data = DataLoader(vocabs, lexical_mapping, args.dev_data, args.dev_batch_size, for_train=False)  # load data @kiro
    pp = PostProcessor(vocabs['rel'])

    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():
        if name.endswith('bias') or 'layer_norm' in name:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    grouped_params = [{'params': weight_decay_params, 'weight_decay': args.weight_decay},
                      {'params': no_weight_decay_params, 'weight_decay': 0.}]
    optimizer = AdamWeightDecayOptimizer(grouped_params, 1., betas=(0.9, 0.999), eps=1e-6)  # "correct" L2 @kiro

    used_batches = 0
    batches_acm = 0

    if args.resume_ckpt:  # false, not supported @kiro
        ckpt = torch.load(args.resume_ckpt)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        batches_acm = ckpt['batches_acm']
        del ckpt

    silver_file = open(args.silver_train_data, 'r')
    print("read silver file from {}".format(args.silver_train_data))
    silver_train_data = DynamicDataLoader(
        vocabs, lexical_mapping, silver_file, args.train_batch_size, for_train=True
    )
    silver_train_data.set_unk_rate(args.unk_rate)
    silver_queue = mp.Queue(10)
    silver_train_data_generator = mp.Process(target=data_proc, args=(silver_train_data, silver_queue))
    silver_data_loss_weight = 1.0 if args.silver_data_loss_weight is None else args.silver_data_loss_weight

    silver_train_data_generator.start()
    eval_tool = eval('%s/%s' % (args.ckpt, "checkpoint.txt"), args.dev_data, )
    model.train()
    epoch, loss_avg, srl_loss_avg, concept_loss_avg, arc_loss_avg, rel_loss_avg, concept_repr_loss_avg =\
        0, 0, 0, 0, 0, 0, 0
    silver_loss_avg, silver_concept_loss_avg, silver_arc_loss_avg, silver_rel_loss_avg, silver_concept_repr_loss_avg = \
        0, 0, 0, 0, 0
    max_training_epochs = int(args.epochs)  # @kiro、
    print("Start training...")
    is_start = True
    while epoch < max_training_epochs:  # there is no stop! @kiro fixed by me
        while True:
            batch = silver_queue.get()
            if isinstance(batch, str):
                silver_train_data_generator.terminate()
                silver_train_data_generator.join()

                # read the next sample batches
                silver_train_data = DynamicDataLoader(
                    vocabs, lexical_mapping, silver_file, args.train_batch_size, for_train=True
                )
                silver_train_data.set_unk_rate(args.unk_rate)
                silver_queue = mp.Queue(10)
                silver_train_data_generator = mp.Process(target=data_proc, args=(silver_train_data, silver_queue))
                silver_train_data_generator.start()

                if args.world_size == 1 or (dist.get_rank() == 0):
                    if len(silver_train_data.data) < 20000:
                        epoch += 1
                        model.eval()
                        output_dev_file = '%s/epoch%d_batch%d_dev_out' % (args.ckpt, epoch, batches_acm)
                        parse_data(model, pp, dev_data, args.dev_data, output_dev_file, args)

                        saved_model = '%s/epoch%d_batch%d' % (args.ckpt, epoch, batches_acm)
                        torch.save({'args': args,
                                    'model': model.state_dict(),
                                    'batches_acm': batches_acm,
                                    'optimizer': optimizer.state_dict()},
                                   saved_model)

                        eval_task = MyThread(eval_tool.eval, (output_dev_file, saved_model, not args.no_post_process))
                        eval_task.start()
                        model.train()
                        print('epoch', epoch, 'done', 'batches', batches_acm)
                print('batches', batches_acm)
            else:
                batch = move_to_device(batch, model.device)  # data moved to device
                silver_concept_loss, silver_arc_loss, silver_rel_loss, silver_graph_arc_loss = model.forward(
                    batch, encoder_graph=args.encoder_graph, decoder_graph=args.decoder_graph)
                # model forward, please note that graph_arc_loss is not used
                loss = (silver_concept_loss + silver_arc_loss + silver_rel_loss) / args.batches_per_update  # compute
                loss_value = loss.item()
                silver_concept_loss_value = silver_concept_loss.item()
                silver_arc_loss_value = silver_arc_loss.item()
                silver_rel_loss_value = silver_rel_loss.item()
                # concept_repr_loss_value = concept_repr_loss.item()
                silver_loss_avg = silver_loss_avg * args.batches_per_update * 0.8 + 0.2 * loss_value
                silver_concept_loss_avg = silver_concept_loss_avg * 0.8 + 0.2 * silver_concept_loss_value
                silver_arc_loss_avg = silver_arc_loss_avg * 0.8 + 0.2 * silver_arc_loss_value
                silver_rel_loss_avg = silver_rel_loss_avg * 0.8 + 0.2 * silver_rel_loss_value
                # concept_repr_loss_avg = concept_repr_loss_avg * 0.8 + 0.2 * concept_repr_loss_value
                loss = silver_data_loss_weight * loss
                loss.backward()  # loss backward
                used_batches += 1
                if not (used_batches % args.batches_per_update == -1 % args.batches_per_update):
                    continue

                batches_acm += 1
                if args.world_size > 1:
                    average_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = update_lr(optimizer, args.lr_scale, args.embed_dim, batches_acm, args.warmup_steps)
                optimizer.step()  # update the model parameters according to the losses @kiro
                optimizer.zero_grad()
                if args.world_size == 1 or (dist.get_rank() == 0):
                    if batches_acm % args.print_every == -1 % args.print_every:
                        print('Train Epoch %d, Batch %d, LR %.6f, conc_loss %.3f, arc_loss %.3f, rel_loss %.3f, concept_repr_loss %.3f, srl_loss %.3f' % (
                        epoch, batches_acm, lr, concept_loss_avg, arc_loss_avg, rel_loss_avg, concept_repr_loss_avg, srl_loss_avg))
                        print('==============>, silver_conc_loss %.3f, silver_arc_loss %.3f, silver_rel_loss %.3f' % (
                                silver_concept_loss_avg, silver_arc_loss_avg, silver_rel_loss_avg)
                              )
                        model.train()
                    # if (batches_acm > 100 or args.resume_ckpt is not None) and batches_acm % args.eval_every == -1 % args.eval_every:
                break
    silver_train_data_generator.terminate()
    silver_train_data_generator.join()
    print("Training process is done.")  # @kiro


def init_processes(local_rank, args, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    main(local_rank, args)


if __name__ == "__main__":
    args = parse_config()
    if not os.path.exists(args.ckpt):  # create the ckpt dir @kiro
        os.mkdir(args.ckpt)
    assert len(args.cnn_filters) % 2 == 0
    args.cnn_filters = list(zip(args.cnn_filters[:-1:2], args.cnn_filters[1::2]))
    gpu_number = torch.cuda.device_count()
    print("number of available GPUs", gpu_number)
    args.world_size = args.gpus = gpu_number
    if args.world_size == 1:
        main(0, args)
        exit(0)
    mp.spawn(init_processes, args=(args,), nprocs=args.gpus)
