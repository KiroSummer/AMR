import torch
from torch import nn
import torch.nn.functional as F
import math


from parser.encoder import WordEncoder, ConceptEncoder
# from parser.decoder import DecodeLayer
from parser.pure_decoder import DecodeLayer
from parser.transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask
from parser.data import ListsToTensor, ListsofStringToTensor, DUM, NIL, PAD
from parser.search import Hypothesis, Beam, search_by_batch
from parser.utils import move_to_device, generate_adj, generate_directional_self_loop_adj
from parser.srl import SRL_module


class Parser(nn.Module):
    def __init__(self, vocabs, word_char_dim, word_dim, pos_dim, ner_dim,
                 concept_char_dim, concept_dim,
                 cnn_filters, char2word_dim, char2concept_dim,
                 embed_dim, ff_embed_dim, num_heads, dropout,
                 snt_layers, graph_layers, inference_layers, rel_dim,
                 pretrained_file=None, bert_encoder=None,
                 device=0, sum_loss=False,
                 use_srl=False, soft_mtl=False, loss_weight=False,
                 pred_size=0, argu_size=0, span_size=0, label_space_size=0,
                 ffnn_size=0, ffnn_depth=0, use_gold_predicates=False, use_gold_arguments=False):
        super(Parser, self).__init__()
        self.vocabs = vocabs

        self.word_encoder = WordEncoder(vocabs,
                                        word_char_dim, word_dim, pos_dim, ner_dim,
                                        embed_dim, cnn_filters, char2word_dim, dropout, pretrained_file)

        self.concept_encoder = ConceptEncoder(vocabs,
                                              concept_char_dim, concept_dim, embed_dim,
                                              cnn_filters, char2concept_dim, dropout, pretrained_file)
        self.amr_snt_encoder = Transformer(snt_layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.graph_encoder = Transformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True,
                                         weights_dropout=False)
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, device=device)
        self.word_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.concept_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_mask = SelfAttentionMask(device=device)
        self.num_heads = num_heads
        self.decoder = DecodeLayer(vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, concept_dim, rel_dim,
                                   dropout)
        self.dropout = dropout
        self.probe_generator = nn.Linear(embed_dim, embed_dim)
        self.device = device
        self.bert_encoder = bert_encoder
        self.sum_loss = sum_loss
        if bert_encoder is not None:
            self.bert_adaptor = nn.Linear(768, embed_dim)
        self.use_srl = use_srl
        self.soft_mtl = soft_mtl
        self.loss_weight = loss_weight
        if self.use_srl:
            self.pred_size = pred_size
            self.argu_size = argu_size
            self.span_size = span_size
            self.label_space_size = label_space_size
            self.ffnn_size = ffnn_size
            self.ffnn_depth = ffnn_depth
            self.dropout = dropout
            self.use_gold_predicates = use_gold_predicates
            self.use_gold_arguments = use_gold_arguments
            if self.soft_mtl is True:
                self.srl_sent_encoder = Transformer(snt_layers, embed_dim, ff_embed_dim, num_heads, dropout)
                self.srl_probe_generator = nn.Linear(embed_dim, embed_dim)
            self.srl = SRL_module(self.embed_dim, self.pred_size, self.argu_size, self.span_size, self.label_space_size,
                                  self.ffnn_size, self.ffnn_depth, self.dropout,
                                  self.use_gold_predicates, self.use_gold_arguments, sum_loss=self.sum_loss)
            if self.loss_weight:
                self.loss_weights = nn.Parameter(torch.ones(2))  # loss weights for amr and srl
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.probe_generator.weight, std=0.02)
        nn.init.constant_(self.probe_generator.bias, 0.)
        if self.soft_mtl:
            nn.init.normal_(self.srl_probe_generator.weight, std=0.02)
            nn.init.constant_(self.srl_probe_generator.bias, 0.)

    def cut_input(self, word_repr, word_mask):
        if self.soft_mtl:
            probe = torch.tanh(self.srl_probe_generator(word_repr[:1]))
        else:
            probe = torch.tanh(self.probe_generator(word_repr[:1]))
        word_repr = word_repr[1:]
        word_mask = word_mask[1:]
        return word_repr, word_mask, probe

    def encode_input_layer(self, tok, lem, pos, ner, word_char):
        word_repr = self.embed_scale * self.word_encoder(word_char, tok, lem, pos, ner) + self.embed_positions(tok)
        word_repr = self.word_embed_layer_norm(word_repr)
        word_mask = torch.eq(lem, self.vocabs['lem'].padding_idx)
        return word_repr, word_mask

    def amr_sentence_encoder(self, word_repr, word_mask, edge, use_adj=False):
        if use_adj is True:
            adj, self_adj, undir_adj = generate_adj(
                edge, num_heads=self.num_heads, device=self.device)  # adj is [batch_size, max_word_num, max_word_num]
        else:
            undir_adj = None
        if self.soft_mtl:
            word_repr = self.srl_sent_encoder(word_repr, self_padding_mask=word_mask, adj_mask=undir_adj)
        else:
            word_repr = self.amr_snt_encoder(word_repr, self_padding_mask=word_mask, adj_mask=undir_adj)

        word_repr, word_mask, probe = self.cut_input(word_repr, word_mask)
        return word_repr, word_mask, probe

    def encode_bert_input(self, tok, lem, pos, ner, word_char, bert_token, token_subword_index):
        word_repr = self.word_encoder(word_char, tok, lem, pos, ner)
        bert_embed, _ = self.bert_encoder(bert_token, token_subword_index=token_subword_index)
        bert_embed = bert_embed.transpose(0, 1)
        word_repr = word_repr + self.bert_adaptor(bert_embed)
        word_repr = self.embed_scale * word_repr + self.embed_positions(tok)
        word_repr = self.word_embed_layer_norm(word_repr)
        word_mask = torch.eq(lem, self.vocabs['lem'].padding_idx)
        return word_repr, word_mask

    def encode_step(self, tok, lem, pos, ner, edge, word_char, use_adj=False):
        word_repr, word_mask = self.encode_input_layer(tok, lem, pos, ner, word_char)
        amr_word_repr, amr_word_mask, amr_probe = self.amr_sentence_encoder(word_repr, word_mask, edge, use_adj=use_adj)
        if self.soft_mtl:
            srl_word_repr, srl_word_mask, srl_probe = self.srl_sentence_encoder(word_repr, word_mask, edge, use_adj=use_adj)
            word_repr = torch.mean(torch.stack([amr_word_repr, srl_word_repr]), dim=0)  # average pooling for fusion
            probe = torch.mean(torch.stack([amr_probe, srl_probe]), dim=0)
        else:
            word_repr, word_mask, probe = amr_word_repr, amr_word_mask, amr_probe
        return word_repr, amr_word_mask, probe

    def encode_step_with_bert(self, tok, lem, pos, ner, edge, word_char, bert_token, token_subword_index,
                              use_adj=False):
        word_repr, word_mask = self.encode_bert_input(tok, lem, pos, ner, word_char, bert_token, token_subword_index)
        amr_word_repr, amr_word_mask, amr_probe = self.amr_sentence_encoder(word_repr, word_mask, edge, use_adj=use_adj)
        if self.soft_mtl:
            srl_word_repr, srl_word_mask, srl_probe = self.srl_sentence_encoder(word_repr, word_mask, edge, use_adj=use_adj)
            word_repr = torch.mean(torch.stack([amr_word_repr, srl_word_repr]), dim=0)  # average pooling for fusion
            probe = torch.mean(torch.stack([amr_probe, srl_probe]), dim=0)
        else:
            word_repr, word_mask, probe = amr_word_repr, amr_word_mask, amr_probe
        return word_repr, word_mask, probe

    def srl_sentence_encoder(self, word_repr, word_mask, edge, use_adj=False):
        if use_adj is True:
            adj, self_adj, undir_adj = generate_adj(
                edge, num_heads=self.num_heads, device=self.device)  # adj is [batch_size, max_word_num, max_word_num]
        else:
            undir_adj = None
        if self.soft_mtl:
            word_repr = self.srl_sent_encoder(word_repr, self_padding_mask=word_mask, adj_mask=undir_adj)
        else:
            word_repr = self.amr_snt_encoder(word_repr, self_padding_mask=word_mask, adj_mask=undir_adj)

        word_repr, word_mask, probe = self.cut_input(word_repr, word_mask)
        return word_repr, word_mask, probe

    def encode_step_for_srl_mtl(
            self, tok, lem, pos, ner, edge, word_char, use_adj=False):
        word_repr, word_mask = self.encode_input_layer(tok, lem, pos, ner, word_char)
        word_repr, word_mask, probe = self.srl_sentence_encoder(word_repr, word_mask, edge, use_adj=use_adj)
        return word_repr, word_mask, probe

    def encode_step_with_bert_for_srl_mtl(
            self, tok, lem, pos, ner, edge, word_char, bert_token, token_subword_index, use_adj=False):
        word_repr, word_mask = self.encode_bert_input(tok, lem, pos, ner, word_char, bert_token, token_subword_index)
        word_repr, word_mask, probe = self.srl_sentence_encoder(word_repr, word_mask, edge, use_adj=use_adj)
        return word_repr, word_mask, probe

    def work(self, data, beam_size, max_time_step, min_time_step=1, args=None):  # beam size == 8
        with torch.no_grad():
            if self.bert_encoder is not None:
                word_repr, word_mask, probe = self.encode_step_with_bert(
                    data['tok'], data['lem'], data['pos'], data['ner'], data['edge'], data['word_char'],
                    data['bert_token'], data['token_subword_index'], use_adj=args.encoder_graph
                )
            else:
                word_repr, word_mask, probe = self.encode_step(
                    data['tok'], data['lem'], data['pos'], data['ner'], data['edge'],
                    data['word_char'], use_adj=args.encoder_graph
                )

            mem_dict = {'snt_state': word_repr,
                        'snt_padding_mask': word_mask,
                        'probe': probe,
                        'local_idx2token': data['local_idx2token'],
                        'copy_seq': data['copy_seq']}
            init_state_dict = {}
            init_hyp = Hypothesis(init_state_dict, [DUM], 0.)
            bsz = word_repr.size(1)
            beams = [Beam(beam_size, min_time_step, max_time_step, [init_hyp]) for i in range(bsz)]  # init beams
            search_by_batch(self, beams, mem_dict, args)
        return beams

    def prepare_incremental_input(self, step_seq):
        conc = ListsToTensor(step_seq, self.vocabs['concept'])
        conc_char = ListsofStringToTensor(step_seq, self.vocabs['concept_char'])
        conc, conc_char = move_to_device(conc, self.device), move_to_device(conc_char, self.device)
        return conc, conc_char

    def prepare_incremental_graph_adj(self, adjs):
        """
        adjs: a list of adj matrix
        """
        batch_adj = torch.stack(adjs)
        return batch_adj

    def decode_step(self, inp, state_dict, mem_dict, offset, topk, args):
        step_concept, step_concept_char = inp
        word_repr = snt_state = mem_dict['snt_state']
        word_mask = snt_padding_mask = mem_dict['snt_padding_mask']
        probe = mem_dict['probe']
        copy_seq = mem_dict['copy_seq']
        local_vocabs = mem_dict['local_idx2token']
        _, bsz, _ = word_repr.size()

        new_state_dict = {}

        concept_repr = self.embed_scale * self.concept_encoder(step_concept_char, step_concept) + self.embed_positions(
            step_concept, offset)
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        adj = state_dict['adj']
        if args.decoder_graph is False:
            adj = None
        concept_reprs = []
        for idx, layer in enumerate(self.graph_encoder.layers):  # only for encoding concepts @kiro
            name_i = 'concept_repr_%d' % idx
            if name_i in state_dict:
                prev_concept_repr = state_dict[name_i]
                new_concept_repr = torch.cat([prev_concept_repr, concept_repr], 0)
            else:  # for start position DUM
                new_concept_repr = concept_repr

            new_state_dict[name_i] = new_concept_repr
            concept_repr, arc_weight, alignment_weight, attn_x_repr = layer(
                concept_repr, kv=new_concept_repr,
                external_memories=word_repr, external_padding_mask=word_mask,
                need_weights='max'
            )
            concept_reprs.append(concept_repr)
        name = 'graph_state'
        if name in state_dict:
            prev_graph_state = state_dict[name]
            new_graph_state = torch.cat([prev_graph_state, concept_repr], 0)
        else:
            new_graph_state = concept_repr
        new_state_dict[name] = new_graph_state

        name = 'graph_repr'  # to store the graph representation for relation identification
        if name in state_dict:
            prev_graph_state = state_dict[name]
            new_graph_state = torch.cat([prev_graph_state, concept_reprs[1]], 0)
        else:
            new_graph_state = concept_reprs[1]
        new_state_dict[name] = new_graph_state
        # Transformer decoder, [set_state, graph_state]
        conc_ll, arc_ll, rel_ll = self.decoder(
            concept_repr, alignment_weight,
            new_graph_state, arc_weight,
            copy_seq, work=True
        )
        # conc_ll, arc_ll, rel_ll = self.decoder(probe, snt_state, new_graph_state, snt_padding_mask, None, None,
        #                                        copy_seq, work=True)
        for i in range(offset):  # restore these variables from old state -> new state @kiro
            name = 'arc_ll%d' % i
            new_state_dict[name] = state_dict[name]
            name = 'rel_ll%d' % i
            new_state_dict[name] = state_dict[name]
        name = 'arc_ll%d' % offset
        new_state_dict[name] = arc_ll
        name = 'rel_ll%d' % offset
        new_state_dict[name] = rel_ll

        pred_arc_prob = torch.exp(arc_ll)
        arc_confidence = torch.log(torch.max(pred_arc_prob, 1 - pred_arc_prob))  # what is this? @kiro
        arc_confidence[:, :, 0] = 0.  # [1, bsz, steps], 0 means the arc to dummy @kiro TODO
        # pred_arc = torch.lt(pred_arc_prob, 0.5)
        # pred_arc[:,:,0] = 1
        # rel_confidence = rel_ll.masked_fill(pred_arc, 0.).sum(-1, keepdim=True)
        LL = conc_ll + arc_confidence.sum(-1, keepdim=True)  # + rel_confidence, joint concept and arc during decoding

        def idx2token(idx, local_vocab):
            if idx in local_vocab:
                return local_vocab[idx]
            return self.vocabs['predictable_concept'].idx2token(idx)

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1)  # bsz x k  # return values, indices @kiro

        results = []  # results only contains the concepts, no edges. @kiro
        for s, t, local_vocab in zip(topk_scores.tolist(), topk_token.tolist(), local_vocabs):
            res = []
            for score, token in zip(s, t):
                res.append((idx2token(token, local_vocab), score))
            results.append(res)

        return new_state_dict, results

    def srl_forward(self, data, encoder_graph=False):
        assert self.use_srl is True
        if self.bert_encoder is not None:
            word_repr, word_mask, probe = self.encode_step_with_bert_for_srl_mtl(
                data['tok'], data['lem'], data['pos'], data['ner'], data['edge'],
                data['word_char'], data['bert_token'],
                data['token_subword_index'], use_adj=encoder_graph
            )
        else:
            assert self.soft_mtl is True  # not implement for word embeddings without bert
            word_repr, word_mask, probe = self.encode_step(
                data['tok'], data['lem'], data['pos'], data['ner'], data['edge'],
                data['word_char'], use_adj=encoder_graph
            )
        srl_loss = self.srl.forward(word_repr, word_mask, data['gold_preds'][0], data['srl'])
        if self.loss_weight:
            normed_weights = F.softmax(self.loss_weights, dim=0)
            srl_loss = normed_weights[1] * srl_loss
        return srl_loss

    def forward(self, data, encoder_graph=False, decoder_graph=False):
        if self.bert_encoder is not None:
            word_repr, word_mask, probe = self.encode_step_with_bert(
                data['tok'], data['lem'], data['pos'], data['ner'], data['edge'],
                data['word_char'], data['bert_token'],
                data['token_subword_index'], use_adj=encoder_graph
            )
        else:
            word_repr, word_mask, probe = self.encode_step(
                data['tok'], data['lem'], data['pos'], data['ner'], data['edge'],
                data['word_char'], use_adj=encoder_graph
            )
        concept_repr = self.embed_scale * self.concept_encoder(data['concept_char_in'],
                                                               data['concept_in']) + self.embed_positions(
            data['concept_in'])
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        concept_repr = F.dropout(concept_repr, p=self.dropout, training=self.training)
        concept_mask = torch.eq(data['concept_in'], self.vocabs['concept'].padding_idx)
        attn_mask = self.self_attn_mask(data['concept_in'].size(0))
        # gold ans
        graph_target_rel = data['rel'][:-1]
        graph_target_arc = torch.ne(graph_target_rel, self.vocabs['rel'].token2idx(NIL))  # 0 or 1
        graph_arc_mask = torch.eq(graph_target_rel, self.vocabs['rel'].token2idx(PAD))
        graph_arc = graph_target_arc * (graph_arc_mask == 0)  # @kiro, the arc matrix
        graph_arc = generate_directional_self_loop_adj(graph_arc.transpose(0, 1), device=self.device)
        if decoder_graph is False:
            graph_arc = None
        # concept_repr = self.graph_encoder(concept_repr,
        #                          self_padding_mask=concept_mask, self_attn_mask=attn_mask,
        #                          external_memories=word_repr, external_padding_mask=word_mask)
        concept_reprs = []
        for idx, layer in enumerate(self.graph_encoder.layers):
            concept_repr, arc_weight, alignment_weight, attn_x_repr\
                = layer(concept_repr,
                        self_padding_mask=concept_mask, self_attn_mask=attn_mask,
                        adj_mask=graph_arc,
                        external_memories=word_repr, external_padding_mask=word_mask,
                        need_weights='max'
                        )
            concept_reprs.append(attn_x_repr)
        graph_arc_loss = F.binary_cross_entropy(arc_weight, graph_target_arc.float(), reduction='none')
        graph_arc_loss = graph_arc_loss.masked_fill_(graph_arc_mask, 0.).sum((0, 2))

        concept_loss, arc_loss, rel_loss, graph_state = \
            self.decoder(concept_repr, alignment_weight,
                         concept_reprs[1], arc_weight,
                         data['copy_seq'],
                         target=data['concept_out'], target_rel=data['rel'][1:]
                         )

        print(concept_mask.size())
        exit()
        concept_repr_loss = F.mse_loss(concept_repr[:, :-1, :], concept_reprs[1][:, 1:, :], reduction="sum")

        if self.sum_loss is False:
            concept_tot = concept_mask.size(0) - concept_mask.float().sum(0)
            concept_loss = concept_loss / concept_tot
            arc_loss = arc_loss / concept_tot
            rel_loss = rel_loss / concept_tot
            graph_arc_loss = graph_arc_loss / concept_tot
            concept_loss, arc_loss, rel_loss, graph_arc_loss = \
                concept_loss.mean(), arc_loss.mean(), rel_loss.mean(), graph_arc_loss.mean()
            concept_repr_loss = concept_repr_loss / concept_tot
        else:
            concept_loss, arc_loss, rel_loss, graph_arc_loss = \
                concept_loss.sum(), arc_loss.sum(), rel_loss.sum(), graph_arc_loss.sum()
        if self.loss_weight:
            normed_weights = F.softmax(self.loss_weights, dim=0)
            concept_loss, arc_loss, rel_loss, graph_arc_loss = \
                normed_weights[0] * concept_loss, normed_weights[0] * arc_loss, normed_weights[0] * rel_loss, \
                normed_weights[0] * graph_arc_loss
        return concept_loss, arc_loss, rel_loss, graph_arc_loss, concept_repr_loss


if __name__ == "__main__":
    def generate_adj(edges):  # add by kiro
        """
        edges: [batch_size, max_word_num]
        """
        edges = F.pad(edges, [1, 0], "constant", -1)  # dummy node $root
        edge_shape = edges.size()
        mask = ((edges > -1) == False).unsqueeze(-1)
        adj = torch.zeros([edge_shape[0], edge_shape[1], edge_shape[1]], dtype=torch.bool)  # init adj
        edges[edges == -1] = 0
        edges = edges.unsqueeze(-1).type(torch.LongTensor)
        adj.scatter_(2, edges, 1)
        adj.masked_fill_(mask, 0)
        # adj.transpose_(1, 2)
        # adj = adj.flip(1)  # flip according to dim 1
        # add diagonal
        dia = torch.ones(edge_shape, dtype=torch.int)
        dia = torch.diag_embed(dia)  # .flip(1)
        self_adj = adj | dia
        # un-directional adj
        undir_adj = adj.transpose(1, 2) | self_adj
        return adj, self_adj, undir_adj

    edges = [[2, 0, 4, 2, 4], [2, 3, 0, 3, -1]]
    edges = torch.IntTensor(edges)
    adj, self_adj, undir_adj = generate_adj(edges)
    print(adj)
    print(self_adj)
    print(undir_adj)
