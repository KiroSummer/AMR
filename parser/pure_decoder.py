import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

from parser.data import NIL, PAD
from parser.utils import compute_f_by_tensor
from parser.transformer import MultiheadAttention, Transformer, TiedTransformer
from parser.utils import label_smoothed_nll_loss


class ArcGenerator(nn.Module):
    def __init__(self, vocabs, embed_dim, ff_embed_dim, num_heads, dropout):
        super(ArcGenerator, self).__init__()
        self.vocabs = vocabs
        self.dropout = dropout

    def reset_parameters(self):
        self.arc_layer.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, arc_weight, target_rel=None, work=False, non_probabilistic=False):
        if work:
            if non_probabilistic:
                return arc_weight
            arc_ll = torch.log(arc_weight + 1e-12)
            return arc_ll
        target_arc = torch.ne(target_rel, self.vocabs['rel'].token2idx(NIL))  # 0 or 1
        arc_mask = torch.eq(target_rel, self.vocabs['rel'].token2idx(PAD))
        pred = torch.ge(arc_weight, 0.5)
        if not self.training:
            print('arc p %.3f r %.3f f %.3f' % compute_f_by_tensor(pred, target_arc, arc_mask))
        arc_loss = F.binary_cross_entropy(arc_weight, target_arc.float(), reduction='none')
        arc_loss = arc_loss.masked_fill_(arc_mask, 0.).sum((0, 2))
        return arc_loss


class ConceptGenerator(nn.Module):
    def __init__(self, vocabs, embed_dim, ff_embed_dim, conc_size, dropout):
        super(ConceptGenerator, self).__init__()
        self.transfer = nn.Linear(embed_dim, conc_size)
        self.generator = nn.Linear(conc_size, vocabs['predictable_concept'].size)
        self.diverter = nn.Linear(conc_size, 3)
        self.vocabs = vocabs
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.transfer.weight, std=0.02)
        nn.init.normal_(self.diverter.weight, std=0.02)
        nn.init.normal_(self.generator.weight, std=0.02)
        nn.init.constant_(self.diverter.bias, 0.)
        nn.init.constant_(self.transfer.bias, 0.)
        nn.init.constant_(self.generator.bias, 0.)

    def forward(self, outs, alignment_weight, copy_seq,
                target=None, work=False, non_probabilistic=False):
        seq_len, bsz, _ = outs.size()
        outs_concept = torch.tanh(self.transfer(outs))
        outs_concept = F.dropout(outs_concept, p=self.dropout, training=self.training)  # concept representation @kiro

        gen_gate, map_gate, copy_gate = F.softmax(self.diverter(outs_concept), -1).chunk(3, dim=-1)  # prob
        copy_gate = torch.cat([copy_gate, map_gate], -1)

        if non_probabilistic:  # if non probabilistic, then no softmax, for globally training @kiro
            probs = gen_gate * self.generator(outs_concept)
        else:
            probs = gen_gate * F.softmax(self.generator(outs_concept), -1)  # generate prob

        tot_ext = 1 + copy_seq.max().item()
        vocab_size = probs.size(-1)

        if tot_ext - vocab_size > 0:
            ext_probs = probs.new_zeros((1, 1, tot_ext - vocab_size)).expand(seq_len, bsz, -1)
            probs = torch.cat([probs, ext_probs], -1)
        # copy_seq: src_len x bsz x 2
        # copy_gate: tgt_len x bsz x 2
        # alignment_weight: tgt_len x bsz x src_len
        # index: tgt_len x bsz x (src_len x 2)
        index = copy_seq.transpose(0, 1).contiguous().view(1, bsz, -1).expand(seq_len, -1, -1)
        copy_probs = (copy_gate.unsqueeze(2) * alignment_weight.unsqueeze(-1)).view(seq_len, bsz, -1)
        # print("copy_probs", copy_probs)
        probs = probs.scatter_add_(-1, index, copy_probs)
        ll = torch.log(probs + 1e-12)

        if work:
            if non_probabilistic:
                return probs, outs  # actually, the probs is scores, without softmax @kiro
            return ll, outs

        if not self.training:
            _, pred = torch.max(ll, -1)
            total_concepts = torch.ne(target, self.vocabs['predictable_concept'].padding_idx)
            acc = torch.eq(pred, target).masked_select(total_concepts).float().sum().item()
            tot = total_concepts.sum().item()
            print('conc acc', acc / tot)

        concept_loss = -ll.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        concept_mask = torch.eq(target, self.vocabs['predictable_concept'].padding_idx)
        concept_loss = concept_loss.masked_fill_(concept_mask, 0.).sum(0)
        return concept_loss, outs


class RelationGenerator(nn.Module):

    def __init__(self, vocabs, embed_dim, rel_size, dropout):
        super(RelationGenerator, self).__init__()
        self.vocabs = vocabs
        self.transfer_head = nn.Linear(embed_dim, rel_size)
        self.transfer_dep = nn.Linear(embed_dim, rel_size)

        self.proj = nn.Linear(rel_size + 1, vocabs['rel'].size * (rel_size + 1))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.transfer_head.weight, std=0.02)
        nn.init.normal_(self.transfer_dep.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

        nn.init.constant_(self.proj.bias, 0.)
        nn.init.constant_(self.transfer_head.bias, 0.)
        nn.init.constant_(self.transfer_dep.bias, 0.)

    def forward(self, outs, graph_state, target_rel=None, work=False):

        def get_scores(dep, head):
            head = torch.tanh(self.transfer_head(head))
            dep = torch.tanh(self.transfer_dep(dep))

            head = F.dropout(head, p=self.dropout, training=self.training)
            dep = F.dropout(dep, p=self.dropout, training=self.training)

            dep_num, bsz, _ = dep.size()
            head_num = head.size(0)

            bias_dep = dep.new_ones((dep_num, bsz, 1))
            bias_head = head.new_ones((head_num, bsz, 1))

            # seq_len x bsz x dim
            dep = torch.cat([dep, bias_dep], 2)
            head = torch.cat([head, bias_head], 2)

            # bsz x dep_num x vocab_size x dim
            dep = self.proj(dep).view(dep_num, bsz, self.vocabs['rel'].size, -1).transpose(0, 1).contiguous()
            # bsz x dim x head_num
            head = head.permute(1, 2, 0)

            # bsz x dep_num x vocab_size x head_num
            scores = torch.bmm(dep.view(bsz, dep_num * self.vocabs['rel'].size, -1), head).view(bsz, dep_num,
                                                                                                self.vocabs['rel'].size,
                                                                                                head_num)
            return scores

        scores = get_scores(outs, graph_state).permute(1, 0, 3, 2).contiguous()

        dep_num, bsz, _ = outs.size()
        head_num = graph_state.size(0)
        log_probs = F.log_softmax(scores, dim=-1)
        _, rel = torch.max(log_probs, -1)
        if work:
            # dep_num x bsz x head x vocab
            return log_probs

        rel_mask = torch.eq(target_rel, self.vocabs['rel'].token2idx(NIL)) + torch.eq(target_rel,
                                                                                      self.vocabs['rel'].token2idx(PAD))
        rel_acc = (torch.eq(rel, target_rel).float().masked_fill_(rel_mask, 0.)).sum().item()
        rel_tot = rel_mask.numel() - rel_mask.float().sum().item()
        if not self.training:
            print('rel acc %.3f' % (rel_acc / rel_tot))
        rel_loss = label_smoothed_nll_loss(log_probs.view(-1, self.vocabs['rel'].size), target_rel.view(-1), 0.).view(
            dep_num, bsz, head_num)
        rel_loss = rel_loss.masked_fill_(rel_mask, 0.).sum((0, 2))
        return rel_loss


class MLPRelationGenerator(nn.Module):

    def __init__(self, vocabs, embed_dim, rel_size, dropout):
        super(MLPRelationGenerator, self).__init__()
        self.vocabs = vocabs
        self.transfer = nn.Linear(2 * embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, vocabs['rel'].size)

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.transfer.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

        nn.init.constant_(self.transfer.bias, 0.)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, outs, graph_state, target_rel=None, work=False):
        """
        outs: dep_num, bsz, dep_dim
        graph_state (heads): head_num, bsz, head_dim
        """
        def get_scores(dep, head):
            dep_num, bsz, dep_dim = dep.size()
            head_num, bsz, head_dim = head.size()

            dep = dep.unsqueeze(2)
            head = head.unsqueeze(0).transpose(1, 2)
            dep = dep.expand(-1, -1, head_num, -1)
            head = head.expand(dep_num, -1, -1, -1)
            pair_representations = torch.cat([dep, head], dim=3)  # dep_num, bsz, head_num, 2 * emb_dim
            pair_representations = F.relu(self.transfer.forward(pair_representations))

            pair_representations = F.dropout(pair_representations, p=self.dropout, training=self.training)
            scores = self.proj.forward(pair_representations)  # dep_num, bsz, head_num, rel_size
            nil_scores = torch.zeros([dep_num, bsz, head_num, 1]).cuda()
            scores = torch.cat([nil_scores, scores], dim=-1)  # add O to the scores.
            return scores

        # scores = get_scores(outs, graph_state).permute(1, 0, 3, 2).contiguous()
        scores = get_scores(outs, graph_state)
        # scores: dep_num, bsz, head_num, vocab_size

        dep_num, bsz, _ = outs.size()
        head_num = graph_state.size(0)
        log_probs = F.log_softmax(scores, dim=-1)
        _, rel = torch.max(log_probs, -1)
        if work:
            # dep_num x bsz x head x vocab
            return log_probs

        rel_mask = torch.eq(target_rel, self.vocabs['rel'].token2idx(PAD))
        rel_acc = (torch.eq(rel, target_rel).float().masked_fill_(rel_mask, 0.)).sum().item()
        rel_tot = rel_mask.numel() - rel_mask.float().sum().item()
        if not self.training:
            print('rel acc %.3f' % (rel_acc / rel_tot))
        rel_loss = label_smoothed_nll_loss(log_probs.view(-1, self.vocabs['rel'].size + 1), target_rel.view(-1), 0.).view(
            dep_num, bsz, head_num)
        rel_loss = rel_loss.masked_fill_(rel_mask, 0.).sum((0, 2))  # exclude the NIL ? @kiro
        return rel_loss


class DecodeLayer(nn.Module):

    def __init__(self, vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, conc_size, rel_size, dropout):
        super(DecodeLayer, self).__init__()
        self.arc_generator = ArcGenerator(vocabs, embed_dim, ff_embed_dim, num_heads, dropout)
        self.concept_generator = ConceptGenerator(vocabs, embed_dim, ff_embed_dim, conc_size, dropout)
        self.relation_generator = RelationGenerator(vocabs, embed_dim, rel_size, dropout)
        self.dropout = dropout
        self.vocabs = vocabs

    def reset_parameters(self):
        self.arc_generator.reset_parameters()
        self.concept_generator.reset_parameters()
        self.relation_generator.reset_parameters()

    def forward(self, new_concept_repr, node_to_word_alignment,
                graph_repr, arc_weight,
                copy_seq,
                target=None, target_rel=None,
                work=False, non_probabilistic=False):
        # probe: tgt_len x bsz x embed_dim
        # snt_state, graph_state: seq_len x bsz x embed_dim
        if work:
            concept_ll, outs = self.concept_generator(new_concept_repr, node_to_word_alignment, copy_seq, work=True)
            arc_ll = self.arc_generator(arc_weight, work=True)
            rel_ll = self.relation_generator(new_concept_repr, graph_repr, work=True)
            return concept_ll, arc_ll, rel_ll

        concept_loss, outs = self.concept_generator(new_concept_repr, node_to_word_alignment, copy_seq,
                                                    target=target,
                                                    work=False, non_probabilistic=non_probabilistic)
        arc_loss = self.arc_generator(arc_weight,
                                      target_rel=target_rel,
                                      work=False)
        rel_loss = self.relation_generator(new_concept_repr, graph_repr, target_rel=target_rel, work=False)
        return concept_loss, arc_loss, rel_loss, outs
