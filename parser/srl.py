import json
import codecs
import sys
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict


class SRL_module(nn.Module):  # add by kiro
    def __init__(self, input_size, pred_size, argu_size, span_size, label_space_size,
                 ffnn_size, ffnn_depth, dropout, gold_predicates=False, gold_arguments=False, fl_alpha=1.0, fl_gamma=0.0):
        super(SRL_module, self).__init__()
        self.dropout = dropout
        self.input_size = input_size
        self.pred_size = pred_size
        self.argu_size = argu_size
        self.span_size = span_size
        self.ffnn_size = ffnn_size
        self.ffnn_depth = ffnn_depth
        self.label_space_size = label_space_size
        self.use_gold_predicates = gold_predicates
        self.use_gold_arguments = gold_arguments
        # self.pred_loss_function = nn.CrossEntropyLoss()
        self.focal_loss_alpha = fl_alpha  # 0.25
        self.focal_loss_gamma = fl_gamma  # 2
        # predicate rep
        self.pred_reps = nn.Linear(self.input_size, self.pred_size)
        self.pred_reps_drop = nn.Dropout(self.dropout)
        # predicate scores
        self.pred_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.pred_size, self.ffnn_size) if i == 0
             else nn.Linear(self.ffnn_size, self.ffnn_size) for i
             in range(self.ffnn_depth)])  # [,150]
        self.pred_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.ffnn_depth)])
        self.pred_unary_score_projection = nn.Linear(self.ffnn_size, 2)
        # argument rep
        self.argu_reps_0 = nn.Linear(self.input_size, self.argu_size, bias=False)
        self.argu_reps_drop_0 = nn.Dropout(self.dropout)
        self.span_emb_size = 2 * self.argu_size
        # span_rep
        self.argu_reps = nn.Linear(self.span_emb_size, self.span_size)
        self.argu_reps_drop = nn.Dropout(self.dropout)
        # span scores
        self.arg_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.span_size, self.ffnn_size) if i == 0
             else nn.Linear(self.ffnn_size, self.ffnn_size) for i
             in range(self.ffnn_depth)])  # [,150]
        self.arg_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.ffnn_depth)])
        self.arg_unary_score_projection = nn.Linear(self.ffnn_size, 2)
        # srl scores
        self.srl_unary_score_input_size = self.span_size + self.pred_size
        self.srl_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.srl_unary_score_input_size, self.ffnn_size)
             if i == 0 else nn.Linear(self.ffnn_size, self.ffnn_size)
             for i in range(self.ffnn_depth)])
        self.srl_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.ffnn_depth)])
        self.srl_unary_score_projection = nn.Linear(self.ffnn_size, self.label_space_size)

    def reset_parameters(self):
        # predicate
        nn.init.normal_(self.pred_reps.weight, std=0.02)
        nn.init.constant_(self.pred_reps.bias, 0.0)
        for layer in self.pred_unary_score_layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias, 0.0)
        nn.init.normal_(self.pred_unary_score_projection.weight, std=0.02)
        nn.init.constant_(self.pred_unary_score_projection.bias, 0.0)
        # argument
        nn.init.normal_(self.argu_reps_0.weight, std=0.02)
        nn.init.constant_(self.argu_reps_0.bias, 0.0)
        nn.init.normal_(self.argu_reps.weight, std=0.02)
        nn.init.constant_(self.argu_reps.bias, 0.0)
        for layer in self.arg_unary_score_layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias, 0.0)
        nn.init.normal_(self.arg_unary_score_projection.weight, std=0.02)
        nn.init.constant_(self.arg_unary_score_projection.bias, 0.0)
        # srl
        for layer in self.srl_unary_score_layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias, 0.0)
        nn.init.normal_(self.srl_unary_score_projection.weight, std=0.02)
        nn.init.constant_(self.srl_unary_score_projection.bias, 0.0)

    def get_pred_unary_scores(self, span_emb):
        input = span_emb
        for i, ffnn in enumerate(self.pred_unary_score_layers):
            input = F.relu(ffnn.forward(input))
            input = self.pred_dropout_layers[i].forward(input)
        output = self.pred_unary_score_projection.forward(input)
        return output

    def get_candidate_spans(self, sent_lengths, max_sent_length):
        num_sentences = len(sent_lengths)
        max_arg_width = max_sent_length  # max_arg_width = max_sent_length, since we don't need this constraint
        # Attention the order
        candidate_starts = torch.arange(0, max_sent_length).expand(num_sentences, max_arg_width, -1)
        candidate_width = torch.arange(0, max_arg_width).view(1, -1, 1)
        candidate_ends = candidate_starts + candidate_width

        candidate_starts = candidate_starts.contiguous().view(num_sentences, max_sent_length * max_arg_width)
        candidate_ends = candidate_ends.contiguous().view(num_sentences, max_sent_length * max_arg_width)
        actual_sent_lengths = sent_lengths.view(-1, 1).expand(-1, max_sent_length * max_arg_width)
        candidate_mask = candidate_ends < actual_sent_lengths.type(torch.LongTensor)
        float_candidate_mask = candidate_mask.type(torch.LongTensor)

        candidate_starts = candidate_starts * float_candidate_mask
        candidate_ends = candidate_ends * float_candidate_mask
        return candidate_starts.cuda(), candidate_ends.cuda(), candidate_mask.cuda()

    @staticmethod
    def exclusive_cumsum(input, exclusive=True):
        """
        :param input: input is the sentence lengths tensor.
        :param exclusive: exclude the last sentence length
        :return: the sum of y_i = x_1 + x_2 + ... + x_{i - 1} (i >= 1, and x_0 = 0)
        """
        assert exclusive is True
        if exclusive is True:
            exclusive_sent_lengths = torch.zeros(1).type(torch.cuda.LongTensor)
            result = torch.cumsum(torch.cat([exclusive_sent_lengths, input], 0)[:-1], 0).view(-1, 1)
        else:
            result = torch.cumsum(input, 0).view(-1, 1)
        return result

    def flatten_emb(self, emb):
        num_sentences, max_sentence_length = emb.size()[0], emb.size()[1]
        assert len(emb.size()) == 3
        flatted_emb = emb.contiguous().view(num_sentences * max_sentence_length, -1)
        return flatted_emb

    def flatten_emb_in_sentence(self, emb, batch_sentences_mask):
        num_sentences, max_sentence_length = emb.size()[0], emb.size()[1]
        flatted_emb = self.flatten_emb(emb)
        return flatted_emb[batch_sentences_mask.contiguous().view(num_sentences * max_sentence_length)]

    def get_span_emb(self, flatted_context_emb, flatted_candidate_starts, flatted_candidate_ends):
        batch_word_num = flatted_context_emb.size()[0]
        span_num = flatted_candidate_starts.size()[0]  # candidate span num.
        # gather slices from embeddings according to indices
        span_start_emb = flatted_context_emb[flatted_candidate_starts]
        span_end_emb = flatted_context_emb[flatted_candidate_ends]

        span_sum_reps = span_start_emb + span_end_emb
        span_minus_reps = span_start_emb - span_end_emb
        span_emb_feature_list = [span_sum_reps, span_minus_reps]
        span_emb = torch.cat(span_emb_feature_list, 1)
        return span_emb

    def get_arg_unary_scores(self, span_emb):
        """
        Compute span score with FFNN(span embedding)
        :param span_emb: tensor of [num_sentences, num_spans, emb_size]
        :param config:
        :param dropout:
        :param num_labels:
        :param name:
        :return:
        """
        input = span_emb
        for i, ffnn in enumerate(self.arg_unary_score_layers):
            input = F.relu(ffnn.forward(input))
            input = self.arg_dropout_layers[i].forward(input)
        output = self.arg_unary_score_projection.forward(input)
        return output

    def get_candidate_predicates_index(self, pred_scores, mask):
        y = F.softmax(pred_scores, -1)  # [batch_size, max_sentence_length, 2]
        max_y, max_indexes = y.max(dim=-1)  # max_indexes is the indices [batch_suze, max_sentence_length]
        max_indexes = max_indexes.type(torch.cuda.LongTensor) * mask.type(torch.cuda.LongTensor)  # since the padded position should be 0
        return max_indexes

    def get_pred_loss(self, pred_scores, gold_predicates, mask=None):
        assert len(pred_scores.size()) == 3  # [batch_size, max_sent_len, 2]
        pred_scores = pred_scores.view(-1, pred_scores.size(-1))  # [-1, 2], [total_words, 2]
        # print(pred_scores)
        y = F.log_softmax(pred_scores, -1)
        # print(y)
        y_hat = gold_predicates.view(-1, 1)
        # print(y_hat)
        loss_flat = -torch.gather(y, dim=-1, index=y_hat)
        # print(loss_flat)
        losses = loss_flat.view(*gold_predicates.size())
        tot_pred_num = mask.float().sum(1)
        tot_pred_num = tot_pred_num + torch.zeros_like(tot_pred_num).masked_fill_((tot_pred_num == 0), 1.0)
        losses = (losses * mask.float()).sum(1) / tot_pred_num
        loss = losses.mean()
        return loss

    def get_pred_focal_loss(self, pred_scores, gold_predicates, mask=None):
        assert len(pred_scores.size()) == 3  # [batch_size, max_sent_len, 2]
        pred_scores = pred_scores.view(-1, pred_scores.size(-1))  # [-1, 2], [total_words, 2]
        y = F.log_softmax(pred_scores, -1)
        y_hat = gold_predicates.view(-1, 1)
        loss_flat = torch.gather(y, dim=-1, index=y_hat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * loss_flat
        losses = loss.view(*gold_predicates.size())
        losses = losses * mask.float()
        loss = losses.mean()
        return loss

    def batch_index_select(self, emb, indices):
        num_sentences = emb.size()[0]
        max_sent_length = emb.size()[1]

        flatten_emb = self.flatten_emb(emb)
        offset = (torch.arange(0, num_sentences) * max_sent_length).unsqueeze(1).cuda()
        return torch.index_select(flatten_emb, 0, (indices + offset).view(-1))\
            .view(indices.size()[0], indices.size()[1], -1)

    def get_predicates_according_to_index(self, pred_reps, candidate_pred_scores, candidate_pred_ids, pred_dense_index):
        """
        get the pred_emb, pred_score, pred_ids, the pred_dense_index can be pred or gold
        :param pred_reps: [num_sentence, max_sentence_length, pred_rep_dim]
        :param candidate_pred_scores: [num_sentence, max_sentence_length, 2]
        :param candidate_pred_ids: [num_sentence, max_sentence_length]
        :param pred_dense_index: [num_sentence, max_sentence_length]
        :return:
        """
        num_sentence, max_sentence_length = pred_dense_index.size(0), pred_dense_index.size(1)
        pred_nums = pred_dense_index.sum(dim=-1)
        max_pred_num = max(pred_nums)
        sparse_pred_index = pred_dense_index.nonzero()
        if max_pred_num == 0:  # if there is no predicate in this batch
            padded_sparse_pred_index = torch.zeros([num_sentence, 1]).type(torch.cuda.LongTensor)
            pred_nums[0] = 1  # give an artificial predicate
        else:
            padded_sparse_pred_index = torch.zeros([num_sentence, max_pred_num]).type(torch.cuda.LongTensor)
            # sent_wise_sparse_pred_index = sparse_pred_index.chunk(2, dim=-1)[1].view(-1)
            sent_wise_sparse_pred_index = sparse_pred_index[:, -1]
            offset = 0
            for i, pred_num in enumerate(pred_nums):
                pred_num = int(pred_num)
                ith_pred_index = sent_wise_sparse_pred_index[offset: offset + pred_num]
                padded_sparse_pred_index[i, :pred_num] = ith_pred_index
                offset += pred_num
        padded_sparse_pred_index = padded_sparse_pred_index.cuda()
        # get the returns
        pred_emb = self.batch_index_select(pred_reps, padded_sparse_pred_index)
        pred_ids = torch.gather(candidate_pred_ids, 1, padded_sparse_pred_index)
        pred_scores = self.batch_index_select(candidate_pred_scores, padded_sparse_pred_index)
        return pred_ids, pred_emb, pred_scores, pred_nums

    def get_candidate_argument_index(self, argu_scores, mask):
        # argu_scores: [batch_size, max_argu_number, 2]
        y = F.softmax(argu_scores, -1)
        y_value, max_indexes = y.max(dim=-1)  #
        max_indexes = max_indexes.type(torch.cuda.LongTensor) * mask.type(torch.cuda.LongTensor)
        return max_indexes

    def get_gold_dense_argu_index(self, gold_labels, max_sentence_length, candidate_argu_mask):
        span_starts, span_ends, num_spans = gold_labels[1], gold_labels[2], gold_labels[4]
        num_sentences = span_starts.size(0)
        max_spans_num = span_starts.size(1)
        x_num_sentences = candidate_argu_mask.size(0)
        x_max_spans_num = candidate_argu_mask.size(1)

        gold_argu_num = int(num_spans.sum())

        total_span_num = num_sentences * max_spans_num
        flat_indices = (span_ends - span_starts) * max_sentence_length + span_starts

        sentence_indices_2d = torch.arange(0, num_sentences).unsqueeze(1).expand(-1, max_spans_num).cuda()
        sparse_argument_indices = torch.cat([sentence_indices_2d.unsqueeze(2), flat_indices.unsqueeze(2)], 2)

        sparse_argument_indices = torch.cat([sparse_argument_indices[i, :num_spans[i], :]
                                    for i in range(num_sentences)], dim=0)
        # print(sparse_argument_indices.size(), sparse_argument_indices)
        # print(span_starts, span_ends, sparse_argument_indices)
        # print(sparse_argument_indices.size(), total_span_num, x_max_spans_num)
        dense_gold_argus = torch.sparse.FloatTensor(sparse_argument_indices.cpu().view(gold_argu_num, -1).t(),
                                                    torch.LongTensor([1] * gold_argu_num).view(-1),
                                                    torch.Size([num_sentences, x_max_spans_num])).to_dense()
        # dense_gold_argus = dense_gold_argus * candidate_argu_mask  # TODO
        # print(num_spans, gold_argu_num, int(dense_gold_argus.sum()))
        assert gold_argu_num == int(dense_gold_argus.sum())
        # print(dense_gold_argus.nonzero(), candidate_argu_mask.nonzero())
        # exit()
        dense_gold_argus = (dense_gold_argus > 0).type(torch.cuda.LongTensor)
        # print(dense_gold_argus.size(), dense_gold_argus)
        # exit()
        return dense_gold_argus

    def get_argument_loss(self, argument_scores, gold_argument_index, candidate_argu_mask):
        """
        :param argument_scores: [num_sentence, max_candidate_span_num, 2]
        :param gold_argument_index: [num_sentence, max_candidate_span_num, 1]
        :param candidate_argu_mask: [num_sentence, max_candidate_span_num]
        :return:
        """
        y = F.log_softmax(argument_scores, dim=-1)
        y = y.view(-1, argument_scores.size(2))
        y_hat = gold_argument_index.view(-1, 1)
        loss_flat = -torch.gather(y, dim=-1, index=y_hat)
        losses = loss_flat.view(*gold_argument_index.size())
        tot_pred_num = candidate_argu_mask.float().sum(1)
        tot_pred_num = tot_pred_num + torch.zeros_like(tot_pred_num).masked_fill_((tot_pred_num == 0), 1.0)
        losses = (losses * candidate_argu_mask.float()).sum(1) / tot_pred_num
        loss = losses.mean()
        return loss

    def get_argument_focal_loss(self, argument_scores, gold_argument_index, candidate_argu_mask):
        """
        :param argument_scores: [num_sentence, max_candidate_span_num, 2]
        :param gold_argument_index: [num_sentence, max_candidate_span_num, 1]
        :param candidate_argu_mask: [num_sentence, max_candidate_span_num]
        :return:
        """
        y = F.log_softmax(argument_scores, dim=-1)
        y = y.view(-1, argument_scores.size(2))
        y_hat = gold_argument_index.view(-1, 1)
        loss_flat = torch.gather(y, dim=-1, index=y_hat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * loss_flat
        losses = loss.view(*gold_argument_index.size())
        losses = losses * candidate_argu_mask.float()
        loss = losses.mean()
        return loss

    def get_arguments_according_to_index(self, argument_reps, argument_scores,
                                         candidate_span_starts, candidate_span_ends, candidate_span_ids,
                                         argu_dense_index):
        """
        return predicted arguments
        :param argument_reps: flatted; [batch_total_span_num]
        :param argument_scores:
        :param candidate_span_starts:
        :param candidate_span_ends:
        :param candidate_span_ids:
        :param argu_dense_index:
        :return:
        """
        num_sentences, max_span_num = argu_dense_index.size(0), argu_dense_index.size(1)
        num_args = argu_dense_index.sum(-1)
        sparse_argu_index = argu_dense_index.nonzero()
        max_argu_num = int(max(num_args))
        if max_argu_num == 0:
            padded_sparse_argu_index = torch.zeros([num_sentences, 1]).type(torch.LongTensor)
            num_args[0] = 1
        else:
            padded_sparse_argu_index = torch.zeros([num_sentences, max_argu_num]).type(torch.LongTensor)
            sent_wise_sparse_argu_index = sparse_argu_index[:, -1]
            offset = 0
            for i, argu_num in enumerate(num_args):
                argu_num = int(argu_num)
                ith_argu_index = sent_wise_sparse_argu_index[offset: offset + argu_num]
                padded_sparse_argu_index[i, :argu_num] = ith_argu_index
                offset += argu_num
        padded_sparse_argu_index = padded_sparse_argu_index.cuda()

        argu_starts = torch.gather(candidate_span_starts, 1, padded_sparse_argu_index)
        argu_ends = torch.gather(candidate_span_ends, 1, padded_sparse_argu_index)
        argu_scores = self.batch_index_select(argument_scores, padded_sparse_argu_index)
        arg_span_indices = torch.gather(candidate_span_ids, 1, padded_sparse_argu_index)  # [num_sentences, max_num_args]
        arg_emb = argument_reps.index_select(0, arg_span_indices.view(-1)).view(
            arg_span_indices.size()[0], arg_span_indices.size()[1], -1
        )
        return arg_emb, argu_scores, argu_starts, argu_ends, num_args

    def sequence_mask(self, sent_lengths, max_sent_length=None):
        batch_size, max_length = sent_lengths.size()[0], torch.max(sent_lengths)
        if max_sent_length is not None:
            max_length = max_sent_length
        indices = torch.arange(0, max_length).unsqueeze(0).expand(batch_size, -1)
        mask = indices < sent_lengths.unsqueeze(1).cpu()
        mask = mask.type(torch.LongTensor)
        if torch.cuda.is_available():
            mask = mask.cuda()
        return mask

    def get_dense_span_labels(self, span_starts, span_ends, span_labels, num_spans, max_sentence_length, span_parents=None):
        num_sentences = span_starts.size()[0]
        max_spans_num = span_starts.size()[1]

        span_starts = span_starts + 1 - self.sequence_mask(num_spans)
        sentence_indices = torch.arange(0, num_sentences).unsqueeze(1).expand(-1, max_spans_num).type(torch.LongTensor).cuda()

        sparse_indices = torch.cat([sentence_indices.unsqueeze(2), span_starts.unsqueeze(2), span_ends.unsqueeze(2)],
                                   dim=2)
        if span_parents is not None:  # semantic span labels
            sparse_indices = torch.cat([sparse_indices, span_parents.unsqueeze(2)], 2)

        rank = 3 if span_parents is None else 4
        dense_labels = torch.sparse.FloatTensor(sparse_indices.cpu().view(num_sentences * max_spans_num, rank).t(),
                                                span_labels.view(-1).type(torch.FloatTensor),
                                                torch.Size([num_sentences] + [max_sentence_length] * (rank - 1)))\
            .to_dense()  # ok @kiro
        return dense_labels

    def gather_4d(self, params, indices):
        assert len(params.size()) == 4 and len(indices.size()) == 4
        params = params.type(torch.LongTensor)
        indices_a, indices_b, indices_c, indices_d = indices.chunk(4, dim=3)
        result = params[indices_a, indices_b, indices_c, indices_d]
        return result

    def get_srl_labels(self, arg_starts, arg_ends, predicates, labels, max_sentence_length):
        num_sentences = arg_starts.size()[0]
        max_arg_num = arg_starts.size()[1]
        max_pred_num = predicates.size()[1]

        sentence_indices_2d = torch.arange(0, num_sentences).unsqueeze(1).unsqueeze(2).expand(-1, max_arg_num, max_pred_num).cuda()
        expanded_arg_starts = arg_starts.unsqueeze(2).expand(-1, -1, max_pred_num)
        expanded_arg_ends = arg_ends.unsqueeze(2).expand(-1, -1, max_pred_num)
        expanded_predicates = predicates.unsqueeze(1).expand(-1, max_arg_num, -1)

        pred_indices = torch.cat([sentence_indices_2d.unsqueeze(3), expanded_arg_starts.unsqueeze(3),
                                  expanded_arg_ends.unsqueeze(3), expanded_predicates.unsqueeze(3)], 3)

        dense_srl_labels = self.get_dense_span_labels(labels[1], labels[2], labels[3], labels[4],
                                                      max_sentence_length, span_parents=labels[0])  # ans

        srl_labels = self.gather_4d(dense_srl_labels, pred_indices.type(torch.LongTensor))  # TODO !!!!!!!!!!!!!
        # print(pred_indices, dense_srl_labels, srl_labels)
        # exit()
        return srl_labels

    def get_srl_unary_scores(self, span_emb):
        input = span_emb
        for i, ffnn in enumerate(self.srl_unary_score_layers):
            input = F.relu(ffnn.forward(input))
            # if self.training:
            input = self.srl_dropout_layers[i].forward(input)
        output = self.srl_unary_score_projection.forward(input)
        return output

    def get_srl_scores(self, arg_emb, pred_emb, num_labels):
        num_sentences = arg_emb.size()[0]
        num_args = arg_emb.size()[1]  # [batch_size, max_arg_num, arg_emb_size]
        num_preds = pred_emb.size()[1]  # [batch_size, max_pred_num, pred_emb_size]

        unsqueezed_arg_emb = arg_emb.unsqueeze(2)
        unsqueezed_pred_emb = pred_emb.unsqueeze(1)
        expanded_arg_emb = unsqueezed_arg_emb.expand(-1, -1, num_preds, -1)
        expanded_pred_emb = unsqueezed_pred_emb.expand(-1, num_args, -1, -1)
        pair_emb_list = [expanded_arg_emb, expanded_pred_emb]
        pair_emb = torch.cat(pair_emb_list, 3)  # concatenate the argument emb and pre emb
        pair_emb_size = pair_emb.size()[3]
        flat_pair_emb = pair_emb.view(num_sentences * num_args * num_preds, pair_emb_size)
        flat_srl_scores = self.get_srl_unary_scores(flat_pair_emb)
        srl_scores = flat_srl_scores.view(num_sentences, num_args, num_preds, -1)
        dummy_scores = torch.zeros([num_sentences, num_args, num_preds, 1]).cuda()
        srl_scores = torch.cat([dummy_scores, srl_scores], 3)
        return srl_scores

    def get_srl_softmax_loss(self, srl_scores, srl_labels, num_predicted_args, num_predicted_preds):
        bsz = srl_scores.size(0)
        max_num_arg = srl_scores.size()[1]
        max_num_pred = srl_scores.size()[2]
        num_labels = srl_scores.size()[3]

        # num_predicted_args, 1D tensor; max_num_arg: a int variable means the gold ans's max arg number
        args_mask = self.sequence_mask(num_predicted_args, max_num_arg)
        pred_mask = self.sequence_mask(num_predicted_preds, max_num_pred)
        srl_mask = (args_mask.unsqueeze(2) == 1) & (pred_mask.unsqueeze(1) == 1)

        srl_scores = srl_scores.view(-1, num_labels)
        srl_labels = srl_labels.view(-1, 1).cuda()
        output = F.log_softmax(srl_scores, 1)

        negative_log_likelihood_flat = -torch.gather(output, dim=1, index=srl_labels).view(-1)
        negative_log_likelihood_flat = negative_log_likelihood_flat.view(bsz, max_num_arg * max_num_pred)
        srl_mask = srl_mask.view(bsz, max_num_arg * max_num_pred)
        srl_loss_mask = (srl_mask.view(-1) == 1).nonzero()
        if int(srl_labels.sum()) == 0 or int(sum(srl_loss_mask)) == 0:
            loss = negative_log_likelihood_flat.mean()
            return loss, srl_mask
        srl_mask = srl_mask.type(torch.cuda.FloatTensor)
        tot_pred_num = srl_mask.float().sum(1)
        tot_pred_num = tot_pred_num + torch.zeros_like(tot_pred_num).masked_fill_((tot_pred_num == 0), 1.0)
        negative_log_likelihood_flat = (negative_log_likelihood_flat.view(srl_mask.size()) * srl_mask).sum(1) / tot_pred_num
        loss = negative_log_likelihood_flat.mean()
        return loss, srl_mask

    def get_srl_softmax_focal_loss(self, srl_scores, srl_labels, num_predicted_args, num_predicted_preds):
        # print(srl_scores.size(), srl_labels.size(), num_predicted_args, num_predicted_preds)
        max_num_arg = srl_scores.size()[1]
        max_num_pred = srl_scores.size()[2]
        num_labels = srl_scores.size()[3]

        # num_predicted_args, 1D tensor; max_num_arg: a int variable means the gold ans's max arg number
        args_mask = self.sequence_mask(num_predicted_args, max_num_arg)
        pred_mask = self.sequence_mask(num_predicted_preds, max_num_pred)
        srl_mask = ((args_mask.unsqueeze(2) == 1) & (pred_mask.unsqueeze(1) == 1))

        srl_scores = srl_scores.view(-1, num_labels)
        srl_labels = (srl_labels.view(-1, 1)).cuda()
        output = F.log_softmax(srl_scores, 1)

        negative_log_likelihood_flat = torch.gather(output, dim=1, index=srl_labels).view(-1)

        pt = negative_log_likelihood_flat.exp()  # pt is the softmax score
        negative_log_likelihood_flat = \
            -1.0 * self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * negative_log_likelihood_flat

        srl_loss_mask = (srl_mask.view(-1) == 1).nonzero()
        if int(srl_labels.sum()) == 0 or int(sum(srl_loss_mask)) == 0:
            loss = negative_log_likelihood_flat.mean()
            return loss, srl_mask
        srl_mask = srl_mask.type(torch.cuda.FloatTensor)
        negative_log_likelihood_flat = negative_log_likelihood_flat.view(srl_mask.size()) * srl_mask
        loss = negative_log_likelihood_flat.mean()
        return loss, srl_mask

    def forward(self, input_emb, masks, gold_predicates, labels):
        input_emb = input_emb.transpose(0, 1)
        masks = masks.transpose(0, 1)
        masks = masks == False
        num_sentences, max_sent_length = input_emb.size(0), input_emb.size(1)
        sent_lengths = masks.sum(-1)
        """Get the candidate predicate"""
        pred_reps = self.pred_reps_drop(F.relu(self.pred_reps.forward(input_emb)))
        candidate_pred_ids = torch.arange(0, max_sent_length).unsqueeze(0).expand(num_sentences, -1).cuda()
        candidate_pred_emb = pred_reps
        candidate_pred_scores = self.get_pred_unary_scores(candidate_pred_emb)
        # 1. get the predicted predicate index [batch_size, max_sentence_length]
        predicted_predicates_index = self.get_candidate_predicates_index(candidate_pred_scores, masks)
        # 3. compute the predicate process loss
        pred_loss = self.get_pred_loss(candidate_pred_scores, gold_predicates[2], mask=masks)
        # 4. get the candidate predicates, scores, num_preds according to the index
        if self.use_gold_predicates:
            predicates, pred_emb, pred_scores, num_preds = \
                self.get_predicates_according_to_index(candidate_pred_emb, candidate_pred_scores, candidate_pred_ids,
                                                       gold_predicates[2])
            assert int((predicates == gold_predicates[0]).sum()) == predicates.size(0) * predicates.size()[1]

        else:
            predicates, pred_emb, pred_scores, num_preds = \
                self.get_predicates_according_to_index(candidate_pred_emb, candidate_pred_scores, candidate_pred_ids,
                                                       predicted_predicates_index)

        """Get the candidate arguments"""
        argu_reps = self.argu_reps_drop_0(F.relu(self.argu_reps_0.forward(input_emb)))
        # candidate_starts [num_sentences, max_sent_length * max_arg_width]
        candidate_starts, candidate_ends, candidate_mask = self.get_candidate_spans(
            sent_lengths, max_sent_length)
        flatted_candidate_mask = candidate_mask.view(-1)
        batch_word_offset = SRL_module.exclusive_cumsum(sent_lengths)  # get the word offset in a batch
        # choose the flatted_candidate_starts with the actual existing positions, i.e. exclude the illegal starts
        flatted_candidate_starts = candidate_starts + batch_word_offset
        flatted_candidate_starts = flatted_candidate_starts.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        flatted_candidate_ends = candidate_ends + batch_word_offset
        flatted_candidate_ends = flatted_candidate_ends.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        # flatten the lstm output according to the sentence mask, i.e. exclude the illegal (padding) lstm output
        byte_sentence_mask = masks.type(torch.bool).cuda()  # convert cpu type to cuda type
        flatted_context_output = self.flatten_emb_in_sentence(argu_reps, byte_sentence_mask)  # cuda type
        # generate the span embedding
        if torch.cuda.is_available():
            flatted_candidate_starts, flatted_candidate_ends = \
                flatted_candidate_starts.cuda(), flatted_candidate_ends.cuda()

        candidate_span_emb = self.get_span_emb(flatted_context_output,
                                               flatted_candidate_starts, flatted_candidate_ends)
        candidate_span_emb = self.argu_reps_drop(F.relu(self.argu_reps.forward(candidate_span_emb)))
        # Get the span ids
        candidate_span_number = candidate_span_emb.size()[0]
        max_candidate_spans_num_per_sentence = candidate_mask.size()[1]
        sparse_indices = (candidate_mask == 1).nonzero()
        sparse_values = torch.arange(0, candidate_span_number).cuda()
        candidate_span_ids = \
            torch.sparse.FloatTensor(sparse_indices.t(), sparse_values,
                                     torch.Size([num_sentences, max_candidate_spans_num_per_sentence])).to_dense()
        candidate_span_ids = candidate_span_ids.type(torch.LongTensor).cuda()  # ok @kiro
        # Get unary scores and topk of candidate argument spans.
        flatted_candidate_arg_scores = self.get_arg_unary_scores(candidate_span_emb)
        candidate_arg_scores = flatted_candidate_arg_scores.index_select(0, candidate_span_ids.view(-1)) \
            .view(candidate_span_ids.size()[0], candidate_span_ids.size()[1], -1)
        # spans_log_mask = torch.log(candidate_mask.type(torch.Tensor)).cuda()
        # candidate_arg_scores = candidate_arg_scores + spans_log_mask
        spans_mask = candidate_mask.unsqueeze(2).expand(-1, -1, 2)
        candidate_arg_scores = candidate_arg_scores * spans_mask.float()
        # print(candidate_arg_scores)
        # print(spans_mask.size(), candidate_arg_scores.size())

        # 1. get predicted arguments
        predicted_arguments_index = \
            self.get_candidate_argument_index(candidate_arg_scores, candidate_mask.view(num_sentences, -1))
        # 2. eval predicted argument
        dense_gold_argus_index = \
            self.get_gold_dense_argu_index(labels, max_sent_length,
                                           candidate_mask.view(num_sentences, -1))
        # 3. compute argument loss
        argument_loss = self.get_argument_loss(candidate_arg_scores, dense_gold_argus_index,
                                                     candidate_mask.view(num_sentences, -1))
        # 4. get the predicted argument representations according to the index
        if self.use_gold_arguments:
            arg_emb, arg_scores, arg_starts, arg_ends, num_args = \
                self.get_arguments_according_to_index(candidate_span_emb, candidate_arg_scores,
                                                      candidate_starts, candidate_ends, candidate_span_ids,
                                                      dense_gold_argus_index)
        else:
            arg_emb, arg_scores, arg_starts, arg_ends, num_args = \
                self.get_arguments_according_to_index(candidate_span_emb, candidate_arg_scores,
                                                      candidate_starts, candidate_ends, candidate_span_ids,
                                                      predicted_arguments_index)
        """Compute the candidate predicates and arguments semantic roles"""
        srl_labels = self.get_srl_labels(arg_starts, arg_ends, predicates, labels, max_sent_length)
        srl_scores = self.get_srl_scores(arg_emb, pred_emb, self.label_space_size)
        srl_loss, srl_mask = self.get_srl_softmax_loss(srl_scores, srl_labels, num_args, num_preds)
        return pred_loss + argument_loss + srl_loss


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


def read_srl_file(filepath):
    with open(filepath, 'r') as f:
        sentence_number = 0
        print("read srl file from {}".format(filepath))
        tokens, lemmas, pos, ner, edge, srl_tags, srl_tuples = [], [], [], [], [], [], []
        for line in f:  # read line
            sentence_number += 1
            if sentence_number % 1000 == 0:
                print("processed {} sentences".format(sentence_number))
            # stanford parser parse
            srl_sen = srl_example(json.loads(line))
            # for srl_tuple in srl_sen.srl:  # because of CLS added in the beginning of the sentence @kiro
            #     srl_tuple[0] += 1
            #     srl_tuple[1] += 1
            #     srl_tuple[2] += 1
            preprocessing_srl = [t for t in srl_sen.srl if t[-1] not in ['V', 'C-V']]
            if len(preprocessing_srl) > 0:
                tokens.append(srl_sen.tokens)
                lemmas.append(srl_sen.lemmas)
                pos.append(srl_sen.pos_tags)
                ner.append(srl_sen.ner_tags)
                edge.append(srl_sen.dependency_edges)
                srl_tags.append([tuple[-1] for tuple in srl_sen.srl])
                srl_tuples.append(preprocessing_srl)
        print("{} total sentences number {}".format(filepath, sentence_number))
    return tokens, lemmas, pos, ner, edge, srl_tags, srl_tuples
