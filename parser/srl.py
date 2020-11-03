import json
import codecs
import sys
import torch
from torch import nn
from collections import OrderedDict


class SRL_module(nn.Module):  # add by kiro
    def __init__(self):
        super(SRL_module, self).__init__()
        # predicate rep
        self.pred_reps = nn.Linear(2 * self.lstm_hidden_size, self.config.pred_size)
        self.pred_reps_drop = nn.Dropout(self.dropout)
        # predicate scores
        self.pred_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.config.pred_size, self.config.ffnn_size) if i == 0
             else nn.Linear(self.config.ffnn_size, self.config.ffnn_size) for i
             in range(self.config.ffnn_depth)])  # [,150]
        self.pred_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.pred_unary_score_projection = nn.Linear(self.config.ffnn_size, 2)

        # argument rep
        self.argu_reps_0 = nn.Linear(2 * self.lstm_hidden_size, self.config.argu_size, bias=False)
        self.argu_reps_drop_0 = nn.Dropout(self.dropout)
        self.span_emb_size = 2 * self.config.argu_size
        # span_rep
        self.argu_reps = nn.Linear(self.span_emb_size, self.config.argu_size_u)
        self.argu_reps_drop = nn.Dropout(self.dropout)
        # span scores
        self.arg_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.config.argu_size_u, self.config.ffnn_size) if i == 0
             else nn.Linear(self.config.ffnn_size, self.config.ffnn_size) for i
             in range(self.config.ffnn_depth)])  # [,150]
        self.arg_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.arg_unary_score_projection = nn.Linear(self.config.ffnn_size, 2)

        # srl scores
        self.srl_unary_score_input_size = self.config.argu_size_u + self.config.pred_size
        self.srl_unary_score_layers = nn.ModuleList(
            [nn.Linear(self.srl_unary_score_input_size, self.config.ffnn_size)
             if i == 0 else nn.Linear(self.config.ffnn_size, self.config.ffnn_size)
             for i in range(self.config.ffnn_depth)])
        self.srl_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.srl_unary_score_projection = nn.Linear(self.config.ffnn_size, self.label_space_size - 1)

    def forward(self):
        predict_dict = dict()
        """Get the candidate predicate"""
        pred_reps = self.pred_reps_drop(F.relu(self.pred_reps.forward(lstm_out)))
        candidate_pred_ids = torch.arange(0, max_sent_length).unsqueeze(0).expand(num_sentences, -1)
        candidate_pred_emb = pred_reps
        candidate_pred_scores = self.get_pred_unary_scores(candidate_pred_emb, self.config, self.dropout,
                                                           1, "pred_scores")
        # 1. get the predicted predicate index [batch_size, max_sentence_length]
        predicted_predicates_index = self.get_candidate_predicates_index(candidate_pred_scores, masks)
        # 2. eval the predicted predicates in P R F1
        match_num, predicted_pred_num, gold_pred_num = \
            self.eval_predicted_predicates(predicted_predicates_index,
                                           gold_predicates[2], gold_pred_num=gold_predicates[1])
        # 3. compute the predicate process loss
        pred_loss = self.get_pred_focal_loss(candidate_pred_scores, gold_predicates[2], mask=masks)
        # get the predicted predicates indexes
        predict_dict.update({"match_predicates": match_num,
                             "sys_predicate": predicted_pred_num,
                             "gold_predicate": gold_pred_num})
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
        argu_reps = self.argu_reps_drop_0(F.relu(self.argu_reps_0.forward(lstm_out)))
        # candidate_starts [num_sentences, max_sent_length * max_arg_width]
        candidate_starts, candidate_ends, candidate_mask = BiLSTMTaggerModel.get_candidate_spans(
            sent_lengths, max_sent_length, self.config.max_arg_width)
        flatted_candidate_mask = candidate_mask.view(-1)
        batch_word_offset = BiLSTMTaggerModel.exclusive_cumsum(sent_lengths)  # get the word offset in a batch
        # choose the flatted_candidate_starts with the actual existing positions, i.e. exclude the illegal starts
        flatted_candidate_starts = candidate_starts + batch_word_offset
        flatted_candidate_starts = flatted_candidate_starts.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        flatted_candidate_ends = candidate_ends + batch_word_offset
        flatted_candidate_ends = flatted_candidate_ends.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        # flatten the lstm output according to the sentence mask, i.e. exclude the illegal (padding) lstm output
        byte_sentence_mask = masks.type(torch.bool).cuda()  # convert cpu type to cuda type
        flatted_context_output = self.flatten_emb_in_sentence(argu_reps, byte_sentence_mask)  # cuda type
        batch_word_num = flatted_context_output.size()[0]  # total word num in batch sentences
        # generate the span embedding
        if torch.cuda.is_available():
            flatted_candidate_starts, flatted_candidate_ends = \
                flatted_candidate_starts.cuda(), flatted_candidate_ends.cuda()

        candidate_span_emb = self.get_span_emb(flatted_context_output,
                                               flatted_candidate_starts, flatted_candidate_ends,
                                               self.config, dropout=self.dropout)
        candidate_span_emb = self.argu_reps_drop(F.relu(self.argu_reps.forward(candidate_span_emb)))
        # Get the span ids
        candidate_span_number = candidate_span_emb.size()[0]
        max_candidate_spans_num_per_sentence = candidate_mask.size()[1]
        sparse_indices = (candidate_mask == 1).nonzero()
        sparse_values = torch.arange(0, candidate_span_number)
        candidate_span_ids = \
            torch.sparse.FloatTensor(sparse_indices.t(), sparse_values,
                                     torch.Size([num_sentences, max_candidate_spans_num_per_sentence])).to_dense()
        candidate_span_ids = candidate_span_ids.type(torch.LongTensor).cuda()  # ok @kiro
        predict_dict.update({"candidate_starts": candidate_starts, "candidate_ends": candidate_ends})
        # Get unary scores and topk of candidate argument spans.
        flatted_candidate_arg_scores = self.get_arg_unary_scores(candidate_span_emb, self.config, self.dropout,
                                                                 1, "argument scores")
        candidate_arg_scores = flatted_candidate_arg_scores.index_select(0, candidate_span_ids.view(-1)) \
            .view(candidate_span_ids.size()[0], candidate_span_ids.size()[1], -1)
        # spans_log_mask = torch.log(candidate_mask.type(torch.Tensor)).cuda()
        # candidate_arg_scores = candidate_arg_scores + spans_log_mask
        spans_mask = candidate_mask.type(torch.Tensor).unsqueeze(2).expand(-1, -1, 2).cuda()
        candidate_arg_scores = candidate_arg_scores * spans_mask
        # print(candidate_arg_scores)
        # print(spans_mask.size(), candidate_arg_scores.size())

        # 1. get predicted arguments
        predicted_arguments_index = \
            self.get_candidate_argument_index(candidate_arg_scores,
                                              candidate_mask.type(torch.cuda.LongTensor).view(num_sentences, -1))
        # 2. eval predicted argument
        dense_gold_argus_index = \
            self.get_gold_dense_argu_index(labels, max_sent_length,
                                           candidate_mask.type(torch.LongTensor).view(num_sentences, -1))
        matched_argu_num, sys_argu_num, gold_argu_num = \
            self.eval_predicted_arguments(predicted_arguments_index, dense_gold_argus_index)
        predict_dict.update({"matched_argu_num": matched_argu_num,
                             "sys_argu_num": sys_argu_num,
                             "gold_argu_num": gold_argu_num})
        # 3. compute argument loss
        argument_loss = self.get_argument_focal_loss(candidate_arg_scores, dense_gold_argus_index,
                                                     candidate_mask.type(torch.cuda.LongTensor).view(num_sentences, -1))
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
        srl_labels = self.get_srl_labels(arg_starts.cpu(), arg_ends.cpu(), predicates, labels, max_sent_length)
        srl_scores = self.get_srl_scores(arg_emb, pred_emb, self.label_space_size, self.config,
                                         self.dropout)
        srl_loss, srl_mask = self.get_srl_softmax_focal_loss(srl_scores, srl_labels, num_args, num_preds)
        # 4. eval the predicted srl
        matched_srl_num, sys_srl_num, gold_srl_num = self.eval_srl(srl_scores, srl_labels, srl_mask)
        predict_dict.update({"matched_srl_num": matched_srl_num,
                             "sys_srl_num": sys_srl_num,
                             "gold_srl_num": gold_srl_num})


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
        srl_tags = []
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
            srl_tags.append([tuple[-1] for tuple in srl_sen.srl])
        print("{} total sentences number {}".format(filepath, sentence_number))
    return srl_tags
