import torch
import torch.nn.functional as F
from parser.data import END, UNK
from parser.AMRGraph import is_attr_or_abs_form
from parser.utils import repreat_matrix

"""
 Beam search by batch
 need model has two functions:
    (1) decode_step
    (2) prepare_incremental_input
 when adapted to other use, modify those parts that are labeled by ##rewrite## accordingly. 
"""


###########
##rewrite##
###########
class Hypothesis(object):
    def __init__(self, state_dict, seq, score, previous_hypo_adj=None):
        '''
        state_dict: hidden states of the last step (has not yet consider seq[-1])
            for each item in state_dict, it must have shape of (seq_len x bsz x *) or (bsz x dim)
        seq: current generated sequence
        score: accumlated score so far (include seq[-1])
        '''
        self.state_dict = state_dict
        self.seq = seq
        self.score = score
        if previous_hypo_adj is None:  # only step 0 need to init adj @kiro
            self.init_adj()
        else:  # generate new adj according to previous hypo adj @kiro
            self.update_adj(previous_hypo_adj)

    def init_adj(self):
        self.graph_adj = torch.ones([1, 1], dtype=torch.bool).cuda()

    def update_adj(self, previous_hypo_adj):  # add by kiro
        offset = len(self.seq) - 2  # include the new added token @kiro
        name = 'arc_ll%d' % offset
        arc_ll = self.state_dict[name]
        pred_arc_prob = torch.exp(arc_ll)
        pred = torch.ge(pred_arc_prob, 0.5).squeeze(0)  # check the pred TODO @kiro
        # pred = pred_arc_prob.squeeze(0)
        self.graph_adj = F.pad(pred, [0, 1], 'constant', 0)
        self.graph_adj[-1, -1] = 1  # self-loop @kiro
        self.graph_adj[-1, 0] = 0  # no arc to dummy node @kiro

    def is_completed(self):
        ###########
        ##rewrite##
        ###########
        if self.seq[-1] == END:
            return True
        return False

    def __len__(self):
        return len(self.seq)


class Beam(object):
    """each beam for a test instance"""

    def __init__(self, beam_size, min_time_step, max_time_step, hypotheses):
        self.beam_size = beam_size
        self.min_time_step = min_time_step
        self.max_time_step = max_time_step
        self.completed_hypotheses = []
        self.steps = 0
        self.hypotheses = hypotheses  # hypotheses are the collection of *alive* hypotheses only

    def merge_score(self, prev_hyp, step):
        # step has two attributes: token and score
        ###########
        ##rewrite##
        ###########
        token, score = step
        prefix = prev_hyp.seq

        if len(prefix) == 1 and is_attr_or_abs_form(token):
            return float('-inf')
        if not token.endswith('_') and (':' in token or '/' in token or ',' in token):
            return float('-inf')
        if token == UNK:
            return float('-inf')
        new_score = prev_hyp.score + score
        return new_score

    def update(self, new_states, next_steps):
        '''
        each hypothesis in the beam have consumed its seq[-1], producing new states and #beam_size possible next steps
        new_states: for each item in new_states, it must have the shape of (seq_len x #num_hypotheses x *) or (#num_hypotheses x dim)
        next_steps: list (#num_hypotheses) of list (#beam_size) of (token, score)
        '''
        # collect the top (#beam_size-len(self.completed_hypotheses)) new candidates
        candidates = []  # list of triples (prev_hyp_idx, token, score)
        for prev_hyp_idx, steps in enumerate(next_steps):
            for step in steps:
                token = step[0]
                score = self.merge_score(self.hypotheses[prev_hyp_idx], step)
                candidates.append((prev_hyp_idx, token, score))

        candidates.sort(key=lambda x: x[-1], reverse=True)
        live_nyp_num = self.beam_size - len(self.completed_hypotheses)
        candidates = candidates[:live_nyp_num]

        # collect new states for selected top candidates  
        _split_state = dict()  # key => list of length live_nyp_num (number of selected top candidates)
        _prev_hyp_idx = torch.tensor([x[0] for x in candidates]).cuda()
        for k, v in new_states.items():
            split_dim = 1 if len(v.size()) >= 3 else 0
            _split_state[k] = v.index_select(split_dim, _prev_hyp_idx).split(1, dim=split_dim)

        # pack new hypotheses
        new_hyps = []
        for idx, (prev_hyp_idx, token, score) in enumerate(candidates):
            state = dict()
            for k, v in _split_state.items():
                state[k] = _split_state[k][idx]
            seq = self.hypotheses[prev_hyp_idx].seq + [token]
            new_hyps.append(Hypothesis(state, seq, score, previous_hypo_adj=self.hypotheses[prev_hyp_idx].graph_adj))
            # new_hyps[-1].update_adj(self.hypotheses[prev_hyp_idx].graph_adj)  # update the adj @kiro

        # send new hypotheses to self.completed_hypotheses or self.hypotheses accordingly
        self.hypotheses = []
        for hyp in new_hyps:
            if hyp.is_completed():
                if self.steps >= self.min_time_step:
                    self.completed_hypotheses.append(hyp)
            else:
                self.hypotheses.append(hyp)
        self.steps += 1
        # self.print_everything()

    def completed(self):
        if len(self.completed_hypotheses) < self.beam_size and self.steps < self.max_time_step:
            return False
        return True

    def get_k_best(self, k, alpha):
        if len(self.completed_hypotheses) == 0:
            self.completed_hypotheses = self.hypotheses
        self.completed_hypotheses.sort(key=lambda x: x.score / ((1 + len(x.seq)) ** alpha), reverse=True)
        return self.completed_hypotheses[:k]

    def print_everything(self):
        print('alive:')
        for x in self.hypotheses:
            print(x.seq)
        print('completed:')
        for x in self.completed_hypotheses:
            print(x.seq)


def search_by_batch(model, beams, mem_dict, args):
    '''
    beams, list of Beam, initial beams
    mem_dict, dict, those info. that will not change as decoding goes
        for each item in mem_dict, it must be a list of length len(beams) or a tensor with size(1) == len(beams)
    '''

    def ready_to_submit(hypotheses):
        '''
        prepare state_dict and next token,
        output them as one batch
        '''
        inp = model.prepare_incremental_input([hyp.seq[-1:] for hyp in hypotheses])  # input to Tensor, DUM for START
        adj = model.prepare_incremental_graph_adj([hyp.graph_adj for hyp in hypotheses])
        concat_hyps = dict()
        for hyp in hypotheses:
            for k, v in hyp.state_dict.items():
                concat_hyps[k] = concat_hyps.get(k, []) + [v]
        for k, v in concat_hyps.items():
            if len(v[0].size()) >= 3:
                concat_hyps[k] = torch.cat(v, 1)
            else:
                concat_hyps[k] = torch.cat(v, 0)
        # repreat adj
        concat_hyps["adj"] = repreat_matrix(adj, num_heads=model.num_heads)
        return concat_hyps, inp

    while True:
        # collect incomplete beams and put all hypotheses together
        hypotheses = []
        indices = []  #
        offset = -1  # the position of last token | @kiro right? offset += , offset = here?
        for idx, beam in enumerate(beams):  # a beam corresponding to a sample @kiro
            if not beam.completed():  # if the beam is not complete [EOG] @kiro
                for hyp in beam.hypotheses:
                    hypotheses.append(hyp)  # all the hypotheses in different beams into the same hypotheses? @kiro
                    indices.append(idx)
                    offset = len(hyp.seq) - 1  # offset is the last/0/1 len(hypothesis.seq) - 1? (same) @kiro
        if not hypotheses:  # if there is no hypotheses in the beams, break
            break

        state_dict, inp = ready_to_submit(hypotheses)  # combine separate hypotheses into a batch for computing @kiro

        # collect mem_dict
        cur_mem_dict = dict()
        indices = torch.tensor(indices).cuda()  # indices is the sentence id
        for k, v in mem_dict.items():  # for exclude some samples @kiro
            if isinstance(v, list):
                cur_mem_dict[k] = [v[i] for i in indices]
            else:
                cur_mem_dict[k] = v.index_select(1, indices)

        # run one decode step
        # state_dict: for each item in state_dict, it must have the shape of (seq_len x bsz x *) or (bsz x dim)
        # next_steps: list (bsz) of list (#beam_size) of (token, score)
        # print("step", offset)
        state_dict, results = model.decode_step(inp, state_dict, cur_mem_dict, offset, beams[0].beam_size, args)

        # dispatch the outcome to each beam
        _len_each_beam = [len(beam.hypotheses) for beam in beams if not beam.completed()]
        _state_dict_each_beam = [dict() for _ in _len_each_beam]

        # collect state dict (concept_rep, graph_state, arc_ll, rel_ll) for each beam @kiro
        for k, v in state_dict.items():
            split_dim = 1 if len(v.size()) >= 3 else 0
            for i, x in enumerate(v.split(_len_each_beam, dim=split_dim)):
                _state_dict_each_beam[i][k] = x

        _pos = 0
        _idx = 0
        for beam in beams:
            if not beam.completed():
                _len = len(beam.hypotheses)
                beam.update(_state_dict_each_beam[_idx], results[_pos:_pos + _len])
                _pos += _len
                _idx += 1
