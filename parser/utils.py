import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
import os, subprocess

POSTPROCESSING1_SCRIPT = "postprocess_1.0.sh"  # TODO for amr 1.0 @kiro
POSTPROCESSING2_SCRIPT = "postprocess_2.0.sh"  # TODO for amr 1.0 @kiro
EVAL_SCRIPT = "compute_smatch.sh"


def eval_smatch(dev_file, gold_dev_file):
    subprocess.Popen('bash {} {}'.format(POSTPROCESSING2_SCRIPT, dev_file), shell=True)
    postprocessing_file = dev_file + ".post"
    child = subprocess.Popen('bash {} {} {}'.format(EVAL_SCRIPT, postprocessing_file, gold_dev_file),
                             shell=True, stdout=subprocess.PIPE)
    eval_info = child.communicate()[0].decode()
    smatch = eval_info.split('\n')[0].strip().split()[-1]
    return smatch


def remove_files(filename):
    subprocess.Popen('rm {}'.format(filename), shell=True)
    print("Remove file {}".format(filename))


class checkpoint:
    def __init__(self, file_path):
        if os.path.exists(file_path):
            print(file_path, "exists! will be rewrite!")
        self.file = open(file_path, 'w')

    def write_checkpoint(self, info):
        self.file.write(info + "\n")

    def close(self):
        self.file.close()


def move_to_device(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor


def compute_f_by_tensor(input, target, mask):
    input = input.view(-1).tolist()
    target = target.view(-1).tolist()
    mask = mask.view(-1).tolist()
    tp, fp, tn, fn = 0., 0., 0., 0.
    for i, t, m in zip(input, target, mask):
        if m == 1:
            continue
        else:
            if i == 1:
                if t == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if t == 1:
                    fn += 1
                else:
                    tn += 1
    if tp == 0:
        return 0., 0., 0.

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    return P, R, F


def generate_self_adj(adj, device=None):  # add by kiro
    bsz, node_num = adj.size(0), adj.size(1)
    dia = torch.ones((bsz, node_num), dtype=torch.bool).to(device)
    dia = torch.diag_embed(dia)  # .flip(1)
    self_adj = adj | dia
    return self_adj


def repreat_matrix(adj, num_heads=8):
    # repeat
    bsz, len1, max_word_num = list(adj.size())  # [bsz, len1, max_word_num] len1 is 1 for decoding graph @kiro
    adj = adj.repeat_interleave(int(num_heads / 2), dim=0)
    adj = torch.stack((adj, torch.ones_like(adj)), dim=1).view(bsz * num_heads, len1, max_word_num)
    return adj


def generate_undirectional_adj(adj, num_heads=8, self_adj=None, device=None):
    if self_adj is None:
        self_adj = generate_self_adj(adj, device)
    undir_adj = adj.transpose(1, 2) | self_adj
    undir_adj = repreat_matrix(undir_adj, num_heads=num_heads)
    return undir_adj


def generate_adj(edges, num_heads=8, device=None):  # add by kiro
    """
    edges: [batch_size, max_word_num]
    """
    edges = F.pad(edges, [1, 0], "constant", -1)  # dummy node $root
    edge_shape = edges.size()
    mask = ((edges > -1) == False).unsqueeze(-1)
    adj = torch.zeros([edge_shape[0], edge_shape[1], edge_shape[1]], dtype=torch.bool).to(device)  # init adj
    edges[edges == -1] = 0
    edges = edges.unsqueeze(-1).type(torch.LongTensor).to(device)
    adj.scatter_(2, edges, 1)
    adj.masked_fill_(mask, 0)
    # adj.transpose_(1, 2)
    # adj = adj.flip(1)  # flip according to dim 1
    # add diagonal
    self_adj = generate_self_adj(adj, device)
    # un-directional adj
    undir_adj = generate_undirectional_adj(adj, num_heads=num_heads, self_adj=self_adj, device=device)
    return adj, self_adj, undir_adj


def gelu_fast(x):
    if not hasattr(gelu_fast, "_a"):
        gelu_fast._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_fast._a * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def label_smoothed_nll_loss(log_probs, target, eps):
    # log_probs: N x C
    # target: N
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    if eps == 0.:
        return nll_loss
    smooth_loss = -log_probs.sum(dim=-1)
    eps_i = eps / log_probs.size(-1)
    loss = (1. - eps) * nll_loss + eps_i * smooth_loss
    return loss
