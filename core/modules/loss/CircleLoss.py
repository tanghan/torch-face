"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class CircleLoss(Module):
    """Implementation for "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    Note: this is the classification based implementation of circle loss.
    """
    def __init__(self, local_rank, world_size, margin=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

        self.O_p = 1 + margin
        self.O_n = -margin
        self.delta_p = 1-margin
        self.delta_n = margin
        self.local_rank = local_rank
        self.world_size = world_size
        #self.iter = 0

    def forward(self, cos_theta: torch.Tensor, labels):
        cos_theta = cos_theta.clamp(-1, 1)
        valid_label_index = torch.where(labels != -1)[0]

        index_pos = torch.zeros_like(cos_theta)
        valid_index_pos = index_pos[valid_label_index]
        valid_index_pos.scatter_(1, labels[valid_label_index].view(-1, 1), 1)
        index_pos[valid_label_index] = valid_index_pos
        index_neg = 1 - index_pos
        
        index_pos = index_pos.to(torch.bool)
        index_neg = index_neg.to(torch.bool)

        alpha_p = torch.clamp_min(self.O_p - cos_theta.detach(), min=0.)
        alpha_n = torch.clamp_min(cos_theta.detach() - self.O_n, min=0.)

        logit_p = alpha_p * (cos_theta - self.delta_p)
        logit_n = alpha_n * (cos_theta - self.delta_n)

        output = cos_theta * 1.0
        output[index_pos] = logit_p[index_pos]
        output[index_neg] = logit_n[index_neg]
        output *= self.gamma

        #self.iter += 1
        return output
