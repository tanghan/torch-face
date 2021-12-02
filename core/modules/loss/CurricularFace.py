"""
@author: Jun Wang
@date: 20201126
@contact: jun21wangustc@gmail.com
"""

# based on
# https://github.com/HuangYG123/CurricularFace/blob/master/head/metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Parameter
import math

class CurricularFace(nn.Module):
    """Implementation for "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition".
    """
    def __init__(self, local_rank, world_size, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.local_rank = local_rank
        self.world_size = world_size
        self.register_buffer('t', torch.zeros(1, device="cuda:{}".format(self.local_rank)))

    def forward(self, cos_theta: torch.Tensor, labels):
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        valid_label_index = torch.where(labels != -1)[0]
        with torch.no_grad():

            total_target_logit = torch.zeros(cos_theta.size()[0], dtype=cos_theta.dtype, device=cos_theta.device)
            #print("valid label index shape: {}, total target logits shape: {}, valid label shape: {}".format(valid_label_index.shape, total_target_logit.shape, cos_theta[valid_label_index, labels[valid_label_index]].shape))
            total_target_logit.scatter_(0, valid_label_index, cos_theta[valid_label_index, labels[valid_label_index]])
            #print("rank: {}, valid index: {}, total_target_logit: {}".format(self.local_rank, valid_label_index, total_target_logit))
            dist.all_reduce(total_target_logit)
            #print("rank: {}, total_target_logit: {}".format(self.local_rank, total_target_logit))
            target_logit = total_target_logit.view(-1, 1)

            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
            mask = cos_theta > cos_theta_m
        valid_target_logit = cos_theta[valid_label_index, labels[valid_label_index]]
        valid_sin_theta = torch.sqrt(1.0 - torch.pow(valid_target_logit, 2))
        valid_cos_theta_m = valid_target_logit * self.cos_m - valid_sin_theta * self.sin_m #cos(target+margin)
        valid_final_target_logit = torch.where(valid_target_logit > self.threshold, valid_cos_theta_m, valid_target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t

        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta[valid_label_index] = cos_theta[valid_label_index].scatter_(1, labels[valid_label_index, None], valid_final_target_logit.view(-1, 1))
        output = cos_theta * self.s
        return output

