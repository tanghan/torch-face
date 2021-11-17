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
    def __init__(self, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.local_rank = dist.get_rank()
        self.register_buffer('t', torch.zeros(1, device="cuda:{}".format(self.local_rank)))

    def forward(self, cos_theta: torch.Tensor, labels):
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()

        valid_label_index = torch.where(labels != -1)[0]
        target_logit = cos_theta[valid_label_index][torch.arange(0, cos_theta.size(0)), labels[valid_label_index]].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t

        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, labels[valid_label_index].data.view(-1, 1), final_target_logit[valid_label_index])
        output = cos_theta * self.s
        return output

