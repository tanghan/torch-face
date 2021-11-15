"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""

import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class ArcFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, margin_arc=0.5, margin_am=0.0, scale=64.):
        super(ArcFace, self).__init__()
        self.margin_arc = margin_arc
        self.margin_am = margin_am
        self.scale = scale
        self.cos_margin = math.cos(margin_arc)
        self.sin_margin = math.sin(margin_arc)
        self.min_cos_theta = math.cos(math.pi - margin_arc)

    def forward(self, cos_theta: torch.Tensor, labels):
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        valid_label_index = torch.where(labels != -1)[0]
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels[valid_label_index, None], 1)
        index = index.to(torch.bool)
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output
