"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""

import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

import numpy as np

class ArcFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, local_rank, world_size, feat_dim, num_class, margin_arc=0.5, margin_am=0.0, scale=64., name="local_arc"):
        super(ArcFace, self).__init__()
        self.local_rank = local_rank
        self.world_size = world_size
        self.device: torch.device = torch.device(self.local_rank)

        weights_list = []
        for i in range(world_size):
            gen = torch.Generator(device=self.device)
            gen = gen.manual_seed(1234 + i)
            weight = torch.normal(0, 0.01, (num_class // world_size, feat_dim), device=self.device, generator=gen)
            weights_list.append(weight)

        weights = torch.cat(weights_list, 0)
        torch.save(weights.cpu(), "{}_init_w.pt".format(name))
        self.weight = Parameter(weights)
        #self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_arc = margin_arc
        self.margin_am = margin_am
        self.scale = scale
        self.cos_margin = math.cos(margin_arc)
        self.sin_margin = math.sin(margin_arc)
        self.min_cos_theta = math.cos(math.pi - margin_arc)

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight)
        #feats = F.normalize(feats)
        cos_theta = F.linear(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        #print("cos theta: {}".format(cos_theta))
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output
