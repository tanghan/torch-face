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
    def __init__(self, local_rank, world_size, feat_dim, num_class, margin=0.25, gamma=256, name="local_arc"):
        super(CircleLoss, self).__init__()
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

        self.margin = margin
        self.gamma = gamma
        self.O_p = 1 + margin
        self.O_n = -margin
        self.delta_p = 1-margin
        self.delta_n = margin

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = F.linear(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        torch.save(cos_theta.cpu(), "circle_theta.pt")
        index_pos = torch.zeros_like(cos_theta)        
        index_pos.scatter_(1, labels.data.view(-1, 1), 1)
        index_pos = index_pos.byte().bool()
        index_neg = torch.ones_like(cos_theta)        
        index_neg.scatter_(1, labels.data.view(-1, 1), 0)
        index_neg = index_neg.byte().bool()

        alpha_p = torch.clamp_min(self.O_p - cos_theta.detach(), min=0.)
        alpha_n = torch.clamp_min(cos_theta.detach() - self.O_n, min=0.)

        logit_p = alpha_p * (cos_theta - self.delta_p)
        logit_n = alpha_n * (cos_theta - self.delta_n)

        output = cos_theta * 1.0
        output[index_pos] = logit_p[index_pos]
        output[index_neg] = logit_n[index_neg]
        output *= self.gamma
        return output
