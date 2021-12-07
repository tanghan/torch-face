"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""

import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class MagFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, local_rank, world_size, feat_dim, num_class, margin_am=0.0, scale=32, l_a=10, u_a=110, l_margin=0.45, u_margin=0.8, lamda=20, name="local_arc"):
        super(MagFace, self).__init__()
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

        self.margin_am = margin_am
        self.scale = scale        
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.lamda = lamda

    def calc_margin(self, x):
        margin = (self.u_margin-self.l_margin) / \
                 (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin
    
    def forward(self, feats, labels):
        x_norm = torch.norm(feats, dim=1, keepdim=True).clamp(self.l_a, self.u_a)# l2 norm
        ada_margin = self.calc_margin(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        loss_g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        kernel_norm = F.normalize(self.weight)
        feats = F.normalize(feats)
        cos_theta = F.linear(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        min_cos_theta = torch.cos(math.pi - ada_margin)        
        cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output, self.lamda*loss_g
