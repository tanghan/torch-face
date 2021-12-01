import torch
from torch import nn


def get_loss(local_rank, world_size, name):
    if name == "cosface":
        return CosFace(local_rank, world_size)
    elif name == "arcface":
        return ArcFace(local_rank, world_size)
    else:
        raise ValueError()


class CosFace(nn.Module):
    def __init__(self, local_rank, world_size, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.local_rank = local_rank
        self.world_size = world_size

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, local_rank, world_size, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.local_rank = local_rank
        self.world_size = world_size

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine
