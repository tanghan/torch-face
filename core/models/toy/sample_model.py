import torch
from torch import nn


class Model(nn.Module):

    def __init__(self, in_filter, out_filter):
        super(Model, self).__init__()
        self.dense = nn.Linear(in_filter, out_filter)

    def forward(self, x):
        return self.dense(x)
