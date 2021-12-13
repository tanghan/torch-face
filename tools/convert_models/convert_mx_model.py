import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import torch
import mxnet as mx
from mxnet import nn as gluon_nn


class BaseModel(mx.gluon.HybridBlock):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.stage1 = gluon_nn.HybridSequential()
        self.stage1.add(


    
