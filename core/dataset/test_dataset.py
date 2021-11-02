import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pickle

from mxnet import nd

class MXTestFaceDataset(Dataset):
    def __init__(self, rec_path, idx_path, local_rank, image_size=(112, 112)):
        super(MXTestFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.local_rank = local_rank
        self.imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
        self.imgidx = np.array(list(self.imgrec.keys))

        self.image_size = image_size

    def __getitem__(self, index):
        s = self.imgrec.read_idx(self.imgidx[index])
        header, img = mx.recordio.unpack(s)
        img = mx.image.imdecode(img)
        label = header.label

        if img.shape[1] != self.image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        #img = nd.transpose(img, axes=(2, 0, 1))
        img = self.transform(img.asnumpy())
        index = header.index

        return img, int(label), index

    def __len__(self):
        return len(self.imgidx)


