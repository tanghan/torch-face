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
from core.dataset.transforms.base_transform import NormalizeTransformer, ToTensor, MirrorTransformer

class MXTestFaceDataset(Dataset):
    def __init__(self, rec_path, idx_path, local_rank, image_size=(112, 112), origin_preprocess=False):
        super(MXTestFaceDataset, self).__init__()

        if origin_preprocess:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 ])
        else:
            self.transform = transforms.Compose(
                [
                 NormalizeTransformer(bias=128., scale=0.078125),
                 ToTensor(),
                ])

        self.local_rank = local_rank
        self.imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")

        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        print("total img num: {}".format(len(self.imgidx)))

        self.image_size = image_size

    def __getitem__(self, index):
        s = self.imgrec.read_idx(self.imgidx[index])
        header, img = mx.recordio.unpack(s)
        img = mx.image.imdecode(img)
        label = header.label

        if isinstance(label, float) is not True:
            label = label[0]

        if img.shape[1] != self.image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        #img = nd.transpose(img, axes=(2, 0, 1))
        img = self.transform(img.asnumpy())

        return img, int(label), 0

    def __len__(self):
        return len(self.imgidx)


