import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from core.dataset.transforms.base_transform import NormalizeTransformer, ToTensor, MirrorTransformer

class MXMultiFaceDataset(Dataset):
    def __init__(self, data_prefix, local_rank, origin_preprocess=False):
        super(MXFaceDataset, self).__init__()
        if origin_preprocess:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 ])
        else:
            self.transform = transforms.Compose(
                [MirrorTransformer(),
                 NormalizeTransformer(bias=128., scale=0.078125),
                 ToTensor(),
                ])
        self.local_rank = local_rank
        path_imgrec = "{}.rec".format(data_prefix)
        path_imgidx = "{}.idx".format(data_prefix)
        print("path imgrec: {}, path_imgidx: {}".format(path_imgrec, path_imgidx))
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

