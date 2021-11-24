import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.mx_rec_utils.parse_rec_utils import unpack_fp64
from core.dataset.transforms.base_transform import NormalizeTransformer, ToTensor, MirrorTransformer
from collections import defaultdict


class MXIDCardDataset(Dataset):
    def __init__(self, rec_path, idx_path, local_rank, origin_preprocess=False, training=True):
        super(MXIDCardDataset, self).__init__()
        self.training = training

        base_transforms = [] 
        if origin_preprocess:
            base_transforms.append(transforms.ToPILImage())
            if training:
                base_transforms.append(transforms.RandomHorizontalFlip())

            base_transforms.append(transforms.ToTensor())
            base_transforms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        else:
            if training:
                base_transforms.append(MirrorTransformer())
            base_transforms.append(NormalizeTransformer(bias=128., scale=0.078125))
            base_transforms.append(ToTensor())

        self.transform = transforms.Compose(
                base_transforms
                         )

        self.local_rank = local_rank
        path_imgrec = rec_path
        path_imgidx = idx_path
        print("path imgrec: {}, path_imgidx: {}, is training: {}, use origin preprocess: {}".format(path_imgrec, path_imgidx, training, origin_preprocess))
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header0, _ = unpack_fp64(s)
        self.id_seq = list(range(int(header0.label[0]),
                                 int(header0.label[1])))

        self.id2range = {}
        self.imgidx = []

        self.img_pair = {}
        self.pair_identity = dict()
        self.id_offset = 0
        for identity in self.id_seq:
            s = self.imgrec.read_idx(identity)
            header, _ = unpack_fp64(s)
            id_start, id_end = int(header.label[0]), int(header.label[1])
            self.id2range[identity] = (id_start, id_end)
            sample_num = id_end - id_start
            #self.imgidx += list(range(*self.id2range[identity]))
            if sample_num == 2:
                #self.pair_identity[self.id_offset] = np.arange(id_start, id_end).astype(np.int32)
                self.pair_identity[self.id_offset] = range(id_start, id_end) 
                self.id_offset += 1
        print("init fin, id num: {}".format(self.id_offset))
        
    def __getitem__(self, index):
        idx_list = self.pair_identity[index]
        sample_list = []
        label_list = []
        for idx in idx_list:
            s = self.imgrec.read_idx(idx)
            header, img = unpack_fp64(s)
            label = header.label

            if not isinstance(label, numbers.Number):
                label = label[0]
            label = torch.tensor(label, dtype=torch.long)
            sample = mx.image.imdecode(img).asnumpy()
            if self.transform is not None:
                sample = self.transform(sample)
            sample_list.append(sample) 
            #label_list.append(label)
        samples = torch.cat(sample_list, dim=0)
        #labels = torch.cat(label_list)
        return samples, label

    def __len__(self):
        return len(self.pair_identity)


class SyntheticDataset(Dataset):
    def __init__(self, local_rank):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000
