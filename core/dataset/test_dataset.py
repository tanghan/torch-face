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

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class MXTestFaceDataset(Dataset):
    def __init__(self, rec_path, local_rank, image_size=(112, 112)):
        super(MXTestFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.local_rank = local_rank
        self.imgrec = mx.recordio.MXRecordIO(rec_path, "r")

        self.imgidx = np.arange(len(issame_list) * 2)
        self.image_size = image_size
        
    def __getitem__(self, index):
        idx = self.imgidx[index]
        _bin = self.bins[idx]
        img = mx.image.imdecode(_bin)

        if img.shape[1] != self.image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        #img = nd.transpose(img, axes=(2, 0, 1))
        flip_img = mx.ndarray.flip(data=img, axis=2)
        img = self.transform(img.asnumpy())
        flip_img = self.transform(flip_img.asnumpy())

        return img, flip_img

    def __len__(self):
        return len(self.imgidx)

class EvalDataLoader(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(EvalDataLoader, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(EvalDataLoader, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

