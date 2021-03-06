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
from core.dataset.transforms.gray_transform import To3CGray
from core.dataset.transforms.spatial_variant_brightness_transform import SpatialVariantBrightness
from core.dataset.transforms.random_occlusion_transform import RandomOcclusion
from core.dataset.transforms.jpeg_compress_transform import JPEGCompress
from core.dataset.transforms.lmks_jitter_transform import LmksJitter
from core.dataset.transforms.random_downsample_transform import RandomDownSample
from core.dataset.transforms.motion_blur_transform import GaussianBlur, MotionBlur


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


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
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


class MXFaceDataset(Dataset):
    def __init__(self, rec_path, idx_path, local_rank, origin_preprocess=False, training=True, seed=1234):
        super(MXFaceDataset, self).__init__()
        self.training = training
        self.rng = np.random.RandomState(seed + local_rank)

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
                base_transforms.append(To3CGray(0.08, self.rng))
                base_transforms.append(SpatialVariantBrightness(p=0.3, rng=self.rng, brightness=0.5))
                base_transforms.append(RandomOcclusion(p=0.12, rng=self.rng, rand_occlusion_type=1))
                base_transforms.append(JPEGCompress(p=0.3, rng=self.rng, max_quality=90, min_quality=35))
                base_transforms.append(RandomDownSample(p=0.05, rng=self.rng, min_downsample_width=60, inter_method=1, data_shape=(3, 112, 112)))
                base_transforms.append(GaussianBlur(p=0.15, rng=self.rng, kernel_size_max=9, kernel_size_min=2, sigma_min=0, sigma_max=0))
                base_transforms.append(MotionBlur(p=0.05, rng=self.rng, length_min=9, length_max=18, angle_min=1, angle_max=359))
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
        self.id_num = {}
        self.imgidx = []
        for identity in self.id_seq:
            s = self.imgrec.read_idx(identity)
            header, _ = unpack_fp64(s)
            id_start, id_end = int(header.label[0]), int(header.label[1])
            self.id2range[identity] = (id_start, id_end)
            self.id_num[identity] = id_end - id_start
            self.imgidx += list(range(*self.id2range[identity]))
        
    def __getitem__(self, index):
        idx = self.imgidx[index]
        sample = 1
        s = self.imgrec.read_idx(idx)
        #header, img = mx.recordio.unpack(s)
        header, img = unpack_fp64(s)
        label = header.label

        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, int(label)

    def __len__(self):
        return len(self.imgidx)


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
