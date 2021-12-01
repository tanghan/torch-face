import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse

import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from core.modules.tail.partial_fc import PartialFC
from core.modules.tail.dist_softmax import DistSoftmax 
from core.models.resnet import iresnet

from core.modules.loss.SST_Prototype import SST_Prototype
from core.models.toy.sample_model import Model

def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)


def run_test(args, rank, world_size):
    torch.cuda.set_device(rank)
    batch_size = args.batch_size
    loss_type = args.loss_type

    num_classes = 10
    num_samples = 3000
    in_filter = 256

    rng = np.random.RandomState(1234)
    dataloaer = build_dataset(rng, batch_size, num_samples, in_filter)
    model = build_model(in_filter, rank, world_size, num_classes=num_classes)
    margin_fn = build_loss(rank, world_size, batch_size)
    margin_fn.cuda(rank)

    i = 0
    while 1:
        for data in dataloaer:
            img1, img2, label = data
            out1 = model(img1.cuda(rank))
            out2 = model(img2.cuda(rank))
            loss = margin_fn(out1, out2, out2, out1, label)

        if i % 500 == 0:
            print(i)
        i += 1
        
        #print(loss)

def build_dataset(rng, batch_size, num_samples, in_filter):
    input_feat1 = rng.normal(size=(num_samples, in_filter))
    input_feat1 = torch.tensor(input_feat1).to(torch.float32)

    input_feat2 = rng.normal(size=(num_samples, in_filter))
    input_feat2 = torch.tensor(input_feat2).to(torch.float32)
    labels = rng.randint(low=0, high=10, size=(num_samples, ))
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_feat1, input_feat2, labels)
    sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    return dataloader


def build_model(in_filter, local_rank, world_size, num_classes, embedding_size=512):

    backbone = Model(in_filter, embedding_size)
    backbone.cuda(local_rank)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    return backbone

def build_loss(local_rank, world_size, batch_size):
    margin_fn = SST_Prototype(local_rank, world_size, batch_size)
    return margin_fn

def main(args):
    size = args.gpu_num
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, args, run_test))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test loss")
    parser.add_argument("--gpu_num", type=int, default=2, help="")
    parser.add_argument("--batch_size", type=int, default=2048, help="")
    parser.add_argument("--loss_type", type=str, default="arcface", help="")
    args = parser.parse_args()
    main(args)

