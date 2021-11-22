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

    rng = np.random.RandomState(1234)
    dataloaer = build_dataset(rng, batch_size)

    for data in dataloaer:
        print(data)

def build_dataset(rng, batch_size):
    input_feat = rng.normal(size=(12, 4))
    input_feat = torch.tensor(input_feat)
    labels = rng.randint(low=0, high=10, size=(12, ))
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_feat, labels)
    sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader


def build_model(local_rank, world_size, num_classes, batch_size, margin_softmax, embedding_size=512, fp16=True):

    backbone = iresnet.iresnet100(dropout=0.0, fp16=fp16, num_features=embedding_size)
    backbone.to(self.device)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=self.backbone, broadcast_buffers=False, device_ids=[self.local_rank])
    backbone.train()

    fc = DistSoftmax(
        rank=local_rank, local_rank=local_rank, world_size=world_size, 
        batch_size=self.batch_size, margin_softmax=margin_softmax, num_classes=num_classes,
        embedding_size=self.emb_size)
    return backbone, fc

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
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--loss_type", type=str, default="arcface", help="")
    args = parser.parse_args()
    main(args)

