import os
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

import argparse
import logging

from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.dense = nn.Linear(3, 3)

    def forward(self, x):
        x = self.dense(x)
        return x

def build_dataset(rank, local_rank):
    a = torch.arange(100)

    dataset = TensorDataset(a)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
    return dataloader


def main_func(rank, local_rank, args):
    print(args.config)
    print("devices: ", local_rank)
    torch.manual_seed(1234)
    device = "cuda:{}".format(local_rank)
    dataloader = build_dataset(rank, local_rank)
    torch_rank = dist.get_rank()
    torch_world_size = dist.get_world_size()

    model = Model()
    model.cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    print("rank: {}, torch_rank: {}, torch_world_size: {}".format(rank, torch_rank, torch_world_size))
    for data in dataloader:
        img = data[0]
        print("rank: {}, value: {}".format(rank, img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["torch", "mpi"],
        default=None,
        help="job launcher for multi machines",
    )

    args = parser.parse_args()
    dist_url = args.dist_url
    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD
    local_rank = comm.Get_rank()
    world_size = comm.Get_size()
    print('MPI local_rank=%d, world_size=%d' % (local_rank, world_size))
    print("dist url:", dist_url)

    #os.environ["MASTER_ADDR"] = dist_url
    #os.environ["MASTER_PORT"] = "8000"

    num_devices = 3
    current_device = local_rank % num_devices

    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=local_rank,
        )
    except Exception as e:
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    torch.cuda.set_device(current_device)
    main_func(local_rank, current_device, args)
    
