import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from utils.parse_utils import parse_eval_bin
from core.dataset.eval_dataset import MXBinFaceDataset, EvalDataLoader
from core.dataset.test_dataset import MXTestFaceDataset

lfw_path = "/home/users/han.tang/data/public_face_data/glint/glint360k/lfw.bin"
ijbc_rec_path = "/home/users/han.tang/data/test/ijbc_lmks_V135PNGAff.rec"
ijbc_idx_path = "/home/users/han.tang/data/test/ijbc_lmks_V135PNGAff.idx"

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import argparse

def build_dataset(bin_path, local_rank, batch_size):
    dataset = MXBinFaceDataset(bin_path, local_rank)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = EvalDataLoader(
        local_rank=local_rank, dataset=dataset, batch_size=batch_size,
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
    return dataloader

def build_test_dataset(rec_path, idx_path, local_rank, batch_size):
    dataset = MXTestFaceDataset(rec_path, idx_path, local_rank)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = EvalDataLoader(
        local_rank=local_rank, dataset=dataset, batch_size=batch_size,
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
    return dataloader


def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)

def run(args, rank, world_size):
    #dataloader = build_dataset(lfw_path, rank, batch_size=64)
    dataloader = build_test_dataset(ijbc_rec_path, ijbc_idx_path, rank, batch_size=64)
    for data in dataloader:
        print(data[0].shape)

def main(args):
    gpu_num = args.gpu_num
    size = gpu_num
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, args, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--gpu_num", type=int, default=2, help="")
    #parser.add_argument("--local_rank", type=int, default=0, help="")
    #parser.add_argument("--world_size", type=int, default=1, help="")
    args= parser.parse_args()
    main(args)

    main(args)



