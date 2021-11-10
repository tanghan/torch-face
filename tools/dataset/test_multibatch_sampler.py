import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from core.dataset.baseline_dataset import MXFaceDataset, DataLoaderX
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from core.dataset.sampler.multi_sampler import DistributedMultiBatchSampler


import torch.multiprocessing as mp
import numpy as np

def run(args, rank, size):
    print("rank: {}, size: {}".format(rank, size))
    rec_path = args.rec_path
    idx_path = args.idx_path
    #train_sampler = build_dataloader(rec_path, idx_path, rank, batch_size=1)
    data_loader = build_multibatch_sampler_test()

    for data in data_loader:
        print(data)
    #for batch_idx in multi_batch_sampler:
    #    print(batch_idx)


    '''
    total_test_num = 10
    for i, idx in enumerate(train_sampler):
        print("local_rank: {}, idx: {}".format(rank, idx))
        if i > total_test_num:
            break
    '''
    '''
    print(len(dataloader))
    sample_idx = 0
    total_sample_num = 10
    for data in dataloader:
        if sample_idx > total_sample_num:
            break
        print(data[0].shape)
        sample_idx += 1
    '''

def collate_fn(batch):
    #print(batch)
    return batch


def build_multibatch_sampler_test():
    idx1 = np.arange(0, 40)
    idx2 = np.arange(40, 50)
    dataset1 = TensorDataset(torch.tensor(idx1, dtype=torch.float32))
    dataset2 = TensorDataset(torch.tensor(idx2, dtype=torch.float32))

    dataset = ConcatDataset([dataset1, dataset2])
    print(len(dataset))
    train_sampler1 = torch.utils.data.distributed.DistributedSampler(dataset1, shuffle=False)
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(dataset2, shuffle=False)
    sampler_list = [train_sampler1, train_sampler2]
    total_sample_num_list = [len(dataset1), len(dataset2)]
    batch_list = [4, 2]
    multi_batch_sampler = DistributedMultiBatchSampler(samplers=sampler_list,
            batch_size_list=batch_list, total_sample_num_list=total_sample_num_list)
    dataloader = DataLoader(dataset, batch_sampler=multi_batch_sampler, num_workers=2,
            pin_memory=True, collate_fn=collate_fn)
    return dataloader


def build_dataloader(rec_path, idx_path, local_rank, batch_size):
    train_set = MXFaceDataset(rec_path=rec_path, idx_path=idx_path, local_rank=local_rank)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False)

    '''
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    '''
    return train_sampler


def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)


def main(args):
    size = 2
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
    parser.add_argument("--rec_path", type=str, default="/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.rec", help="")
    parser.add_argument("--idx_path", type=str, default="/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.idx", help="")
    #parser.add_argument("--local_rank", type=int, default=0, help="")
    #parser.add_argument("--world_size", type=int, default=1, help="")
    args= parser.parse_args()
    main(args)

