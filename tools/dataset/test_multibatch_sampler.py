import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

#from core.dataset.baseline_dataset import MXFaceDataset, DataLoaderX
from core.dataset.dataset import MXFaceDataset, DataLoaderX
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from core.dataset.sampler.multi_sampler import DistributedMultiBatchSampler


import torch.multiprocessing as mp
import numpy as np

from easydict import EasyDict as edict 
from importlib.machinery import SourceFileLoader

def run(opt, rank, size):
    print("rank: {}, size: {}".format(rank, size))
    data_loader = get_dataloader(rank, opt, seed=1234)
    #data_loader = build_multibatch_sampler_test(rank)

    total_test_num = 10
    for i in range(2):
        for batch_idx, data in enumerate(data_loader):

            if batch_idx > total_test_num:
                break
            print(data[0].shape)
            print(data[1])
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
    return torch.tensor(batch)


def build_multibatch_sampler_test(local_rank):
    idx1 = np.arange(0, 42)
    idx2 = np.arange(42, 50)
    dataset1 = TensorDataset(torch.tensor(idx1, dtype=torch.float32))
    dataset2 = TensorDataset(torch.tensor(idx2, dtype=torch.float32))

    dataset = ConcatDataset([dataset1, dataset2])
    print(len(dataset))
    train_sampler1 = torch.utils.data.distributed.DistributedSampler(dataset1, shuffle=True, seed=1234)
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(dataset2, shuffle=True, seed=1234)
    sampler_list = [train_sampler1, train_sampler2]
    total_sample_num_list = [len(dataset1), len(dataset2)]
    batch_list = [4, 2]
    multi_batch_sampler = DistributedMultiBatchSampler(samplers=sampler_list,
            batch_size_list=batch_list, total_sample_num_list=total_sample_num_list)
    dataloader = DataLoaderX(local_rank=local_rank, dataset=dataset, batch_sampler=multi_batch_sampler, num_workers=2,
            pin_memory=True, collate_fn=collate_fn)
    return dataloader

def get_dataloader(local_rank, opt, seed):
    trainset = opt.dataset.trainset

    total_samples_list = []
    batch_size_list = []
    num_classes_list = []
    sampler_list = []
    dataset_list = []

    for dataname in trainset:
        rec_path = opt.dataset[dataname].rec_path
        idx_path = opt.dataset[dataname].idx_path
        num_samples = opt.dataset[dataname].num_samples
        num_classes = opt.dataset[dataname].num_classes
        batch_size = opt.dataset[dataname].batch_size
        train_set = MXFaceDataset(rec_path=rec_path, idx_path=idx_path, local_rank=local_rank, origin_preprocess=False)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, seed=seed)
        sampler_list.append(train_sampler)
        total_samples_list.append(num_samples)
        batch_size_list.append(batch_size)
        num_classes_list.append(num_classes)
        dataset_list.append(train_set)

    dataset=ConcatDataset(dataset_list)
    batch_sampler = DistributedMultiBatchSampler(samplers=sampler_list, 
            batch_size_list=batch_size_list, total_sample_num_list=total_samples_list)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=dataset,
        batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
    return train_loader



def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)


def main(args):
    config = args.config
    assert os.path.exists(config)
    opt = SourceFileLoader('module.name', './config/pretrain_config.py').load_module().opt

    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, opt, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="./config/pretrain_config.py", help="")
    parser.add_argument("--idx_path", type=str, default="/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.idx", help="")
    #parser.add_argument("--local_rank", type=int, default=0, help="")
    #parser.add_argument("--world_size", type=int, default=1, help="")
    args= parser.parse_args()
    main(args)

