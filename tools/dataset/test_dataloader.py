import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from core.dataset.baseline_dataset import MXFaceDataset, DataLoaderX
import argparse
import torch
import torch.distributed as dist

import torch.multiprocessing as mp

def run(args, rank, size):
    print("rank: {}, size: {}".format(rank, size))
    data_prefix = args.data_prefix
    dataloader = build_dataloader(data_prefix, rank, batch_size=1)
    print(len(dataloader))
    sample_idx = 0
    total_sample_num = 10
    for data in dataloader:
        if sample_idx > total_sample_num:
            break
        print(data[0].shape)
        sample_idx += 1


def build_dataloader(data_prefix, local_rank, batch_size):
    train_set = MXFaceDataset(data_prefix=data_prefix, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    return train_loader


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
    parser.add_argument("--data_prefix", type=str, default="/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2", help="")
    #parser.add_argument("--local_rank", type=int, default=0, help="")
    #parser.add_argument("--world_size", type=int, default=1, help="")
    args= parser.parse_args()
    main(args)

