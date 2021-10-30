import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from core.dataset import dataset
import argparse
import torch
import torch.distributed as dist

import torch.multiprocessing as mp

def run(rank, size):
    print(rank, size)

def init_process(rank, world_size, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(rank, world_size)


def main(args):
    size = 2
    data_dir = args.data_dir
    '''
    local_rank = args.local_rank
    world_size = args.world_size
    init_dist(local_rank, world_size)
    print(torch.distributed.is_nccl_available())
    '''
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", type=str, default="/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.rec", help="")
    #parser.add_argument("--local_rank", type=int, default=0, help="")
    #parser.add_argument("--world_size", type=int, default=1, help="")
    args= parser.parse_args()
    main(args)

