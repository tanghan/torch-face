import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from core.dataset import dataset
import argparse
import torch
import torch.distributed as dist

def init_dist(rank, world_size):
    dist.init_process_group(backend='nccl',
		init_method="tcp://127.0.0.1:12345", rank=rank, world_size=world_size)

def main(args):
    data_dir = args.data_dir
    local_rank = args.local_rank
    world_size = args.world_size
    #init_dist(local_rank, world_size)
    print(torch.distributed.is_nccl_available())
    '''
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist.init_process_group('nccl')
    except KeyError:
        world_size = 1
        rank = 0
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)
    '''



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", type=str, default="/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.rec", help="")
    parser.add_argument("--local_rank", type=int, default=0, help="")
    parser.add_argument("--world_size", type=int, default=1, help="")
    args= parser.parse_args()
    main(args)

