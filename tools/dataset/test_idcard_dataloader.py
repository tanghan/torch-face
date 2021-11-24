import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from core.dataset.baseline_dataset import DataLoaderX
from core.dataset.idcard_dataset import MXIDCardDataset
import argparse
import torch
import torch.distributed as dist
import cv2
import numpy as np

import torch.multiprocessing as mp

def run(args, rank, size):
    print("rank: {}, size: {}".format(rank, size))
    rec_path = args.rec_path
    idx_path = args.idx_path
    dataloader = build_dataloader(rec_path, idx_path, rank, batch_size=32)
    print(len(dataloader))
    sample_idx = 0
    total_sample_num = 3

    save_dir = "/home/users/han.tang/workspace/temp_data"
    for data in dataloader:
        if sample_idx > total_sample_num:
            break
        imgs, labels = data
        img1s, img2s = torch.split(imgs, 3, 1)
        img1s = img1s.cpu().numpy()
        img2s = img2s.cpu().numpy()
        batch_size = len(img1s)
        labels = labels.cpu().numpy()


        for batch_idx, (img1, img2, label) in enumerate(zip(img1s, img2s, labels)):
            label1, label2 = label
            save_path1 = "{}_1_{}_{}.jpg".format(batch_idx + sample_idx * batch_size, label1, rank)
            save_path1 = os.path.join(save_dir, save_path1)
            save_path2 = "{}_2_{}_{}.jpg".format(batch_idx + sample_idx * batch_size, label2, rank)
            save_path2 = os.path.join(save_dir, save_path2)
            #print(label1, save_path1, label2, save_path2)

            im1 = np.transpose(img1, (1, 2, 0))
            im1 = (im1 / 0.078125 + 128.).astype(np.uint8)
            im2 = np.transpose(img2, (1, 2, 0))
            im2 = (im2 / 0.078125 + 128.).astype(np.uint8)

            cv2.imwrite(save_path1, im1)
            cv2.imwrite(save_path2, im2)
        sample_idx += 1



def build_dataloader(rec_path, idx_path, local_rank, batch_size):
    train_set = MXIDCardDataset(rec_path=rec_path, idx_path=idx_path, local_rank=local_rank)

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
    parser.add_argument("--rec_path", type=str, default="/home/users/han.tang/data/small_njn.rec", help="")
    parser.add_argument("--idx_path", type=str, default="/home/users/han.tang/data/small_njn.idx", help="")
    #parser.add_argument("--local_rank", type=int, default=0, help="")
    #parser.add_argument("--world_size", type=int, default=1, help="")
    args= parser.parse_args()
    main(args)

