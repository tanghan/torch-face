import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np

import argparse

from core.models.resnet import iresnet
from core.dataset.eval_dataset import MXBinFaceDataset, EvalDataLoader
from core.dataset.test_dataset import MXTestFaceDataset
import struct
var_target = []

dataset_dict = {"lfw": ["/home/users/han.tang/data/public_face_data/glint/glint360k/lfw.bin"], 
        "ijbc": ["/home/users/han.tang/data/test/ijbc_lmks_V135PNGAff.rec",
                 "/home/users/han.tang/data/test/ijbc_lmks_V135PNGAff.idx"]}


class Eval(object):

    def __init__(self, local_rank, weight_path, fp16=True, emb_size=512, flip=False):
        self.backbone = None
        self.local_rank = local_rank
        self.fp16 = fp16
        self.emb_size = emb_size
        self.device = "cuda:{}".format(local_rank)
        self.weight_path = weight_path
        self.prepare()

    def network_init(self):
        self.backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.emb_size)
        self.backbone.to(self.device)

        self.backbone.load_state_dict(torch.load(self.weight_path, 
            map_location=torch.device(self.local_rank)))

        self.backbone = torch.nn.parallel.DistributedDataParallel(
            module=self.backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        logging.info("init network at {} finished".format(self.local_rank))
        self.backbone.eval()

    def prepare(self):
        self.network_init()

    @torch.no_grad()
    def eval(self, dataloader):
        embeddings_list = []
        label_list = []
        index_list = []
        for step, data in enumerate(dataloader):
            if flip:
                imgs, flip_img, label, index = data
            else:
                imgs, label, index = data
            out = self.backbone(imgs)
            if step % 100 == 0:
                print("process {}".format(step))
            embeddings_list.append(out)
            label_list.append(label)
            index_list.append(index)
        return embeddings_list, label_list, index_list


def build_dataset(bin_path, local_rank, batch_size):
    dataset = MXBinFaceDataset(bin_path, local_rank)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = EvalDataLoader(
        local_rank=local_rank, dataset=dataset, batch_size=batch_size,
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
    return dataloader

def build_rec_dataset(rec_path, idx_path, local_rank, batch_size):
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
    dataset_name = args.dataset
    bin_path = dataset_dict[dataset_name]
    test = Eval(rank, weight_path=args.weight_path, emb_size=512, fp16=True)
    if dataset_name == "ijbc":
        dataloader = build_rec_dataset(dataset_dict[dataset_name][0],
                dataset_dict[dataset_name][1], rank, batch_size=64)
    elif dataset_name == "lfw":
        dataloader = build_dataset(bin_path, rank, batch_size=64)
    else:
        raise AssertionError("not a good dataset name: {}".format(dataset_name))

    embeddings_list, label_list, index_list = test.eval(dataloader)

    output = args.output 
    feature_path = "{}.bin".format(rank)
    feature_path = os.path.join(output, feature_path)
    print(feature_path)

    fw = open(feature_path, "ab")

    for embed in embeddings_list:
        embed_num = len(embed)
        #print(embed_num, embed.shape)
        embed = embed.cpu().numpy().astype(np.float32)
        embed = embed.reshape(-1)
        raw_data = struct.pack("f" * 512 * embed_num, *embed)
        fw.write(raw_data)
    fw.close()

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
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--dataset", type=str, default="ijbc", help="")
    parser.add_argument("--gpu_num", type=int, default=2, help="")
    parser.add_argument("--weight_path", type=str, default="/home/users/han.tang/workspace/pretrain_models/glint360k_cosface_r100_fp16_0.1/backbone.pth", help="")
    parser.add_argument("--output", type=str, default="/home/users/han.tang/data/eval/features/", help="")
    args = parser.parse_args()
    main(args)

