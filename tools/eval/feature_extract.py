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
from core.dataset.baseline_dataset import MXFaceDataset

import struct
from collections import defaultdict
var_target = []

dataset_dict = {"lfw": ["/home/users/han.tang/data/public_face_data/glint/glint360k/lfw.bin"], 
        "ijbc": ["/home/users/han.tang/data/test/ijbc/ijbc_lmks_V135PNGAff.rec",
                 "/home/users/han.tang/data/test/ijbc/ijbc_lmks_V135PNGAff.idx"],
        "baseline": ["/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.rec",
                    "/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.idx"],
        "cluster-baseline": ["/running_package/torch-face/baseline_2030_V0.2/baseline_2030_V0.2.rec",
                    "/running_package/torch-face/baseline_2030_V0.2/baseline_2030_V0.2.idx"],
        "ValLife": ["/home/users/han.tang/data/test/val/ValLife/valLife_V0.2_indexed.rec",
                    "/home/users/han.tang/data/test/val/ValLife/valLife_V0.2_indexed.idx"],
        "ValID": ["/home/users/han.tang/data/test/val/ValID/wanren_V0.2_indexed.rec",
                    "/home/users/han.tang/data/test/val/ValID/wanren_V0.2_indexed.idx"],
        "Val30W_query": ["/home/users/han.tang/data/test/val/Val30W/qry30w_V1.0_lmks_V0.2_clean0916_indexed.rec",
                    "/home/users/han.tang/data/test/val/Val30W/qry30w_V1.0_lmks_V0.2_clean0916_indexed.idx"],
        "Val30W_gallery": ["/home/users/han.tang/data/test/val/Val30W/gly30w_V1.0_lmks_V0.2_indexed.rec",
                    "/home/users/han.tang/data/test/val/Val30W/gly30w_V1.0_lmks_V0.2_indexed.idx"],
        "njn": ["/cluster_home//data/train_data/id_card/njn/small_njn.rec",
                    "/cluster_home//data/train_data/id_card/njn/small_njn.idx"],
        "Val_J2_RealCar": ["/home/users/han.tang/data/test/val/Val_J2_RealCar/Val_J2_RealCar.rec",
                    "/home/users/han.tang/data/test/val/Val_J2_RealCar/Val_J2_RealCar.idx"],
        "abtdge_id1w": ["//home/users/han.tang/data/abtdge_id1w_Above9_miabove10_20200212/abtdge_id1w_Above9_miabove10_20200212.rec",
                    "/home/users/han.tang/data/abtdge_id1w_Above9_miabove10_20200212/abtdge_id1w_Above9_miabove10_20200212.idx"],
        }


class Eval(object):

    def __init__(self, local_rank, weight_path, fp16=True, emb_size=512, flip=False, with_index=False):
        self.backbone = None
        self.local_rank = local_rank
        self.fp16 = fp16
        self.flip = flip
        self.emb_size = emb_size
        self.device = "cuda:{}".format(local_rank)
        self.weight_path = weight_path
        self.with_index = with_index
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
            imgs = data[0]
            if self.flip:
                flip_img = data[1]
                label = data[2]
            else:
                label = data[1]

            if self.with_index:
                index = data[-1]
            else:
                index = -1

            out = self.backbone(imgs)
            if step % 100 == 0:
                print("process {}".format(step))
            embeddings_list.append(out)
            label_list.append(label)
            index_list.append(index)
        return embeddings_list, label_list, index_list


def build_dataset(bin_path, local_rank, batch_size, origin_prepro):
    dataset = MXBinFaceDataset(bin_path, local_rank)
    total_num = len(dataset)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = EvalDataLoader(
        local_rank=local_rank, dataset=dataset, batch_size=batch_size,
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
    return dataloader, total_num

def build_rec_dataset(rec_path, idx_path, local_rank, batch_size, origin_prepro):
    dataset = MXTestFaceDataset(rec_path=rec_path, idx_path=idx_path, local_rank=local_rank, origin_preprocess=origin_prepro)
    total_num = len(dataset)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = EvalDataLoader(
        local_rank=local_rank, dataset=dataset, batch_size=batch_size,
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
    return dataloader, total_num

def build_baseline_dataset(rec_path, idx_path, local_rank, batch_size, origin_prepro):
    dataset = MXFaceDataset(rec_path=rec_path, idx_path=idx_path, local_rank=local_rank, origin_preprocess=origin_prepro, training=False)

    total_num = len(dataset)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = EvalDataLoader(
        local_rank=local_rank, dataset=dataset, batch_size=batch_size,
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
    return dataloader, total_num

def write_label_index(label_list, local_rank, dst_path):
    label_dict = defaultdict(list)

    for label_idx, label in enumerate(label_list):
        label_dict[label].append(label_idx)
    local_dst_path = "{}_{}".format(dst_path, local_rank)

    with open(local_dst_path, "w") as fw:
        for k, v in label_dict.items():
            label_str = "{},".format(k)
            for idx in v:
                label_str += " {}".format(idx)
            fw.writelines(label_str + "\n")


def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)

def run(args, rank, world_size):
    dataset_name = args.dataset
    dataset_type = args.dataset_type
    batch_size = args.batch_size
    test = Eval(rank, weight_path=args.weight_path, emb_size=512, fp16=True)
    if dataset_type == "rec":
        dataloader, total_num = build_rec_dataset(dataset_dict[dataset_name][0],
                dataset_dict[dataset_name][1], rank, batch_size=batch_size, origin_prepro=args.origin_prepro)
    elif dataset_type == "baseline":
        dataloader, total_num = build_baseline_dataset(dataset_dict[dataset_name][0],
                dataset_dict[dataset_name][1], rank, batch_size=batch_size, origin_prepro=args.origin_prepro)
    elif dataset_type == "lfw":
        dataloader, total_num = build_dataset(dataset_dict[dataset_name][0], rank, batch_size=batch_size)
    else:
        raise AssertionError("not a good dataset name: {}".format(dataset_name))


    if total_num < 1:
        raise AssertionError("total num error {}".format(total_num))

    print("total process num: {}".format(total_num))
    remainder_num = total_num % world_size
    remainder_list = np.arange(remainder_num)
    save_num = total_num // world_size
    if rank in remainder_list:
        save_num += 1
    embeddings_list, label_list, index_list = test.eval(dataloader)
    label_list = torch.cat(label_list)
    label_list = label_list[:save_num]
    embeddings_list = torch.cat(embeddings_list)
    embeddings_list = embeddings_list[:save_num]
    label_list = label_list.cpu().numpy()

    output = args.output
    feature_path = "{}.bin".format(rank)
    feature_path = os.path.join(output, feature_path)
    label_path = "{}.txt".format(rank)
    label_path = os.path.join(output, label_path)

    with open(label_path, "w") as fw:
        #for label, index in zip(label_list, index_list):
        #    fw.writelines("{} {}\n".format(label, index))
        for label in label_list:
            fw.writelines("{}\n".format(label))

    label_index_path = "label_index.txt"
    label_index_path = os.path.join(output, label_index_path)
    write_label_index(label_list, rank, label_index_path)
    fw = open(feature_path, "wb")

    for embed in embeddings_list:
        embed_num = len(embed)
        #print(embed_num, embed.shape)
        embed = embed.cpu().numpy().astype(np.float32)
        embed = embed.reshape(-1)
        raw_data = struct.pack("f" * 512, *embed)
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
    parser.add_argument("--dataset_type", type=str, default="rec", help="")
    parser.add_argument("--gpu_num", type=int, default=2, help="")
    parser.add_argument("--batch_size", type=int, default=512, help="")
    parser.add_argument("--weight_path", type=str, default="/home/users/han.tang/workspace/pretrain_models/glint360k_cosface_r100_fp16_0.1/backbone.pth", help="")
    parser.add_argument("--output", type=str, default="/home/users/han.tang/data/eval/features/", help="")
    parser.add_argument("--origin_prepro", action="store_true", help="")
    args = parser.parse_args()
    main(args)

