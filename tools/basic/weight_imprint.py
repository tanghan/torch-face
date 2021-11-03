import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
from collections import defaultdict

import torch.multiprocessing as mp
import struct
import numpy as np

def parse_label_index(part_num, data_dir):
    label_dict = defaultdict(list)
    for i in range(part_num):
        label_path = "label_index.txt_{}".format(part_num)
        label_path = os.path.join(data_dir, label_path)
        with open(path, "r") as fr:
            while 1:
                line = fr.readline()
                if not line:
                    break
                line = line.strip()
                idx_list_pos = line.find(",")
                label_id = int(line[:idx_list_pos])
                idx_list = line[idx_list_pos+1:]
                splits = idx_list.split()
                for idx in splits:
                    label_dict[label_id].append((i, int(idx)))

    return label_dict

def init_process(rank, world_size, args, label_dict, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2134"
    dist.init_process_group(backend='gloo',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size, label_dict)

def run(args, rank, world_size, label_dict):
    total_label_num = len(label_dict)
    split_num = total_label_num // rank
    remainder = total_label_num - world_size * split_num
    if rank == world_size - 1:
        process_num = split_num
    else:
        process_num = split_num + remainder
    print("total num: {}, rank {}, process {}".format(total_label_num, rank, process_num))
    start = split_num * rank

    fea_file_list = []
    part_num = args.part_num
    data_dir = args.data_dir
    emb_size = args.emb_size
    for part_idx in range(part_num):
        fea_path = "{}.bin".format(part_idx)
        fr = open(fea_path, "rb")
        fea_file_list.append(fr)

    all_fea = []
    for label_id in range(start, start + process_num):
        fea_pos_list = label_dict[label_id]

        feas = []
        for pos in fea_pos_list:
            part_idx, offset = pos
            fr = fea_file_list[part_idx]
            fr.seek(emb_size * offset * 4)
            raw_data = fr.read(emb_size * 4)
            fea = struct.unpack("f" * emb_size, raw_data)
            feas.append(fea)
        mean_fea = np.mean(np.array(feas).reshape(-1, emb_size), 0)
        all_fea.append(mean_fea)
    dst_feature_path = "mean_fea_{}_{}.bin".format(start, process_num)
    dst_feature_path = os.path.join(data_dir, dst_feature_path)
    with open(dst_feature_path, "wb") as fw:
        for temp_fea = all_fea:
            fea_bytes = struct.pack("f" * emb_size, temp_fea.reshape(-1))
            fw.write(fea_bytes)

    for part_idx in range(part_num):
        fea_file_list[part_idx].close()
    
 
def main(args):
    part_num = args.part_num
    data_dir = args.data_dir
    label_dict = parse_label_index(part_num, data_dir)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, args, label_dict, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--part_num", type=int, default=4, help="")
    parser.add_argument("--data_dir", type=str, default="/home/users/han.tang/workspace/weight_imprint", help="")
    parser.add_argument("--num_threads", type=int, default=10, help="")
    parser.add_argument("--emb_size", type=int, default=512, help="")
    args = parser.parse_args()
    main(args)
