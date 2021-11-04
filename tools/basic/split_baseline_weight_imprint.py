import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import torch
from torch.nn.parameter import Parameter

weights_path = "/home/users/han.tang/workspace/weight_imprint"
def parse_mean_fea_path(weights_path, fea_prefix="mean_fea_"):
    temp_list = os.listdir(weights_path)
    imprint_list = []
    prefix_len = len(fea_prefix)
    imprint_dict = dict()
    for path in temp_list:
        fea_pos = path.find(fea_prefix)
        if fea_pos == 0:
            end_pos = path.find("_", prefix_len)
            idx = int(path[prefix_len: end_pos])
            imprint_dict[idx] = os.path.join(weights_path, path)
    
    for i in sorted(imprint_dict):
        imprint_list.append(imprint_dict[i])
    return imprint_list

def combine_imprint_list(imprint_list):
    print(imprint_list)
    feas = [] 
    for path_idx, path in enumerate(imprint_list):
        #with open(path, "rb") as fr:
        #    data_raw = fr.read()
        fea = np.fromfile(path, dtype=np.float32)
        fea = fea.reshape(-1, 512)
        feas.append(fea)
    feas = np.concatenate(feas)
    return feas


def save_rank_weights(feas, out_dir="", num_classes=500000, world_size=8, prefix="./"):

    for rank in range(world_size):
        num_local = num_classes // world_size + int(rank < num_classes % world_size)
        class_start = num_classes // world_size * rank + min(rank, num_classes % world_size)

        weight_name = os.path.join(prefix, "rank_{}_softmax_weight.pt".format(rank))
        weight_mom_name = os.path.join(prefix, "rank_{}_softmax_weight_mom.pt".format(rank))
        print(weight_name)
        print(num_local, class_start)
        weight = torch.tensor(feas[class_start: class_start+num_local])
        weight_mom = torch.zeros_like(weight)
        print(weight.size())
        torch.save(weight.data, weight_name)
        torch.save(weight_mom, weight_mom_name)

def main():
    imprint_list = parse_mean_fea_path(weights_path)
    feas = combine_imprint_list(imprint_list)
    save_rank_weights(feas, prefix="/home/users/han.tang/workspace/weight_imprint/imprint_weights")


if __name__ == "__main__":
    main()


