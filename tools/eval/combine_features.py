import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import argparse

def parse_feature(feature_dir, part_num):
    valid_part = np.arange(part_num)

    prefix_list = []
    feature_list = os.listdir(feature_dir)
    label_list = []
    fea_list = []
    num = 0
    for i in valid_part:
        fea_path = os.path.join(feature_dir, "{}.bin".format(i))
        label_path = os.path.join(feature_dir, "{}_label.txt".format(i))
        label_list.append(label_path)

        fea = np.fromfile(fea_path, dtype=np.float32)
        fea = fea.reshape(-1, 512)
        fea_list.append(fea)
        num += len(fea)

    print(num)
    split_num = num // part_num
    remainder = num % part_num
    print(split_num)
    print(remainder)
    print(fea_list[0].shape)
    features = np.concatenate(fea_list, -1)
    features = features.reshape(-1, 512)
    return features, label_list



def main(args):
    feature_dir = args.feature_dir
    output_dir = args.output_dir
    total_num = args.total_num
    part_num = args.part_num
    features, label_list = parse_feature(feature_dir, part_num)
    features = features[:total_num]
    dst_path = "features.npy"
    dst_path = os.path.join(output_dir, dst_path)
    np.save(dst_path, features)


    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--feature_dir", type=str, default="/home/users/han.tang/data/eval/features", help="")
    parser.add_argument("--part_num", type=int, default=3, help="")
    parser.add_argument("--output_dir", type=str, default="/home/users/han.tang/data/eval/features/", help="")
    parser.add_argument("--total_num", type=int, default=469375, help="")
    args = parser.parse_args()
    main(args)
