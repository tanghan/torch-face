import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import argparse

def parse_feature(feature_dir, part_num, samples_num, save_labels=False):
    valid_part = np.arange(part_num)

    prefix_list = []
    feature_list = os.listdir(feature_dir)
    label_list = []
    fea_list = []
    num = samples_num
    split_num = num // part_num
    remainder = num % part_num

    print(num)
    print(split_num)
    print("remainder: {}".format(remainder))
    remainder_fea = []
    remainder_label = []
    padding_fea = np.zeros(shape=(1, 512), dtype=np.float32)
    if remainder > 0:
        padding_list = [i for i in range(remainder, part_num)]
    else:
        padding_list = []

    for i in valid_part:
        fea_path = os.path.join(feature_dir, "{}.bin".format(i))
        if save_labels:
            label_path = os.path.join(feature_dir, "{}.txt".format(i))
            labels = []

            if os.path.exists(label_path):
                with open(label_path, "r") as fr:
                    while 1:
                        line = fr.readline()
                        if not line:
                            break
                        label_id = int(line.strip().split()[0])
                        labels.append(label_id)
                if i in padding_list:
                    labels.append(-1)
            label_list.append(np.expand_dims(np.array(labels), -1))
            print("label shape: {}".format(label_list[i].shape))

        fea = np.fromfile(fea_path, dtype=np.float32)
        fea = fea.reshape(-1, 512)

        if i in padding_list:
            print("do padding: {}".format(i))
            fea = np.concatenate([fea, padding_fea])
                            
        fea_list.append(fea)

    features = np.concatenate(fea_list, -1)
    features = features.reshape(-1, 512)
    print(features.shape)
    label = None
    index = None

    if save_labels:
        label = np.concatenate(label_list, -1)
        print("label shape:", label.shape)
        label = label.reshape(-1)
        index = np.zeros_like(label)
    return features, label, index



def main(args):
    feature_dir = args.feature_dir
    output_dir = args.output_dir
    total_num = args.total_num
    part_num = args.part_num
    save_labels = args.save_labels
    features, label, index  = parse_feature(feature_dir, part_num, total_num, save_labels)
    features = features[:total_num]
    dst_path = "feature.npy"
    dst_path = os.path.join(output_dir, dst_path)
    np.save(dst_path, features)

    if save_labels:
        label = label[:total_num]
        dst_label_path = "label.npy"
        dst_label_path = os.path.join(output_dir, dst_label_path)
        np.save(dst_label_path, label)

        index = index[:total_num]
        dst_index_path = "index.npy"
        dst_index_path = os.path.join(output_dir, dst_index_path)
        np.save(dst_index_path, index)



    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--feature_dir", type=str, default="/home/users/han.tang/data/eval/features", help="")
    parser.add_argument("--part_num", type=int, default=3, help="")
    parser.add_argument("--output_dir", type=str, default="/home/users/han.tang/data/eval/features/ValLife/cache_feature/subcenter", help="")
    parser.add_argument("--total_num", type=int, default=15498, help="")
    parser.add_argument("--save_labels", action="store_true", help="")
    args = parser.parse_args()
    main(args)
