import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import struct
import numpy as np

from utils.eval.verification import evaluate

def main(args):
    part_num = args.part_num
    fea_path_prefix = args.fea_path_prefix
    
    embed_list = []
    for i in range(part_num):
        fea_path = "{}.bin".format(i)
        fea_path = os.path.join(fea_path_prefix, fea_path)
        #with open(fea_path, "rb") as fr:
        #    raw_data = fr.read()
        data = np.fromfile(fea_path, dtype=np.float32)
        embed = data.reshape(-1, 512)
        print(embed.shape)
        embed_list.append(embed)
    embedding = np.stack(embed_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fea_path_prefix", type=str, default="/home/users/han.tang/data/eval/features/", help="")
    parser.add_argument("--part_num", type=int, default=4, help="")
    args = parser.parse_args()
    main(args)

