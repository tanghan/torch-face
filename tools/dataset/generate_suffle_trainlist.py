import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import argparse
import logging

from easydict import EasyDict as edict 
from importlib.machinery import SourceFileLoader

def generate_shuffle_trainlist(dataset_config):
    for dataset in dataset_config.trainset:
        print(dataset)
        print(dataset_config[dataset])


def main(args):
    config = args.config
    assert os.path.exists(config)
    opt = SourceFileLoader('module.name', './config/pretrain_config.py').load_module().opt
    dataset_config = opt.dataset
    generate_shuffle_trainlist(dataset_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config", type=str, default="./config/pretrain_config.py", help="")

    args = parser.parse_args()
    main(args)
