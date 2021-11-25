import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from collections import namedtuple
import argparse

TestInfo = namedtuple("TestInfo", ["dataset_name", "dataset_type", "total_num", "output_dir"])

ijbc_info = TestInfo(dataset_name="ijbc", dataset_type="rec", total_num=469375, output_dir="/home/users/han.tang/data/eval/features/IJBC_V135PNGAff/cache_feature/subcenter")
j2_info = TestInfo(dataset_name="Val_J2_RealCar", dataset_type="rec", total_num=2429, output_dir="/home/users/han.tang/data/eval/features/Val_J2_RealCar/cache_feature/subcenter")
life_info = TestInfo(dataset_name="ValLife", dataset_type="rec", total_num=15498, output_dir="/home/users/han.tang/data/eval/features/ValLife/cache_feature/subcenter")
id_info = TestInfo(dataset_name="ValID", dataset_type="rec", total_num=19897, output_dir="/home/users/han.tang/data/eval/features/ValID/cache_feature/subcenter")
val30w_query_info = TestInfo(dataset_name="Val30W_query", dataset_type="rec", total_num=21669, output_dir="/home/users/han.tang/data/eval/features/Val30W/cache_feature/subcenter/query")
val30w_gallery_info = TestInfo(dataset_name="Val30W_gallery", dataset_type="rec", total_num=300000, output_dir="/home/users/han.tang/data/eval/features/Val30W/cache_feature/subcenter/gallery")
dms_info = TestInfo(dataset_name="Val_DMS_Car", dataset_type="rec", total_num=4960, output_dir="/home/users/han.tang/data/eval/features/Val_DMS_Car/cache_feature/subcenter")

test_list = []
#test_list.append(ijbc_info)
#test_list.append(j2_info)
#test_list.append(life_info)
#test_list.append(id_info)
#test_list.append(val30w_query_info)
#test_list.append(val30w_gallery_info)
test_list.append(dms_info)


def do_all_feature_extract(gpu_num, weight_path, batch_size):
    for test_info in test_list:
        dataset_name = test_info.dataset_name
        dataset_type = test_info.dataset_type
        total_num = test_info.total_num
        output_dir = test_info.output_dir

        command_feature_extract = "python3 tools/eval/feature_extract.py "
        command_feature_extract += "--dataset {} ".format(dataset_name)
        command_feature_extract += "--dataset_type {} ".format(dataset_type)
        command_feature_extract += "--gpu_num {} ".format(gpu_num)
        command_feature_extract += "--weight_path {} ".format(weight_path)
        command_feature_extract += "--batch_size {}".format(batch_size)
        print(command_feature_extract)
        assert os.system(command_feature_extract) == 0

        command_combine = "python3 tools/eval/combine_features.py "
        command_combine += "--part_num {} ".format(gpu_num)
        command_combine += "--total_num {} ".format(total_num)
        command_combine += "--output_dir {} ".format(output_dir)
        command_combine += "--save_labels"
        print(command_combine)
        assert os.system(command_combine) == 0


def main(args):
    gpu_num = args.gpu_num
    weight_path = args.weight_path
    batch_size = args.batch_size
    do_all_feature_extract(gpu_num, weight_path, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--gpu_num", type=int, default=2, help="")
    parser.add_argument("--batch_size", type=int, default=512, help="")
    parser.add_argument("--weight_path", type=str, default="/home/users/han.tang/workspace/pretrain_models/glint360k_cosface_r100_fp16_0.1/backbone.pth", help="")
    args = parser.parse_args()
    main(args)


