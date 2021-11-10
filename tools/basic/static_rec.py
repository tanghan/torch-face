import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

baseline = ["/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.rec",
            "/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.idx"]

glint360k = ["/home/users/han.tang/data/public_face_data/glint/glint360k/train.rec",
            "/home/users/han.tang/data/public_face_data/glint/glint360k/train.idx"]

megaface = ["/home/users/han.tang/data/public_face_data/faces_megafacetrain_112x112/train.rec",
            "/home/users/han.tang/data/public_face_data/faces_megafacetrain_112x112/train.idx"]

import mxnet as mx
from utils.mx_rec_utils.parse_rec_utils import unpack_fp64


def static_pulic_dataset(rec_path, idx_path):
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        imgidx = np.array(range(1, int(header.label[0])))
    else:
        imgidx = np.array(list(self.imgrec.keys))
    print(header.flag)  
    print(header.label)  
    print(len(imgidx))



def static_baseline_dataset(rec_path, idx_path):
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    s = imgrec.read_idx(0)
    header0, _ = unpack_fp64(s)
    id_seq = list(range(int(header0.label[0]),
                             int(header0.label[1])))
    id2range = {}
    id_num = {}
    imgidx = []
    for identity in id_seq:
        s = imgrec.read_idx(identity)
        header, _ = unpack_fp64(s)
        id_start, id_end = int(header.label[0]), int(header.label[1])
        img_num = id_end - id_start
        if img_num < 1:
            print(identity)
        id2range[identity] = (id_start, id_end)
        id_num[identity] = id_end - id_start
        imgidx += list(range(*id2range[identity]))

    print(len(id_seq))
    print(len(imgidx))


def main():
    #static_pulic_dataset(glint360k[0], glint360k[1])
    static_pulic_dataset(megaface[0], megaface[1])

if __name__ == "__main__":
    main()

