import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import mxnet as mx
from mxnet.recordio import IRHeader
import struct
import numpy as np

_IR_FORMAT_64 = 'IdQQ'
_IR_SIZE_64 = struct.calcsize(_IR_FORMAT_64)


baseline_path = "/home//users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2"

def unpack_fp64(s):
    header = IRHeader(*struct.unpack(_IR_FORMAT_64, s[:_IR_SIZE_64]))
    s = s[_IR_SIZE_64:]
    if header.flag > 0:
        header = header._replace(
            label=np.frombuffer(s, np.float64, header.flag))
        s = s[header.flag*8:]
    return header, s

def parse_baseline_data():
    rec_path = "{}.rec".format(baseline_path)
    idx_path = "{}.idx".format(baseline_path)

    record = mx.recordio.MXIndexedRecordIO(
                idx_path, rec_path, 'r')
    s = record.read_idx(0)
    header, s = unpack_fp64(s)
    id_seq = list(range(int(header.label[0]),
                             int(header.label[1])))

    img_idx = []
    process_class_num = 0
    for identity in id_seq:
        s = record.read_idx(identity)
        header, _ = unpack_fp64(s)
        id_start, id_end = int(header.label[0]), int(header.label[1])
        #print(id_start, id_end)
        img_idx += list(range(id_start, id_end))
        process_class_num += 1
        if process_class_num % 5000 == 0:
            print("process_class_num: {}".format(process_class_num))

    print(len(img_idx))



    #print(header)
    #print(s)


def main():
    parse_baseline_data()

if __name__ == "__main__":
    main()



