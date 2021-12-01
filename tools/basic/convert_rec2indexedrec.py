import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import mxnet as mx

import argparse

#rec_path = "/home/users/han.tang/data/test/val/ValID/wanren_V0.2.rec"
#dst_rec_path = "/home/users/han.tang/data/test/val/ValID/wanren_V0.2_indexed.rec"
#dst_idx_path = "/home/users/han.tang/data/test/val/ValID/wanren_V0.2_indexed.idx"

def test_write(rec_path, dst_rec_path, dst_idx_path):
    src_rec = mx.recordio.MXRecordIO(rec_path, "r")
    index_rec = mx.recordio.MXIndexedRecordIO(dst_idx_path, dst_rec_path, "w")
    idx = 0
    while True:
        s = src_rec.read()
        if not s:
            break
        '''
        header, s = mx.recordio.unpack(s)
        img = mx.image.imdecode(s)
        print(header)
        print(img.shape)
        '''
        index_rec.write_idx(idx, s)
        idx += 1
    index_rec.close()
    src_rec.close()
    print("total num:", idx)

def test_read(rec_path, dst_rec_path, dst_idx_path):

    src_rec = mx.recordio.MXRecordIO(rec_path, "r")
    index_rec = mx.recordio.MXIndexedRecordIO(dst_idx_path, dst_rec_path, "r")

    for i in range(5):
        src_s = src_rec.read()
        dst_s = index_rec.read_idx(i)

        header0, s0 = mx.recordio.unpack(src_s)
        header1, s1 = mx.recordio.unpack(dst_s)
        print(header0)
        print(header1)
    index_rec.close()
    src_rec.close()

def main(args):
    src_rec_path = args.src_rec_path
    dst_rec_path = args.dst_rec_path
    dst_idx_path = args.dst_idx_path
    if args.write:
        test_write(src_rec_path, dst_rec_path, dst_idx_path)
    else:
        test_read(src_rec_path, dst_rec_path, dst_idx_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--src_rec_path", type=str, default=None, help="")
    parser.add_argument("--dst_rec_path", type=str, default=None, help="")
    parser.add_argument("--dst_idx_path", type=str, default=None, help="")
    parser.add_argument("--write", action="store_true", help="")
    args = parser.parse_args()
    main(args)

