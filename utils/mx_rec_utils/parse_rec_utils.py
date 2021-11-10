import struct
from mxnet.recordio import IRHeader
import numpy as np

_IR_FORMAT_64 = 'IdQQ'
_IR_SIZE_64 = struct.calcsize(_IR_FORMAT_64)

def unpack_fp64(s):
    header = IRHeader(*struct.unpack(_IR_FORMAT_64, s[:_IR_SIZE_64]))
    s = s[_IR_SIZE_64:]
    if header.flag > 0:
        header = header._replace(
            label=np.frombuffer(s, np.float64, header.flag))
        s = s[header.flag*8:]
    return header, s


def static_indexed_rec(rec_path, idx_path, fp64=False):
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    s = imgrec.read_idx(0)
    class_num = 0
    total_imgs = 0
    if fp64:
        header0, _ = unpack_fp64(s)
        id_seq = list(range(int(header0.label[0]),
                                 int(header0.label[1])))
        class_num = len(id_seq)
        s = self.imgrec.read_idx(identity)
        header, _ = unpack_fp64(s)
        id_start, id_end = int(header.label[0]), int(header.label[1])
        id2range[identity] = (id_start, id_end)
        id_num[identity] = id_end - id_start
        imgidx += list(range(*self.id2range[identity]))
    else:
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            header0 = (int(header.label[0]), int(header.label[1]))
            imgidx = np.array(range(1, int(header.label[0])))
        else:
            imgidx = np.array(list(imgrec.keys))
        
    


