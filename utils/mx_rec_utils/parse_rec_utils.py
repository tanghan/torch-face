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

