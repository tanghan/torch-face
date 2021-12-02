import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import cv2

baseline = ["/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.rec",
            "/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.idx"]

glint360k = ["/home/users/han.tang/data/public_face_data/glint/glint360k/train.rec",
            "/home/users/han.tang/data/public_face_data/glint/glint360k/train.idx"]

megaface = ["/home/users/han.tang/data/public_face_data/faces_megafacetrain_112x112/train.rec",
            "/home/users/han.tang/data/public_face_data/faces_megafacetrain_112x112/train.idx"]

j2 = ["/home/users/han.tang/data/test/val/Val_J2_RealCar/Val_J2_RealCar.rec",
            "/home/users/han.tang/data/test/val/Val_J2_RealCar/Val_J2_RealCar.idx"]

ValID = ["/home/users/han.tang/data/test/val/ValID/wanren_V0.2_indexed.rec",
            "/home/users/han.tang/data/test/val/ValID/wanren_V0.2_indexed.idx"]

abtdge_id1w = ["/home/users/han.tang/data/abtdge_id1w_Above9_miabove10_20200212/abtdge_id1w_Above9_miabove10_20200212.rec",
        "/home/users/han.tang/data/abtdge_id1w_Above9_miabove10_20200212/abtdge_id1w_Above9_miabove10_20200212.idx"]

dms_car = ["/home/users/han.tang/data/test/val/Val_DMS_Car/DMS_Car.rec",
        "/home/users/han.tang/data/test/val/Val_DMS_Car/DMS_Car.idx"]

import mxnet as mx
from utils.mx_rec_utils.parse_rec_utils import unpack_fp64


def static_pulic_dataset(rec_path, idx_path):
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        imgidx = np.array(range(1, int(header.label[0])))
    else:
        imgidx = np.array(list(imgrec.keys))

    print("flag: ", header.flag)  
    print("label: ", header.label)  
    print(len(imgidx))
    for i in range(23, 29):
    
        s = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(s)
        img = mx.image.imdecode(img)
        print(img.shape)
        #cv2.imwrite("j2_temp_{}.jpg".format(i), img.asnumpy())
        img = img.asnumpy()
        
        r = np.mean(img[:, :, 0])
        g = np.mean(img[:, :, 1])
        b = np.mean(img[:, :, 2])

        b, g, r = np.split(img, 3, -1)
        '''
        print("r shape: {}".format(r.shape))
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        print("bH shape: {}".format(bH.shape))
        img = np.stack([bH, gH, rH], -1)
        '''
        
        #print(r, g, b)
        print(cv2.meanStdDev(img))
        cv2.imwrite("dms_temp_{}.jpg".format(i), img)

        #print(im.shape)
        #print(im.shape)



def static_baseline_dataset(rec_path, idx_path):
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    s = imgrec.read_idx(0)
    header0, _ = unpack_fp64(s)
    id_seq = list(range(int(header0.label[0]),
                             int(header0.label[1])))
    id2range = {}
    id_num = {}
    imgidx = []
    print("head: ", header0)
    for identity in id_seq:
        s = imgrec.read_idx(identity)
        header, _ = unpack_fp64(s)
        id_start, id_end = int(header.label[0]), int(header.label[1])
        img_num = id_end - id_start
        if img_num < 1:
            print("id: ", identity)
        id2range[identity] = (id_start, id_end)
        id_num[identity] = id_end - id_start
        imgidx += list(range(*id2range[identity]))

    last_idx = imgidx[-1]
    s = imgrec.read_idx(last_idx)
    header, _ = mx.recordio.unpack(s)
    print("last idx: {}".format(last_idx))
    print(header.label)
    print("id len: {}".format(len(id_seq)))
    #print(len(imgidx))


def main():
    #static_pulic_dataset(abtdge_id1w[0], abtdge_id1w[1])
    #static_baseline_dataset(abtdge_id1w[0], abtdge_id1w[1])
    #static_pulic_dataset(ValID[0], ValID[1])
    #static_pulic_dataset(megaface[0], megaface[1])
    static_pulic_dataset(dms_car[0], dms_car[1])
    #static_pulic_dataset(j2[0], j2[1])


if __name__ == "__main__":
    main()

