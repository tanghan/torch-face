import numpy as np
import random
import cv2

class LmksJitter(object):
    def __init__(
            self, p, rng, data_shape, max_scale, max_angle, max_translation, **kwargs):
        self.data_shape = data_shape
        self.max_scale = max_scale
        self.max_angle = max_angle
        self.max_translation = max_translation
        self.rng = rng
        self.p = p

    def lmks_jitter_aug(self, img):
        new_size = [self.data_shape[1], self.data_shape[2]]  # h,w
        input_needs_trans = False
        if img.shape[-1] <= 3:  # H,W,C
            input_needs_trans = False
        else:
            input_needs_trans = True
            img = nd.tranpose(img, (1, 2, 0))
        h, w, c = img.shape
        trans_h = (random.random() - 0.5) * 2 * self.max_translation
        trans_w = (random.random() - 0.5) * 2 * self.max_translation
        angle = (random.random() - 0.5) * 2 * self.max_angle
        scale = (random.random() - 0.5) * 2 * self.max_scale + 1

        center = (w / 2 + trans_w, h / 2 + trans_h)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        target = cv2.warpAffine(img.asnumpy(), M, (w, h))
        img = nd.array(target)
        # Max Allowable range to prevent black fillings
        w_range = (w - new_size[1]) // 2
        w_s = (w - new_size[1]) // 2 +\
            np.clip(int(trans_w), w_range * -1, w_range * 1)
        h_range = (h - new_size[0]) // 2
        h_s = (h - new_size[0]) // 2 +\
            np.clip(int(trans_h), h_range * -1, h_range * 1)
        img = img[h_s:h_s + new_size[0], w_s:w_s + new_size[1], :]
        if input_needs_trans is True:
            img = nd.transpose(img, (2, 0, 1))  # HWC To CHW
        return img.astype('uint8')

    def __call__(self, img):
        assert img.shape[0] > self.data_shape[1] and\
            img.shape[1] > self.data_shape[2], \
            'lmks jitter needs rec image size larger than %dx%d' %\
            (self.data_shape[1], self.data_shape[2])

        do_augment = self.rng.uniform(0, 1)
        if do_augment < self.p:
            img = self.lmks_jitter_aug(img)

        return img


