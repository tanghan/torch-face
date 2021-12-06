import torch
import numpy as np

import cv2

class To3CGray(object):
    def __init__(self, p, rng):
        self.p = p
        self.rng = rng

    def __call__(self, img):
        do_to_gray = self.rng.uniform(0, 1)
        if do_to_gray < self.p:
            new_img = np.zeros(img.shape)
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            for i in range(new_img.shape[-1]):
                new_img[:, :, i] = gray_img

            return new_img
        else:
            return img

