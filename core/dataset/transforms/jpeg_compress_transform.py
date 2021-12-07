import cv2
import numpy as np
import random

class JPEGCompress(object):
    """ Do JPEG compression to downgrade image quality

    Parameters
    ----------
    max_quality : int, (0, 100]
        JPEG compression highest quality
    min_quality : int, (0, 100]
        JPEG compression lowest quality
    """

    def __init__(self, p, rng, max_quality, min_quality):
        assert min_quality > 0 and max_quality <= 100
        assert min_quality <= max_quality
        self.max_quality = max_quality
        self.min_quality = min_quality
        self.p = p
        self.rng = rng

    def jpeg_compress_aug(self, img, scale, return_type=0):
        """
        Parameters
        ----------
        img : numpy.ndarray
            Input image.
        scale : float
            How much to compress
        return_type : [0, 1], default is 0
            0: return a mxnet.nd.NDArray, 1: return a numpy.ndarray

        Returns
        -------
        img : mxnet.nd.NDArray, numpy.ndarray
            Processed image.
        """
        quality = scale * (
                self.max_quality - self.min_quality) + self.min_quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        if random.random() < 0.5:
            res, encimg = cv2.imencode('.jpg', img, encode_param)
            decimg = cv2.imdecode(encimg, -1)
        else:
            img_bgr = img[:, :, [2, 1, 0]]
            res, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
            decimg = cv2.imdecode(encimg, -1)
            decimg = decimg[:, :, [2, 1, 0]]
        if return_type == 0:
            return decimg.astype('uint8')
        elif return_type == 1:
            return decimg
        else:
            raise ValueError('Choose return_type from [0, 1]')

    def __call__(self, img, label=None):
        """
        Parameters
        ----------
        img : numpy.ndarray, mxnet.nd.NDArray
            Input image.
        label : list or mx.nd.NDArray or None
            Image label

        Returns
        -------
        img : mxnet.nd.NDArray
            Processed image.
        label : list or mx.nd.NDArray or None
            Processed label.
        """
        do_augment = self.rng.uniform(0, 1)
        if do_augment < self.p:
            scale = random.random()
            img = self.jpeg_compress_aug(img, scale)
        return img


