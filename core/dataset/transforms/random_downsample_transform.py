import numpy as np
import random
import cv2

class RandomDownSample(object):
    """ First downsample and upsample to original size.

    Parameters
    ----------
    data_shape : tuple of ints
        C, H, W
    min_downsample_width: int
        minimum downsample width
    inter_method : int
        interpolation method index
    """

    def __init__(self, p, rng, data_shape, min_downsample_width, inter_method):
        self.data_shape = data_shape
        self.min_downsample_width = min_downsample_width
        self.inter_method = inter_method
        self.p = p
        self.rng = rng

    def rand_downsample_aug(self, img, scale, return_type=0):
        """
        Parameters
        ----------
        img : mxnet.nd.NDArray
            Input image.
        scale : float
            How much to downsample
        return_type : [0, 1], default is 0
            0: return a mxnet.nd.NDArray, 1: return a numpy.ndarray

        Returns
        -------
        img : mxnet.nd.NDArray, numpy.ndarray
            Processed image.
        """

        def get_inter_method(inter_method, sizes=()):
            if inter_method == 9:
                if sizes:
                    assert len(sizes) == 4
                    oh, ow, nh, nw = sizes
                    if nh > oh and nw > ow:
                        return 2
                    elif nh < oh and nw < ow:
                        return 3
                    else:
                        return 1
                else:
                    return 2
            if inter_method == 10:
                return random.randint(0, 4)
            return inter_method

        new_w = int(
            scale * (self.data_shape[2] - self.min_downsample_width) +
            self.min_downsample_width)
        new_h = int(new_w * self.data_shape[2] / self.data_shape[1])
        org_w = int(self.data_shape[2])
        org_h = int(self.data_shape[1])
        interpolation_method = get_inter_method(
            self.inter_method,
            (self.data_shape[1], self.data_shape[2], new_h, new_w))
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation_method)
        img = cv2.resize(img, (org_w, org_h), interpolation=interpolation_method)
        if return_type == 0:
            return img.astype('uint8')
        elif return_type == 1:
            return img.astype('uint8')
        else:
            raise ValueError('Choose return_type from [0, 1]')

    def __call__(self, img):
        """
        Parameters
        ----------
        img : mxnet.nd.NDArray, numpy.ndarray
            Input image.
        label : list or mx.nd.NDArray or None
            Processed label.

        Returns
        -------
        img : mxnet.nd.NDArray, numpy.ndarray
            Processed image.
        label : list or mx.nd.NDArray or None
            Processed label.
        """
        do_augment = self.rng.uniform(0, 1)
        if do_augment < self.p:
            scale = random.random()
            img = self.rand_downsample_aug(img, scale)
        return img


