import torch
import numpy as np

import cv2

class NormalizeTransformer(object):
  def __init__(self, bias, scale):
    self.bias = bias
    self.scale = scale
    
  def __call__(self, img):
    img = img.astype(np.float32)
    img = (img - self.bias) * self.scale
    return img

class ToTensor(object):
  def __call__(self, img):
    if img.ndim == 2:
      img = img[np.newaxis, :]
    elif img.ndim == 3:
      img = img.transpose((2, 0, 1))
    else:
      raise ValueError('Invalid image dimensions.')
    return torch.from_numpy(img).contiguous()

class MirrorTransformer(object):

  def __call__(self, img):
    if np.random.randint(0, 2):
      img = cv2.flip(img, 1)
    return img
