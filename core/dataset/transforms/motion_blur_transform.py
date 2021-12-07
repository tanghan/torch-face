import numpy as np
import cv2
import math

def motion_blur(img, length_min, length_max, angle_min, angle_max):
    length = np.random.randint(length_min, length_max)
    angle = np.random.randint(angle_min, angle_max)
    if angle in [0, 90, 180, 270, 360]:
        angle += 1

    half = length / 2
    EPS = np.finfo(float).eps
    alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1

    # blur kernel size
    sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
    sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
    psf1 = np.zeros((sy, sx))

    # psf1 is getting small when (x, y) move from left-top to right-bottom
    # at this moment (x, y) is moving from right-bottom to left-top
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
            rad = math.sqrt(i * i + j * j)
            if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs(
                    (j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp * temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
            if psf1[i][j] < 0:
                psf1[i][j] = 0

    # anchor is (0, 0) when (x, y) is moving towards left-top
    anchor = (0, 0)
    # anchor is (width, heigth) when (x, y) is moving towards right-top
    if angle < 90 and angle > 0:  # flip kernel at this moment
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, 0)
    elif angle > -90 and angle < 0:  # moving towards right-bottom
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)
    elif angle < -90:  # moving towards left-bottom
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)
    psf1 = psf1 / psf1.sum()

    img_blur = cv2.filter2D(src=img, ddepth=-1, kernel=psf1, anchor=anchor)
    return img_blur

def gaussian_blur(img,
                  kernel_size_min, kernel_size_max, sigma_min, sigma_max):
    k = np.random.randint(kernel_size_min, kernel_size_max)
    if k % 2 == 0:
        if np.random.rand() > 0.5:
            k += 1
        else:
            k -= 1
    s = np.random.uniform(sigma_min, sigma_max)
    img_blur = cv2.GaussianBlur(src=img, ksize=(k, k), sigmaX=s)
    return img_blur


class MotionBlur(object):
    def __init__(self, p, rng, length_min, length_max, angle_min, angle_max):
        self.length_min = length_min
        self.length_max = length_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.p = p
        self.rng = rng

    def __call__(self, img):

        do_augment = self.rng.uniform(0, 1)
        if do_augment < self.p:
            img = motion_blur(img, self.length_min, self.length_max,
                          self.angle_min, self.angle_max)

            img = np.clip(img, 0, 255)
        return img

class GaussianBlur(object):
    def __init__(self, p, rng,
                 kernel_size_min, kernel_size_max, sigma_min, sigma_max):
        self.kernel_size_min = kernel_size_min
        self.kernel_size_max = kernel_size_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p
        self.rng = rng

    def __call__(self, img):
        do_augment = self.rng.uniform(0, 1)
        if do_augment < self.p:

            img = gaussian_blur(img, self.kernel_size_min, self.kernel_size_max,
                            self.sigma_min, self.sigma_max)
            img = np.clip(img, 0, 255)

        return img

