import numpy as np
import cv2
import random

class RandomOcclusion(object):
    """Randomly occlusion the image with circles or rectangles.

    Parameters
    ----------
    occ_type : str, optional, default "whole"
        Occlusion type, `whole` and `forehead` are available.
    occ_num: int, optional, defualt 1
        Number of occlusion regions.
    color_ratio : float, optional, default 0.5
        Probability of using random color.
    shape_ratio : float, optional, default 0.5
        Probability of circle block, only for `whole` type.
    size_ratio : float, optional, default 0.3
        Occlusion size of height and width.
        In `whole` mode, it's the upper bound of occlusion on the whole image.
        In `forehead` mode, it's the lower bound of occlusion on the top 1/3.
    func : callable, optional, default None
        Label processing funciton.
    """

    def __init__(self, occ_type="whole", occ_num=1, color_ratio=0.5,
                 shape_ratio=0.5, size_ratio=0.3, func=None, **kwargs):
        self.occ_type = occ_type
        self.occ_num = occ_num
        self.color_ratio = color_ratio
        self.shape_ratio = shape_ratio
        self.radius_ratio = size_ratio / 2.0
        self.func = func
        self.kwargs = kwargs

    def __call__(self, img):

        img_h, img_w, _ = img.shape
        for _ in range(self.occ_num):
            if random.random() <= self.color_ratio:
                color = (np.mean(img[:, :, 0]),
                         np.mean(img[:, :, 1]),
                         np.mean(img[:, :, 2]))
                color = tuple(map(int, color))
            else:
                color = (random.randint(0, 255),
                         random.randint(0, 255),
                         random.randint(0, 255))

            if self.occ_type == "whole":
                center = (random.randint(0, img_w), random.randint(0, img_h))
                if random.random() <= self.shape_ratio:
                    radius = int(img_h * random.uniform(0, self.radius_ratio))
                    cv2.circle(img, center, radius, color, -1)
                else:
                    radius_x = int(
                        img_w * random.uniform(0, self.radius_ratio))
                    radius_y = int(
                        img_h * random.uniform(0, self.radius_ratio))
                    left_top = (center[0] - radius_x, center[1] - radius_y)
                    right_down = (center[0] + radius_x, center[1] + radius_y)
                    cv2.rectangle(img, left_top, right_down, color, -1)
            elif self.occ_type == "forehead":
                max_h = img_h // 3
                center = (random.randint(0, img_w), random.randint(0, max_h))
                radius_x = int(img_w * random.uniform(self.radius_ratio, 1.0))
                radius_y = int(max_h * random.uniform(self.radius_ratio, 1.0))
                left_top = (center[0] - radius_x, center[1] - radius_y)
                right_down = (center[0] + radius_x, center[1] + radius_y)
                cv2.rectangle(img, left_top, right_down, color, -1)
            elif self.occ_type == "edge":
                ratio = random.uniform(1/4, 1/3)
                prob = random.uniform(0, 1)
                color = (0, 0, 0)
                if prob < 0.25:  # top
                    cv2.rectangle(
                        img, (0, 0), (img_w, int(img_h * ratio)), color, -1
                    )
                elif prob < 0.5:  # down
                    cv2.rectangle(
                        img, (0, int(img_h * (1 - ratio))), (img_w, img_h),
                        color, -1
                    )
                elif prob < 0.75:  # left
                    cv2.rectangle(
                        img, (0, 0), (int(img_w * ratio), img_h), color, -1
                    )
                else:
                    cv2.rectangle(
                        img, (int(img_w * (1 - ratio)), 0), (img_w, img_h),
                        color, -1
                    )
            else:
                raise ValueError("Not supported block type.")

        return img


