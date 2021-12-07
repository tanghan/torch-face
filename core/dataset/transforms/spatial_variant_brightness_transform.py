import math
import numpy as np
import random

class SpatialVariantBrightness(object):
    """Spatial variant brightness, Enhanced Edition.
    Powered by xin.wang@horizon.ai.

    Parameters
    ----------
    brightness : float, default is 0.6
        Brightness ratio for this augmentation, the value choice
        in Uniform ~ [-brightness, brigheness].
    max_template_type : int, default is 3
        Max number of template type in once process. Note,
        the selection process is repeated.
    online_template : bool, default is False
        Template generated online or offline.
        "False" is recommended to get fast speed.
    """

    def __init__(self, p, rng, brightness=0.6, max_template_type=3,
                 online_template=False):
        self.brightness = brightness
        self.max_template_type = max_template_type
        self.online_template = online_template
        self.template_in_cache = False
        self.p = p
        self.rng = rng

    def _get_line_coeff(self, angle):
        sin_x = math.sin(angle) ** 2 * (2 * (math.sin(angle) > 0) - 1)
        cos_x = math.cos(angle) ** 2 * (2 * (math.cos(angle) > 0) - 1)
        return sin_x, cos_x

    def _normalize_template(self, template_h):
        min_value, max_value = np.min(template_h), np.max(template_h)
        template_h = (template_h - min_value) / (max_value - min_value)  # noqa
        return template_h

    def _linear_template(self, h, w, angle):
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = (sin_x * x / w + cos_x * y / h) / 2
        return self._normalize_template(template_h)

    def _qudratical_template(self, h, w, angle):
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = (sin_x * x / w + cos_x * y / h) ** 2
        return self._normalize_template(template_h)

    def _parabola_template(self, h, w, angle):
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = ((sin_x * x / w + cos_x * y / h) - 0.5) ** 2  # noqa
        return self._normalize_template(template_h)

    def _cubic_template(self, h, w, angle):
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = (sin_x * x / w + cos_x * y / h) ** 3  # noqa
        return self._normalize_template(template_h)

    def _sinwave_template(self, h, w, angle, frequency, theta):
        # sin (fx + theta)
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = (math.sin((sin_x * x / w + cos_x * y / h) * frequency * np.pi +  # noqa
                                             theta * np.pi / 180.0) + 1) / 2  # noqa
        return self._normalize_template(template_h)

    def generate_template(self, h, w):
        # `sinwave` has a bigger proportion than others.
        temp_types = ['parabola', 'linear',
                      'qudratical', 'cubic', 'sinwave', 'sinwave']
        idxs = np.random.randint(0, len(temp_types),
                                 random.randint(1, self.max_template_type))
        temp_type_list = [temp_types[i] for i in idxs]
        template_h_list = []
        for temp_type in temp_type_list:
            template_h = np.ones((h, w))
            angle = random.randint(0, 360) * np.pi / 180.
            if temp_type == 'parabola':
                template_h = self._parabola_template(h, w, angle)
            elif temp_type == 'linear':
                template_h = self._linear_template(h, w, angle)
            elif temp_type == 'qudratical':
                template_h = self._qudratical_template(h, w, angle)
            elif temp_type == 'cubic':
                template_h = self._cubic_template(h, w, angle)
            elif temp_type == 'sinwave':
                frequency = random.choice([0.5, 1, 1.5, 2, 3])
                theta = random.choice([0, 30, 60, 90])
                self._sinwave_template(h, w, angle, frequency, theta)
            template_h_list.append(template_h)
        return np.mean(np.dstack(template_h_list), axis=2, keepdims=True)

    def generate_template_offline(self, h, w):
        if self.template_in_cache is False:
            pi = 3.14159
            line_angle_list = np.arange(0, 360, 10) * pi / 180.0
            template_h_list = []
            # Parabola
            for angle in line_angle_list:
                template_h_list.append(self._parabola_template(h, w, angle))

            # Linearly Light Change
            for angle in line_angle_list:
                template_h_list.append(self._linear_template(h, w, angle))
            # Qudratically Light Change
            for angle in line_angle_list:
                template_h_list.append(self._qudratical_template(h, w, angle))
            # Cubicly Light Change
            for angle in line_angle_list:
                template_h_list.append(self._cubic_template(h, w, angle))
            # Sinwave Light Change
            frequency_list = [0.5, 1, 1.5, 2, 3]
            theta_list = [0, 30, 60, 90]
            for frequency in frequency_list:
                for theta in theta_list:
                    for angle in line_angle_list:
                        template_h_list.append(self._sinwave_template(
                            h, w, angle, frequency, theta))
            self.template_list = template_h_list
            self.template_in_cache = True
            self.cache_template_height = h
            self.cache_template_width = w

        assert (self.cache_template_height == h and
                self.cache_template_width == w), (
            "image shape change detected, please use online "
            "spatial-variant-brightness"
            "or write code to support tmplate resize")
        selected_template_num = np.random.randint(self.max_template_type) + 1
        choice_list = np.random.choice(len(self.template_list),
                                       selected_template_num)
        r_template = self.template_list[choice_list[0]].copy()
        for i in range(selected_template_num - 1):
            r_template += self.template_list[choice_list[i + 1]]
        r_template = r_template / selected_template_num
        return r_template

    def process(self, image):
        h, w = image.shape[:2]
        if self.online_template:
            template_h = self.generate_template(h, w).reshape((h, w, 1))
        else:
            template_h = self.generate_template_offline(h, w).reshape(
                (h, w, 1))
        template_r = np.broadcast_to(
            template_h,
            (template_h.shape[0], template_h.shape[1], image.shape[2]))
        c = random.uniform(-self.brightness, self.brightness)
        image = image * (1 + template_r * c)
        return np.clip(image, 0, 255)

    def __call__(self, image, label=None):
        do_augment = self.rng.uniform(0, 1)
        if do_augment < self.p:
            image = self.process(image)
        return image

