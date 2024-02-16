import numpy as np


def get_pixel(im, x, y, c):
    # TODO Fill this in
    return 0


def set_pixel(im, x, y, c, v):
    # TODO Fill this in
    pass


def make_image(w, h, c):
    return {'data': np.zeros((h, w, c), dtype=np.float32), 'w': w, 'h': h, 'c': c}


def copy_image(im):
    copy = make_image(im['w'], im['h'], im['c'])
    # TODO Fill this in
    return copy


def rgb_to_grayscale(im):
    assert im['c'] == 3
    gray = make_image(im['w'], im['h'], 1)
    # TODO Fill this in
    return gray


def shift_image(im, c, v):
    # TODO Fill this in
    pass


def clamp_image(im):
    # TODO Fill this in
    pass


# These might be handy
def three_way_max(a, b, c):
    return max(a, max(b, c))


def three_way_min(a, b, c):
    return min(a, min(b, c))


def rgb_to_hsv(im):
    # TODO Fill this in
    pass


def hsv_to_rgb(im):
    # TODO Fill this in
    pass
