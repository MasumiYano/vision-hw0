import numpy as np


def get_pixel(im, col, row, channel):
    height, width = im['data'].shape[0], im['data'].shape[1]
    selected_col = 0
    selected_row = 0
    if col > height:
        selected_col = height
    elif row > width:
        selected_row = width
    else:
        selected_col = col
        selected_row = row
    return im['data'][selected_row, selected_col][channel]


def set_pixel(im, col, row, channel, value):
    existing_value = get_pixel(im, col, row, channel)
    existing_value = value


def make_image(width, height, channel):
    return {'data': np.zeros((height, width, channel), dtype=np.float32), 'w': width, 'h': height, 'c': channel}


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
