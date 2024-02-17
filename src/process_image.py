import numpy as np


def get_pixel(im, col, row, channel):
    height, width = im['data'].shape[0], im['data'].shape[1]
    if col < 0 or row < 0:
        selected_col = 0
        selected_row = 0
    elif col > height or row > width:
        selected_col = height - 1
        selected_row = width - 1
    else:
        selected_col = col - 1 if col != 0 else col
        selected_row = row - 1 if row != 0 else row
    return im['data'][selected_row, selected_col][channel]


def set_pixel(im, col, row, channel, value):
    im['data'][row, col][channel] = value


def make_image(width, height, channel):
    if channel == 1:
        return {'data': np.zeros((height, width), dtype=np.float32), 'w': width, 'h': height, 'c': channel}
    else:
        return {'data': np.zeros((height, width, channel), dtype=np.float32), 'w': width, 'h': height, 'c': channel}


def copy_image(im):
    pixel_arr = [pixel for pixel in im['data']]
    copy = make_image(im['w'], im['h'], im['c'])
    copy['data'] = np.array(pixel_arr)
    return copy


def rgb_to_grayscale(im):
    assert im['c'] == 3
    gray = make_image(im['w'], im['h'], 1)
    for row in range(im['h']):
        for col in range(im['w']):
            R = im['data'][row, col, 0]
            G = im['data'][row, col, 1]
            B = im['data'][row, col, 2]
            luma = (0.229 * R) + (0.587 * G) + (0.114 * B)
            gray['data'][row, col] = luma
    return gray


def shift_image(im, channel, value):
    factor = 1 + value
    for row in range(im['h']):
        for col in range(im['w']):
            value = im['data'][row, col, channel]
            im['data'][row, col, channel] = value * factor


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
