import numpy as np


def get_pixel(im, x, y, c):
    height, width = im['data'].shape[0], im['data'].shape[1]
    select_x = max(0, min(x, width - 1))
    select_y = max(0, min(y, height - 1))
    return im['data'][select_y, select_x][c]


def set_pixel(im, x, y, c, v):
    im['data'][y, x][c] = v


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
    for y in range(im['h']):
        for x in range(im['w']):
            R = im['data'][y, x, 0]
            G = im['data'][y, x, 1]
            B = im['data'][y, x, 2]
            luma = (0.229 * R) + (0.587 * G) + (0.114 * B)
            gray['data'][y, x] = luma
    return gray


def shift_image(im, channel, value):
    factor = 1 + value
    for y in range(im['h']):
        for x in range(im['w']):
            value = get_pixel(im, x, y, channel)
            im['data'][y, x][channel] = value * factor


def clamp_image(im):
    for y in range(im['h']):
        for x in range(im['w']):
            r_value = get_pixel(im, x, y, 0)
            g_value = get_pixel(im, x, y, 1)
            b_value = get_pixel(im, x, y, 2)
            fixed_r_value = max(0, min(r_value, 1))
            fixed_g_value = max(0, min(g_value, 1))
            fixed_b_value = max(0, min(b_value, 1))
            set_pixel(im, x, y, 0, fixed_r_value)
            set_pixel(im, x, y, 1, fixed_g_value)
            set_pixel(im, x, y, 2, fixed_b_value)


def rgb_to_hsv(im):
    for y in range(im['h']):
        for x in range(im['w']):
            r = get_pixel(im, x, y, 0)
            g = get_pixel(im, x, y, 1)
            b = get_pixel(im, x, y, 2)
            V = max(r, g, b)
            m = min(r, g, b)
            C = V - m
            if r == 0.0 and g == 0.0 and b == 0.0:
                S = 0
            else:
                S = C / V
            h_prime = calculate_h_prime(C, V, r, g, b)
            if C == 0:
                h = 0
            else:
                h = calculate_h(h_prime)
            set_pixel(im, x, y, 0, h)
            set_pixel(im, x, y, 1, S)
            set_pixel(im, x, y, 2, V)


def calculate_h_prime(C, V, r, g, b):
    if C == 0:
        return 0
    elif V == r:
        return (g - b) / C
    elif V == g:
        return ((b - r) / C) + 2
    elif V == b:
        return ((r - g) / C) + 4


def calculate_h(h_prime):
    if h_prime < 0:
        return (h_prime / 6) + 1
    else:
        return h_prime / 6


def hsv_to_rgb(im):
    for y in range(im['h']):
        for x in range(im['w']):
            H = get_pixel(im, x, y, 0)
            S = get_pixel(im, x, y, 1)
            V = get_pixel(im, x, y, 2)
            C = V * S
            H_prime = H * 6
            X = C * (1 - abs((H_prime % 2) - 1))
            m = V - C
            r, g, b = get_rgb(H_prime, X, C)
            set_pixel(im, x, y, 0, r + m)
            set_pixel(im, x, y, 1, g + m)
            set_pixel(im, x, y, 2, b + m)


def get_rgb(H_prime, X, C):
    if 0 <= H_prime < 1:
        return C, X, 0
    elif 1 <= H_prime < 2:
        return X, C, 0
    elif 2 <= H_prime < 3:
        return 0, C, X
    elif 3 <= H_prime < 4:
        return 0, X, C
    elif 4 <= H_prime < 5:
        return X, 0, C
    elif 5 <= H_prime < 6:
        return C, 0, X


def scale_image(im, scale):
    for y in range(im['h']):
        for x in range(im['w']):
            s_value = get_pixel(im, x, y, 1)
            set_pixel(im, x, y, 1, s_value*scale)