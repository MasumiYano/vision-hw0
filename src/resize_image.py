import math
import numpy as np
from src import process_image


def nn_interpolate(im, x, y, c):
    return


def nn_resize(im, w, h):
    channel = im['c']
    new_im = process_image.make_image(w, h, channel)
    r_value = im['data'][:, :, 0]
    g_value = im['data'][:, :, 1]
    b_value = im['data'][:, :, 2]
    return new_im


def bilinear_interpolate(im, x: float, y: float, c: list[int]) -> tuple:
    distance_btm = abs(1 - y)
    distance_top = abs(1 - distance_btm)
    distance_right = abs(1 - x)
    distance_left = x
    # Q1
    r_top_q1 = process_image.get_pixel(im, math.floor(x), math.floor(y), c[0])
    g_top_q1 = process_image.get_pixel(im, math.floor(x), math.floor(y), c[1])
    b_top_q1 = process_image.get_pixel(im, math.floor(x), math.floor(y), c[2])
    r_btm_q1 = process_image.get_pixel(im, math.floor(x), math.ceil(y), c[0])
    g_btm_q1 = process_image.get_pixel(im, math.floor(x), math.ceil(y), c[1])
    b_btm_q1 = process_image.get_pixel(im, math.floor(x), math.ceil(y), c[2])
    r1 = (distance_btm * r_top_q1) + (distance_top * r_btm_q1)
    g1 = (distance_btm * g_top_q1) + (distance_top * g_btm_q1)
    b1 = (distance_btm * b_top_q1) + (distance_top * b_btm_q1)

    # Q2
    r_top_q2 = process_image.get_pixel(im, math.ceil(x), math.floor(y), c[0])
    g_top_q2 = process_image.get_pixel(im, math.ceil(x), math.floor(y), c[1])
    b_top_q2 = process_image.get_pixel(im, math.ceil(x), math.floor(y), c[2])
    r_btm_q2 = process_image.get_pixel(im, math.ceil(x), math.ceil(y), c[0])
    g_btm_q2 = process_image.get_pixel(im, math.ceil(x), math.ceil(y), c[1])
    b_btm_q2 = process_image.get_pixel(im, math.ceil(x), math.ceil(y), c[2])
    r2 = (distance_btm * r_top_q2) + (distance_top * r_btm_q2)
    g2 = (distance_btm * g_top_q2) + (distance_top * g_btm_q2)
    b2 = (distance_btm * b_top_q2) + (distance_top * b_btm_q2)

    # Q
    r = (r1*distance_right) + (r2*distance_left)
    g = (g1*distance_right) + (g2*distance_left)
    b = (b1*distance_right) + (b2*distance_left)
    return r, g, b


def bilinear_resize(im, w: int, h: int) -> np.array:
    width_old, height_old, c = im['w'], im['h'], im['c']
    new_im = process_image.make_image(w, h, c)
    coefficient_matrix = np.array([[-0.5, 1], [w - 0.5, 1]])
    constants_matrix = np.array([-0.5, width_old - 0.5])
    a, b = np.linalg.solve(coefficient_matrix, constants_matrix)
    for y in range(h):
        for x in range(w):
            old_x = a * x - b
            old_y = a * y - b
            q = bilinear_interpolate(im, old_x, old_y, [0, 1, 2])
            new_im['data'][y, x][0] = q[0]
            new_im['data'][y, x][1] = q[1]
            new_im['data'][y, x][2] = q[2]
    return new_im
