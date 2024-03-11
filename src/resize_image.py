import math
import numpy as np
from src import process_image


def nn_interpolate(im, x: float, y: float, c: list[int]):
    round_up_x = math.ceil(x)
    round_up_y = math.ceil(y)
    r = process_image.get_pixel(im, round_up_x, round_up_y, c[0])
    g = process_image.get_pixel(im, round_up_x, round_up_y, c[1])
    b = process_image.get_pixel(im, round_up_x, round_up_y, c[2])
    return r, g, b


def nn_resize(im, w: int, h: int):
    width_old, height_old, c = im['w'], im['h'], im['c']
    new_im = process_image.make_image(w, h, c)
    coefficient_matrix = np.array([[-0.5, 1], [w-0.5, 1]])
    constants_matrix = np.array([-0.5, width_old-0.5])
    a, b = np.linalg.solve(coefficient_matrix, constants_matrix)
    for y in range(new_im['h']):
        for x in range(new_im['w']):
            x_mapped = a * x - b
            y_mapped = a * y - b
            r, g, b = nn_interpolate(im, x_mapped, y_mapped, [0, 1, 2])
            process_image.set_pixel(new_im, x, y, 0, r)
            process_image.set_pixel(new_im, x, y, 1, g)
            process_image.set_pixel(new_im, x, y, 2, b)
    return new_im


def bilinear_interpolate(im, x: float, y: float, c: list[int]):
    dist_l = x % 1  # To get decimal
    dist_r = 1 - dist_l
    dist_top = y % 1  # To get decimal
    dist_btm = 1 - dist_top

    # Calculating q1
    r1 = (dist_btm * process_image.get_pixel(im, math.floor(x), math.floor(y), 0)) + (
            dist_top * process_image.get_pixel(im, math.floor(x), math.ceil(y), 0))
    g1 = (dist_btm * process_image.get_pixel(im, math.floor(x), math.floor(y), 1)) + (
            dist_top * process_image.get_pixel(im, math.floor(x), math.ceil(y), 1))
    b1 = (dist_btm * process_image.get_pixel(im, math.floor(x), math.floor(y), 2)) + (
            dist_top * process_image.get_pixel(im, math.floor(x), math.ceil(y), 2))

    # Calculating q2
    r2 = (dist_btm * process_image.get_pixel(im, math.ceil(x), math.floor(y), 0)) + (
            dist_top * process_image.get_pixel(im, math.ceil(x), math.ceil(y), 0))
    g2 = (dist_btm * process_image.get_pixel(im, math.ceil(x), math.floor(y), 1)) + (
            dist_top * process_image.get_pixel(im, math.ceil(x), math.ceil(y), 1))
    b2 = (dist_btm * process_image.get_pixel(im, math.ceil(x), math.floor(y), 2)) + (
            dist_top * process_image.get_pixel(im, math.ceil(x), math.ceil(y), 2))

    # Calculating q
    r = (dist_r * r1) + (dist_l * r2)
    g = (dist_r * g1) + (dist_l * g2)
    b = (dist_r * b1) + (dist_l * b2)
    return r, g, b


def bilinear_resize(im, w: int, h: int):
    width_old, height_old, c = im['w'], im['h'], im['c']
    new_im = process_image.make_image(w, h, c)
    coefficient_matrix = np.array([[-0.5, 1], [w-0.5, 1]])
    constants_matrix = np.array([-0.5, width_old-0.5])
    a, b = np.linalg.solve(coefficient_matrix, constants_matrix)
    for y in range(new_im['h']):
        for x in range(new_im['w']):
            x_mapped = a * x - b
            y_mapped = a * y - b
            r, g, b = bilinear_interpolate(im, x_mapped, y_mapped, [0, 1, 2])
            process_image.set_pixel(new_im, x, y, 0, r)
            process_image.set_pixel(new_im, x, y, 1, g)
            process_image.set_pixel(new_im, x, y, 2, b)
    return new_im
