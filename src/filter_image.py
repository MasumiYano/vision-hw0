import numpy as np
from process_image import make_image


def l1_normalize(im):
    # TODO: Fill this in
    pass


def make_box_filter(w):
    # TODO: Fill this in
    return make_image(1, 1, 1)


def convolve_image(im, inc_filter, preserve):
    # TODO: Fill this in
    return make_image(1, 1, 1)


def make_highpass_filter():
    # TODO: Fill this in
    return make_image(1, 1, 1)


def make_sharpen_filter():
    # TODO: Fill this in
    return make_image(1, 1, 1)


def make_emboss_filter():
    # TODO: Fill this in
    return make_image(1, 1, 1)


# Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
# Answer: TODO

# Question 2.2.2: Do we have to do any post-processing for the above filter? Which ones and why?
# Answer: TODO

def make_gaussian_filter(sigma):
    # TODO: Fill this in
    return make_image(1, 1, 1)


def add_image(im_a, im_b):
    # TODO: Fill this in
    return make_image(1, 1, 1)


def sub_image(im_a, im_b):
    # TODO FIll this in
    return make_image(1, 1, 1)


def make_gx_filter():
    # TODO Fill this in
    return make_image(1, 1, 1)


def make_gy_filter():
    # TODO Fill this in
    return make_image(1, 1, 1)


def feature_normalize(im):
    # TODO Fill this in
    pass


def sobel_image(im):
    # TODO
    return [None, None]


def colorize_sobel(im):
    # TODO
    return make_image(1, 1, 1)
