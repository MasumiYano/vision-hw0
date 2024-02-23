import math
import numpy as np

from src import process_image, resize_image


def l1_normalize(im):
    if im['c'] != 1:
        total_sum_r, total_sum_g, total_sum_b = 0, 0, 0
        for y in range(im['h']):
            for x in range(im['w']):
                total_sum_r += process_image.get_pixel(im, x, y, 0)
                total_sum_g += process_image.get_pixel(im, x, y, 1)
                total_sum_b += process_image.get_pixel(im, x, y, 2)

        for y in range(im['h']):
            for x in range(im['w']):
                r = process_image.get_pixel(im, x, y, 0)
                process_image.set_pixel(im, x, y, 0, r / total_sum_r)
                g = process_image.get_pixel(im, x, y, 1)
                process_image.set_pixel(im, x, y, 1, g / total_sum_g)
                b = process_image.get_pixel(im, x, y, 2)
                process_image.set_pixel(im, x, y, 2, b / total_sum_b)
    else:
        total_sum = 0
        for y in range(im['h']):
            for x in range(im['w']):
                total_sum += process_image.get_pixel(im, x, y, 0)

        for y in range(im['h']):
            for x in range(im['w']):
                value = process_image.get_pixel(im, x, y, 0) / total_sum
                process_image.set_pixel(im, x, y, 0, value)


def make_box_filter(w):
    box_filter = process_image.make_image(w, w, 1)
    normal_factor = 1 / (w ** 2)
    for y in range(box_filter['h']):
        for x in range(box_filter['w']):
            process_image.set_pixel(box_filter, x, y, 0, normal_factor)
    l1_normalize(box_filter)
    return box_filter


def convolve_image(im, inc_filter, preserve):
    kernel_center_x = inc_filter['w'] // 2
    kernel_center_y = inc_filter['h'] // 2

    output_c = im['c'] if preserve else 1

    new_img = process_image.make_image(im['w'], im['h'], output_c)

    if inc_filter['c'] == im['c']:
        normal_conv(im, new_img, inc_filter, kernel_center_y, kernel_center_x)
    elif preserve == 1 and inc_filter['c'] > 1:
        preserve_conv(im, new_img, inc_filter, kernel_center_y, kernel_center_x)
    elif inc_filter['c'] == 1 and im['c'] >= 1:
        each_conv(im, new_img, inc_filter, kernel_center_y, kernel_center_x, preserve)
    return new_img


def normal_conv(im, new_img, inc_filter, kernel_y, kernel_x):
    for y in range(im['h']):
        for x in range(im['w']):
            q = 0
            for y_filter in range(inc_filter['h']):
                for x_filter in range(inc_filter['w']):
                    image_y = y + y_filter - kernel_y
                    image_x = x + x_filter - kernel_x
                    for c in range(im['c']):
                        value_img = process_image.get_pixel(im, image_x, image_y, c)
                        value_kernel = process_image.get_pixel(inc_filter, x_filter, y_filter, c)
                        q += value_img * value_kernel
            process_image.set_pixel(new_img, x, y, 0, q)


def preserve_conv(im, new_img, inc_filter, kernel_y, kernel_x):
    for y in range(im['h']):
        for x in range(im['w']):
            for c in range(im['c']):
                q = 0
                for y_filter in range(inc_filter['h']):
                    for x_filter in range(inc_filter['w']):
                        image_y = y + y_filter - kernel_y
                        image_x = x + x_filter - kernel_x
                        value_img = process_image.get_pixel(im, image_y, image_x, c)
                        value_kernel = process_image.get_pixel(inc_filter, y_filter, x_filter, c)
                        q += value_img * value_kernel
                process_image.set_pixel(new_img, x, y, c, q)


def each_conv(im, new_img, inc_filter, kernel_y, kernel_x, preserve):
    for y in range(im['h']):
        for x in range(im['w']):
            if preserve:
                for c in range(im['c']):
                    q = 0
                    for y_filter in range(inc_filter['h']):
                        for x_filter in range(inc_filter['w']):
                            image_y = y + y_filter - kernel_y
                            image_x = x + x_filter - kernel_x
                            value_img = process_image.get_pixel(im, image_x, image_y, c)
                            value_kernel = process_image.get_pixel(inc_filter, x_filter, y_filter, 0)
                            q += value_img * value_kernel
                    process_image.set_pixel(new_img, x, y, c, q)
            else:
                q = 0
                for y_filter in range(inc_filter['h']):
                    for x_filter in range(inc_filter['w']):
                        image_y = y + y_filter - kernel_y
                        image_x = x + x_filter - kernel_x
                        sub_img_value = process_image.get_pixel(im, image_x, image_y, 0)
                        sum_sub = sum(sub_img_value)
                        value_kernel = process_image.get_pixel(inc_filter, x_filter, y_filter, 0)
                        q += sum_sub * value_kernel
                process_image.set_pixel(new_img, x, y, 0, q)


def make_highpass_filter():
    highpass_filter = process_image.make_image(3, 3, 1)
    highpass_filter['data'] = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    l1_normalize(highpass_filter)
    return highpass_filter


def make_sharpen_filter():
    sharpen_filter = process_image.make_image(3, 3, 1)
    sharpen_filter['data'] = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return sharpen_filter


def make_emboss_filter():
    emboss_filter = process_image.make_image(3, 3, 1)
    emboss_filter['data'] = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    return emboss_filter


# Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
"""
Answer: 
Highpass filter: Shouldn't be preserved. The main objective of highpass filter is to detect the edges in the given image,
and highlight them. By not preserving the channel information, the single channel image where each pixel reflects the strength 
of the edge is easier to analyze the edge related applications. 

Sharpen filter: Should be preserved. It is important to preserve the channel information to ensure the effect enhanced the clarity 
and contrast of the image color channels.

Emboss filter: Should be preserved. Maintaining the original color information is essential for maintaining the visual coherence of 
the emboss effect. 
"""
# Question 2.2.2: Do we have to do any post-processing for the above filter? Which ones and why?
"""
It depends on the goal of the filtering operation. For analytic purpose I think raw information is necessary, but for aesthetic purposes, 
I think post-processing might be good. 
"""


def make_gaussian_filter(sigma):
    filter_size = int(6 * sigma) if int(6 * sigma) % 2 == 1 else int(6 * sigma) + 1
    gaussian_filter = process_image.make_image(filter_size, filter_size, 1)
    center = filter_size // 2
    for y in range(gaussian_filter['h']):
        for x in range(gaussian_filter['w']):
            x_offset = x - center
            y_offset = y - center
            gaussian_value = (1 / (2 * math.pi * (sigma ** 2))) * math.exp(
                -((x_offset ** 2 + y_offset ** 2) / (2 * (sigma ** 2))))
            process_image.set_pixel(gaussian_filter, x, y, 0, gaussian_value)
    l1_normalize(gaussian_filter)
    return gaussian_filter


def add_image(im_a, im_b):
    assert im_a['c'] == im_b['c']
    _check_size(im_a, im_b)
    new_img = process_image.make_image(im_a['w'], im_a['h'], im_a['c'])
    for y in range(im_a['h']):
        for x in range(im_a['w']):
            if im_a['c'] != 1 and im_b['c'] != 1:
                r_val = process_image.get_pixel(im_a, x, y, 0) + process_image.get_pixel(im_b, x, y, 0)
                g_val = process_image.get_pixel(im_a, x, y, 1) + process_image.get_pixel(im_b, x, y, 1)
                b_val = process_image.get_pixel(im_a, x, y, 2) + process_image.get_pixel(im_b, x, y, 2)
                process_image.set_pixel(new_img, x, y, 0, r_val)
                process_image.set_pixel(new_img, x, y, 1, g_val)
                process_image.set_pixel(new_img, x, y, 2, b_val)
            else:
                val = process_image.get_pixel(im_a, x, y, 0) + process_image.get_pixel(im_b, x, y, 0)
                process_image.set_pixel(new_img, x, y, 0, val)
    return new_img


def sub_image(im_a, im_b):
    assert im_a['c'] == im_b['c']
    _check_size(im_a, im_b)
    new_img = process_image.make_image(im_a['w'], im_a['h'], im_a['c'])
    for y in range(im_a['h']):
        for x in range(im_a['w']):
            if im_a['c'] != 1 and im_b['c'] != 1:
                r_val = process_image.get_pixel(im_a, x, y, 0) - process_image.get_pixel(im_b, x, y, 0)
                g_val = process_image.get_pixel(im_a, x, y, 1) - process_image.get_pixel(im_b, x, y, 1)
                b_val = process_image.get_pixel(im_a, x, y, 2) - process_image.get_pixel(im_b, x, y, 2)
                process_image.set_pixel(new_img, x, y, 0, r_val)
                process_image.set_pixel(new_img, x, y, 1, g_val)
                process_image.set_pixel(new_img, x, y, 2, b_val)
            else:
                val = process_image.get_pixel(im_a, x, y, 0) - process_image.get_pixel(im_b, x, y, 0)
                process_image.set_pixel(new_img, x, y, 0, val)
    return new_img


def _check_size(im_a, im_b):
    if im_a['w'] < im_b['w']:
        resize_image.bilinear_resize(im_a, im_b['w'], im_b['h'])
    elif im_a['w'] > im_b['w']:
        resize_image.bilinear_resize(im_b, im_a['w'], im_a['h'])
    elif im_a['h'] < im_b['h']:
        resize_image.bilinear_resize(im_a, im_b['w'], im_b['h'])
    elif im_a['h'] > im_b['h']:
        resize_image.bilinear_resize(im_b, im_a['w'], im_b['h'])


def make_gx_filter():
    gx_filter = process_image.make_image(3, 3, 1)
    gx_filter['data'] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return gx_filter


def make_gy_filter():
    gy_filter = process_image.make_image(3, 3, 1)
    gy_filter['data'] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return gy_filter


def feature_normalize(im):
    if im['c'] > 1:
        sml_r, sml_g, sml_b = float('inf'), float('inf'), float('inf')
        max_r, max_g, max_b = float('-inf'), float('-inf'), float('-inf')
        for y in range(im['h']):
            for x in range(im['w']):
                r_val = process_image.get_pixel(im, x, y, 0)
                g_val = process_image.get_pixel(im, x, y, 1)
                b_val = process_image.get_pixel(im, x, y, 2)
                sml_r = min(sml_r, r_val)
                sml_g = min(sml_g, g_val)
                sml_b = min(sml_b, b_val)
                max_r = max(max_r, r_val)
                max_g = max(max_g, g_val)
                max_b = max(max_b, b_val)
        range_r = max_r - sml_r
        range_g = max_g - sml_g
        range_b = max_b - sml_b
        if range_r == 0 or range_g == 0 or range_b == 0:
            for y in range(im['h']):
                for x in range(im['w']):
                    process_image.set_pixel(im, x, y, 0, 0)
        else:
            for y in range(im['h']):
                for x in range(im['w']):
                    r_val = process_image.get_pixel(im, x, y, 0)
                    g_val = process_image.get_pixel(im, x, y, 1)
                    b_val = process_image.get_pixel(im, x, y, 2)
                    r_normalize = (r_val - sml_r) / (max_r - sml_r)
                    g_normalize = (g_val - sml_g) / (max_g - sml_g)
                    b_normalize = (b_val - sml_b) / (max_b - sml_b)
                    process_image.set_pixel(im, x, y, 0, r_normalize)
                    process_image.set_pixel(im, x, y, 1, g_normalize)
                    process_image.set_pixel(im, x, y, 2, b_normalize)

    else:
        sml_val = float('inf')
        max_val = float('-inf')
        for y in range(im['h']):
            for x in range(im['w']):
                curr_val = process_image.get_pixel(im, x, y, 0)
                sml_val = min(sml_val, curr_val)
                max_val = max(max_val, curr_val)
        range_ = max_val - sml_val
        if range_ == 0:
            for y in range(im['h']):
                for x in range(im['w']):
                    process_image.set_pixel(im, x, y, 0, 0)
        else:
            for y in range(im['h']):
                for x in range(im['w']):
                    curr_val = process_image.get_pixel(im, x, y, 0)
                    val_normalize = (curr_val - sml_val) / (max_val - sml_val)
                    process_image.set_pixel(im, x, y, 0, val_normalize)


def sobel_image(im):
    gradient_magnitude = process_image.make_image(im['w'], im['h'], 1)
    gradient_direction = process_image.make_image(im['w'], im['h'], 1)
    gx = make_gx_filter()
    gy = make_gy_filter()
    gx_img = convolve_image(im, gx, 0)
    gy_img = convolve_image(im, gy, 0)
    for y in range(im['h']):
        for x in range(im['w']):
            gx_val = process_image.get_pixel(gx_img, x, y, 0)
            gy_val = process_image.get_pixel(gy_img, x, y, 0)
            mag_val = math.sqrt((gx_val ** 2) + (gy_val ** 2))
            process_image.set_pixel(gradient_magnitude, x, y, 0, mag_val)

            dir_val = math.atan2(gy_val, gx_val)
            process_image.set_pixel(gradient_direction, x, y, 0, dir_val)
    return [gradient_magnitude, gradient_direction]


def colorize_sobel(im):
    # TODO
    return
