import math
import numpy as np
from numpy.typing import NDArray
from src import process_image, resize_image


def l1_normalize(image: dict) -> None:
    data: NDArray = image['data']
    num_channels: int = image['c']

    # Normalize differently based on the number of channels
    if num_channels == 1:
        # For single-channel images
        norm_factor = np.sum(np.abs(data))
        if norm_factor != 0:
            data /= norm_factor
    else:
        # For multi-channel images
        norm_factor = np.sum(np.abs(data), axis=(0, 1), keepdims=True)
        # Avoid division by zero
        norm_factor[norm_factor == 0] = 1
        data /= norm_factor


def make_box_filter(w: int) -> dict:
    box_filter = process_image.make_image(w, w, 1)
    normal_factor = 1 / (w ** 2)
    for y in range(box_filter['h']):
        for x in range(box_filter['w']):
            process_image.set_pixel(box_filter, x, y, 0, normal_factor)
    l1_normalize(box_filter)
    return box_filter


def convolve_image(im: dict, inc_filter: dict, preserve: int) -> dict:
    assert (inc_filter['c'] == im['c']) or (
            inc_filter['c'] == 1), "Filter must have the same number of channels of image or be single-channel."

    kernel_center_x: int = inc_filter['w'] // 2
    kernel_center_y: int = inc_filter['h'] // 2

    output_c: int = im['c'] if preserve else 1

    new_img = process_image.make_image(im['w'], im['h'], output_c)

    if inc_filter['c'] == im['c']:
        normal_conv(im, new_img, inc_filter, kernel_center_y, kernel_center_x)
    elif preserve == 1 and inc_filter['c'] > 1:
        preserve_conv(im, new_img, inc_filter, kernel_center_y, kernel_center_x)
    elif inc_filter['c'] == 1 and im['c'] >= 1:
        if preserve:
            preserve_conv(im, new_img, inc_filter, kernel_center_y, kernel_center_x)
        else:
            normal_conv(im, new_img, inc_filter, kernel_center_y, kernel_center_x)
    return new_img


def normal_conv(im: dict, new_img: dict, inc_filter: dict, kernel_y: int, kernel_x: int) -> None:
    for y in range(im['h']):
        for x in range(im['w']):
            q: float = 0
            for y_filter in range(inc_filter['h']):
                for x_filter in range(inc_filter['w']):
                    image_y: int = y + y_filter - kernel_y
                    image_x: int = x + x_filter - kernel_x
                    for c in range(im['c']):
                        value_img: float = process_image.get_pixel(im, image_x, image_y, c)
                        value_kernel: float = process_image.get_pixel(inc_filter, x_filter, y_filter, c)
                        q += value_img * value_kernel
            process_image.set_pixel(new_img, x, y, 0, q)


def preserve_conv(im: dict, new_img: dict, inc_filter: dict, kernel_y: int, kernel_x: int) -> None:
    for y in range(im['h']):
        for x in range(im['w']):
            for c in range(im['c']):
                q: float = 0
                for y_filter in range(inc_filter['h']):
                    for x_filter in range(inc_filter['w']):
                        image_y: int = y + y_filter - kernel_y
                        image_x: int = x + x_filter - kernel_x
                        value_img: float = process_image.get_pixel(im, image_x, image_y, c)
                        value_kernel: float = process_image.get_pixel(inc_filter, x_filter, y_filter, c)
                        q += value_img * value_kernel
                process_image.set_pixel(new_img, x, y, c, q)


def make_highpass_filter() -> dict:
    highpass_filter: dict = process_image.make_image(3, 3, 1)
    highpass_filter['data'] = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    return highpass_filter


def make_sharpen_filter() -> dict:
    sharpen_filter: dict = process_image.make_image(3, 3, 1)
    sharpen_filter['data'] = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return sharpen_filter


def make_emboss_filter() -> dict:
    emboss_filter: dict = process_image.make_image(3, 3, 1)
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


def make_gaussian_filter(sigma: int) -> dict:
    filter_size: int = int(6 * sigma) if int(6 * sigma) % 2 == 1 else int(6 * sigma) + 1
    gaussian_filter: dict = process_image.make_image(filter_size, filter_size, 1)
    center: int = filter_size // 2
    for y in range(gaussian_filter['h']):
        for x in range(gaussian_filter['w']):
            x_offset: int = x - center
            y_offset: int = y - center
            gaussian_value: float = (1 / (2 * math.pi * (sigma ** 2))) * math.exp(
                -((x_offset ** 2 + y_offset ** 2) / (2 * (sigma ** 2))))
            process_image.set_pixel(gaussian_filter, x, y, 0, gaussian_value)
    l1_normalize(gaussian_filter)
    return gaussian_filter


def add_image(im_a: dict, im_b: dict) -> dict:
    assert im_a['c'] == im_b['c']
    _check_size(im_a, im_b)
    new_img: dict = process_image.make_image(im_a['w'], im_a['h'], im_a['c'])
    for y in range(im_a['h']):
        for x in range(im_a['w']):
            if im_a['c'] != 1 and im_b['c'] != 1:
                r_val: float = process_image.get_pixel(im_a, x, y, 0) + process_image.get_pixel(im_b, x, y, 0)
                g_val: float = process_image.get_pixel(im_a, x, y, 1) + process_image.get_pixel(im_b, x, y, 1)
                b_val: float = process_image.get_pixel(im_a, x, y, 2) + process_image.get_pixel(im_b, x, y, 2)
                process_image.set_pixel(new_img, x, y, 0, r_val)
                process_image.set_pixel(new_img, x, y, 1, g_val)
                process_image.set_pixel(new_img, x, y, 2, b_val)
            else:
                val: float = process_image.get_pixel(im_a, x, y, 0) + process_image.get_pixel(im_b, x, y, 0)
                process_image.set_pixel(new_img, x, y, 0, val)
    return new_img


def sub_image(im_a: dict, im_b: dict) -> dict:
    assert im_a['c'] == im_b['c']
    _check_size(im_a, im_b)
    new_img: dict = process_image.make_image(im_a['w'], im_a['h'], im_a['c'])
    for y in range(im_a['h']):
        for x in range(im_a['w']):
            if im_a['c'] != 1 and im_b['c'] != 1:
                r_val: float = process_image.get_pixel(im_a, x, y, 0) - process_image.get_pixel(im_b, x, y, 0)
                g_val: float = process_image.get_pixel(im_a, x, y, 1) - process_image.get_pixel(im_b, x, y, 1)
                b_val: float = process_image.get_pixel(im_a, x, y, 2) - process_image.get_pixel(im_b, x, y, 2)
                process_image.set_pixel(new_img, x, y, 0, r_val)
                process_image.set_pixel(new_img, x, y, 1, g_val)
                process_image.set_pixel(new_img, x, y, 2, b_val)
            else:
                val: float = process_image.get_pixel(im_a, x, y, 0) - process_image.get_pixel(im_b, x, y, 0)
                process_image.set_pixel(new_img, x, y, 0, val)
    return new_img


def _check_size(im_a: dict, im_b: dict) -> None:
    if im_a['w'] < im_b['w']:
        resize_image.bilinear_resize(im_a, im_b['w'], im_b['h'])
    elif im_a['w'] > im_b['w']:
        resize_image.bilinear_resize(im_b, im_a['w'], im_a['h'])
    elif im_a['h'] < im_b['h']:
        resize_image.bilinear_resize(im_a, im_b['w'], im_b['h'])
    elif im_a['h'] > im_b['h']:
        resize_image.bilinear_resize(im_b, im_a['w'], im_b['h'])


def make_gy_filter() -> dict:
    gx_filter: dict = process_image.make_image(3, 3, 1)
    gx_filter['data'] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return gx_filter


def make_gx_filter() -> dict:
    gy_filter: dict = process_image.make_image(3, 3, 1)
    gy_filter['data'] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return gy_filter


def feature_normalize(im: dict) -> None:
    data: NDArray[float] = im['data']
    channels: int = im['c']

    if channels == 1:
        # For single-channel images
        min_val: float = np.min(data)
        max_val: float = np.max(data)
        range_val: float = max_val - min_val

        if range_val > 0:
            im['data'] = (data - min_val) / range_val
        else:
            im['data'] = np.zeros(data.shape, dtype=data.dtype)
    else:
        # For multi-channel images, process each channel independently
        for i in range(channels):
            min_val: float = np.min(data[..., i])
            max_val: float = np.max(data[..., i])
            range_val: float = max_val - min_val

            if range_val > 0:
                data[..., i] = (data[..., i] - min_val) / range_val
            else:
                data[..., i] = np.zeros(data[..., i].shape, dtype=data.dtype)


def sobel_image(im: dict) -> list:
    gradient_magnitude = process_image.make_image(im['w'], im['h'], 1)
    gradient_direction = process_image.make_image(im['w'], im['h'], 1)
    gx: dict = make_gx_filter()
    gy: dict = make_gy_filter()
    gx_img: dict = convolve_image(im, gx, 0)
    gy_img: dict = convolve_image(im, gy, 0)
    for y in range(im['h']):
        for x in range(im['w']):
            gx_val: float = process_image.get_pixel(gx_img, x, y, 0)
            gy_val: float = process_image.get_pixel(gy_img, x, y, 0)
            mag_val: float = math.sqrt((gx_val ** 2) + (gy_val ** 2))
            process_image.set_pixel(gradient_magnitude, x, y, 0, mag_val)

            dir_val: float = math.atan2(gy_val, gx_val)
            process_image.set_pixel(gradient_direction, x, y, 0, dir_val)
    return [gradient_magnitude, gradient_direction]
