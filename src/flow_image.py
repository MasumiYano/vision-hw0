from scipy.ndimage import gaussian_filter
import math
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Callable

from src import process_image, matrix, resize_image, filter_image


def draw_line(im: dict, x: float, y: float, dx: float, dy: float) -> None:
    assert im['c'] == 3
    angle: float = 6 * (math.atan2(dy, dx) / math.pi * 2 + 0.5)
    index: int = math.floor(angle)
    f: float = angle - index
    if index == 0:
        r, g, b = 1, f, 0
    elif index == 1:
        r, g, b = 1 - f, 1, 0
    elif index == 2:
        r, g, b = 0, 1, f
    elif index == 3:
        r, g, b = 0, 1 - f, 1
    elif index == 4:
        r, g, b = f, 0, 1
    else:
        r, g, b = 1, 0, 1 - f

    d: float = math.sqrt(dx*dx + dy*dy)
    for i in range(int(d)):
        xi: int = int(x + dx * i / d)
        yi: int = int(y + dy * i / d)
        process_image.set_pixel(im, xi, yi, 0, r)
        process_image.set_pixel(im, xi, yi, 1, g)
        process_image.set_pixel(im, xi, yi, 2, b)


def make_integral_image(im: dict) -> dict:
    integ: dict = process_image.make_image(im['w'], im['h'], im['c'])
    # TODO: fill in the integral image
    for y in range(im['h']):
        for x in range(im['w']):
            for c in range(im['c']):
                curr_val: float = im['data'][x, y, c]
                above: float | int = integ['data'][x, y - 1, c] if y != 0 else 0
                left: float | int = integ['data'][x - 1, y, c] if x != 0 else 0
                top_left: float | int = integ['data'][x - 1, y - 1, c] if x != 0 and y != 0 else 0
                store_val: float | int = curr_val + above + left - top_left
                integ['data'][x, y, c] = store_val
    return integ


def box_filter_image(im: dict, s: int) -> dict:
    S: dict = process_image.make_image(im['w'], im['h'], im['c'])
    integ: dict = make_integral_image(im)
    for y in range(im['h']):
        for x in range(im['w']):
            for c in range(im['c']):
                top_x, top_y = x - s, y - s
                btm_x, btm_y = x + s, y + s
                A: float | int = process_image.get_pixel(integ, top_x - 1, top_y - 1, c)
                B: float | int = process_image.get_pixel(integ, top_x - 1, btm_y, c)
                C: float | int = process_image.get_pixel(integ, btm_x, top_y - 1, c)
                D: float | int = process_image.get_pixel(integ, btm_x, btm_y, c)
                val: float | int = D + A - B - C
                avg_val: float = val / math.pow((2 * s + 1), 2)
                process_image.set_pixel(S, x, y, c, avg_val)
    return S


"""
// Calculate the time-structure matrix of an image pair.
// image im: the input image.
// image prev: the previous image in sequence.
// int s: window size for smoothing.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          3rd channel is IxIy, 4th channel is IxIt, 5th channel is IyIt.
"""


def time_structure_matrix(im: dict, prev: dict, s: int) -> dict:
    S: dict = process_image.make_image(im['w'], im['h'], 5)
    if im['c'] == 3:
        im = process_image.rgb_to_grayscale(im)
        prev = process_image.rgb_to_grayscale(prev)

    # TODO: Calculate gradients, structure components, and smooth them.
    gy: dict = filter_image.make_gy_filter()
    gx: dict = filter_image.make_gx_filter()
    Iy: dict = filter_image.convolve_image(im, gy, 0)
    Ix: dict = filter_image.convolve_image(im, gx, 0)
    It: dict = filter_image.sub_image(im, prev)
    for y in range(im['h']):
        for x in range(im['w']):
            Ix_val: float = process_image.get_pixel(Ix, x, y, 0)
            Iy_val: float = process_image.get_pixel(Iy, x, y, 0)
            It_val: float = process_image.get_pixel(It, x, y, 0)
            process_image.set_pixel(S, x, y, 0, Ix_val * Ix_val)
            process_image.set_pixel(S, x, y, 1, Iy_val * Iy_val)
            process_image.set_pixel(S, x, y, 2, Ix_val * Iy_val)
            process_image.set_pixel(S, x, y, 3, Ix_val * It_val)
            process_image.set_pixel(S, x, y, 4, Iy_val * It_val)

    smoothed_data: NDArray[float] = np.zeros_like(S['data'])
    for c in range(S['c']):
        smoothed_data[:, :, c] = smooth_image(S['data'][:, :, c], s)

    S['data'] = smoothed_data
    return S


def smooth_image(image_data: NDArray, radius: int) -> NDArray:
    return gaussian_filter(image_data, sigma=radius)


def velocity_image(S: dict, stride: int) -> dict:
    v: dict = process_image.make_image(S['w'] // stride, S['h'] // stride, 3)
    M: NDArray = matrix.make_matrix(2, 2)
    L: NDArray = matrix.make_matrix(2, 1)
    for y in range((stride - 1) // 2, S['h'], stride):
        for x in range((stride - 1) // 2, S['w'], stride):
            Ixx: float | int = process_image.get_pixel(S, x, y, 0)
            Iyy: float | int = process_image.get_pixel(S, x, y, 1)
            Ixy: float | int = process_image.get_pixel(S, x, y, 2)
            Ixt: float | int = process_image.get_pixel(S, x, y, 3)
            Iyt: float | int = process_image.get_pixel(S, x, y, 4)

            # TODO: calculate vx and vy using the flow equation
            vx, vy = 0, 0
            M[0, 0] = Ixx
            M[0, 1] = Ixy
            M[1, 0] = Ixy
            M[1, 1] = Iyy
            M = matrix.matrix_invert(M)
            L[0, 0] = -1 * Ixt
            L[1, 0] = -1 * Iyt
            result = matrix.matrix_mult_matrix(M, L)
            vx, vy = float(result[0, 0]), float(result[1, 0])

            process_image.set_pixel(v, x // stride, y // stride, 0, vx)
            process_image.set_pixel(v, x // stride, y // stride, 1, vy)

    return v


def draw_flow(im: dict, v: dict, scale: float) -> None:
    stride = im['w'] // v['w']
    for y in range((stride - 1) // 2, im['h'], stride):
        for x in range((stride - 1) // 2, im['w'], stride):
            dx: float | int = scale * process_image.get_pixel(v, x // stride, y // stride, 0)
            dy: float | int = scale * process_image.get_pixel(v, x // stride, y // stride, 1)
            if math.fabs(dx) > im['w']:
                dx = 0
            if math.fabs(dy) > im['h']:
                dy = 0
            draw_line(im, x, y, dx, dy)


def constrain_image(im: dict, v: float) -> None:
    im['data'] = np.clip(im['data'], -v, v)


def optical_flow_images(im: dict, prev: dict, smooth: int, stride: int) -> dict:
    S: dict = time_structure_matrix(im, prev, smooth)
    v: dict = velocity_image(S, stride)
    constrain_image(v, 6)
    smoothed_data: NDArray = np.zeros_like(v['data'])
    for c in range(v['c']):
        smoothed_data[:, :, c] = smooth_image(v['data'][:, :, c], 2)
    vs: dict = process_image.make_image(v['w'], v['h'], v['c'])
    vs['data'] = smoothed_data
    return vs


def optical_flow_webcam(smooth: int, stride: int, div: int) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        return

    # Convert to the expected dictionary format and resize
    prev = process_image.cv2_to_image(prev_frame)
    prev_c = resize_image.nn_resize(prev, prev['w'] // div, prev['h'] // div)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to the expected dictionary format and resize
        im = process_image.cv2_to_image(frame)
        im_c = resize_image.nn_resize(im, im['w'] // div, im['h'] // div)
        # Compute optical flow
        v = optical_flow_images(im_c, prev_c, smooth, stride)
        # Create a copy of the current frame to draw on
        copy = process_image.copy_image(im)
        # Draw optical flow on the copy
        draw_flow(copy, v, smooth * div)
        # Convert the copy back to an OpenCV-friendly format for display
        display_image = process_image.image_to_cv2(copy)
        cv2.imshow('flow', display_image)
        # Update the previous frame and its resized version
        prev, prev_c = im, im_c

        if cv2.waitKey(5) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

