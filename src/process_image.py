import cv2
import numpy as np


def get_pixel(im: dict, x: int, y: int, c: int) -> float | int:
    height, width = im['h'], im['w']
    select_x: int = max(0, min(x, width - 1))
    select_y: int = max(0, min(y, height - 1))
    if im['c'] == 1:
        return im['data'][select_y, select_x]
    else:
        return im['data'][select_y, select_x][c]


def set_pixel(im: dict, x: int, y: int, c: int, v: float | int) -> None:
    height, width = im['h'], im['w']
    select_x: int = int(max(0, min(x, width - 1)))
    select_y: int = int(max(0, min(y, height - 1)))
    if im['c'] == 1:
        im['data'][select_y, select_x] = v
    else:
        im['data'][select_y, select_x][c] = v


def make_image(width: int, height: int, channel: int) -> dict:
    if channel == 1:
        return {'data': np.zeros((height, width), dtype=np.float32), 'w': width, 'h': height, 'c': channel}
    else:
        return {'data': np.zeros((height, width, channel), dtype=np.float32), 'w': width, 'h': height, 'c': channel}


def copy_image(im: dict) -> dict:
    pixel_arr: list = [pixel for pixel in im['data']]
    copy: dict = make_image(im['w'], im['h'], im['c'])
    copy['data'] = np.array(pixel_arr)
    return copy


def rgb_to_grayscale(im: dict) -> dict:
    assert im['c'] == 3
    gray: dict = make_image(im['w'], im['h'], 1)
    for y in range(im['h']):
        for x in range(im['w']):
            R: float | int = im['data'][y, x, 0]
            G: float | int = im['data'][y, x, 1]
            B: float | int = im['data'][y, x, 2]
            luma: float = (0.229 * R) + (0.587 * G) + (0.114 * B)
            gray['data'][y, x] = luma
    return gray


def shift_image(im: dict, channel: int, value: float | int) -> None:
    factor: float | int = 1 + value
    for y in range(im['h']):
        for x in range(im['w']):
            value: float | int = get_pixel(im, x, y, channel)
            im['data'][y, x][channel] = value * factor


def clamp_image(im: dict) -> None:
    for y in range(im['h']):
        for x in range(im['w']):
            r_value: float | int = get_pixel(im, x, y, 0)
            g_value: float | int = get_pixel(im, x, y, 1)
            b_value: float | int = get_pixel(im, x, y, 2)
            fixed_r_value: float | int = max(0, min(r_value, 1))
            fixed_g_value: float | int = max(0, min(g_value, 1))
            fixed_b_value: float | int = max(0, min(b_value, 1))
            set_pixel(im, x, y, 0, fixed_r_value)
            set_pixel(im, x, y, 1, fixed_g_value)
            set_pixel(im, x, y, 2, fixed_b_value)


def rgb_to_hsv(im: dict) -> None:
    for y in range(im['h']):
        for x in range(im['w']):
            r: float | int = get_pixel(im, x, y, 0)
            g: float | int = get_pixel(im, x, y, 1)
            b: float | int = get_pixel(im, x, y, 2)
            V: float | int = max(r, g, b)
            m: float | int = min(r, g, b)
            C: float | int = V - m
            if r == 0.0 and g == 0.0 and b == 0.0:
                S = 0
            else:
                S = C / V
            h_prime: float | int = calculate_h_prime(C, V, r, g, b)
            if C == 0:
                h = 0
            else:
                h: float | int = calculate_h(h_prime)
            set_pixel(im, x, y, 0, h)
            set_pixel(im, x, y, 1, S)
            set_pixel(im, x, y, 2, V)


def calculate_h_prime(C: float | int, V: float | int, r: float | int, g: float | int, b: float | int) -> float:
    if C == 0:
        return 0
    elif V == r:
        return (g - b) / C
    elif V == g:
        return ((b - r) / C) + 2
    elif V == b:
        return ((r - g) / C) + 4


def calculate_h(h_prime: float | int) -> float | int:
    if h_prime < 0:
        return (h_prime / 6) + 1
    else:
        return h_prime / 6


def hsv_to_rgb(im: dict) -> None:
    for y in range(im['h']):
        for x in range(im['w']):
            H: float | int = get_pixel(im, x, y, 0)
            S: float | int = get_pixel(im, x, y, 1)
            V: float | int = get_pixel(im, x, y, 2)
            C: float | int = V * S
            H_prime: float | int = H * 6
            X: float | int = C * (1 - abs((H_prime % 2) - 1))
            m: float | int = V - C
            r, g, b = get_rgb(H_prime, X, C)
            set_pixel(im, x, y, 0, r + m)
            set_pixel(im, x, y, 1, g + m)
            set_pixel(im, x, y, 2, b + m)


def get_rgb(H_prime: float | int, X: float | int, C: float | int):
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


def scale_image(im: dict, scale: int) -> None:
    for y in range(im['h']):
        for x in range(im['w']):
            s_value: float | int = get_pixel(im, x, y, 1)
            set_pixel(im, x, y, 1, s_value*scale)


def cv2_to_image(cv2_img):
    h, w = cv2_img.shape[:2]
    c = cv2_img.shape[2] if len(cv2_img.shape) == 3 else 1
    if c == 1:
        # For grayscale images
        data = cv2_img.reshape((h, w, c)).astype(np.float32) / 255.0
    else:
        # For BGR images, might need to convert to RGB if that's what your processing expects
        data = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return {'data': data, 'w': w, 'h': h, 'c': c}


def image_to_cv2(im):
    if im['c'] == 1:
        cv2_img = (im['data'].squeeze() * 255).astype(np.uint8)
    else:
        # If your internal format is RGB, convert back to BGR for OpenCV compatibility
        cv2_img = cv2.cvtColor((im['data'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2_img
