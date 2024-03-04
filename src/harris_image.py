from src import process_image
from src import filter_image


def mark_spot(im, x, y):
    for i in range(-9, 10):
        if 0 <= x + i < im['w'] and 0 <= y < im['h']:
            process_image.set_pixel(im, x + i, y, 0, 1)
            process_image.set_pixel(im, x + i, y, 1, 0)
            process_image.set_pixel(im, x + i, y, 2, 1)
        if 0 <= x < im['w'] and 0 <= y + i < im['h']:
            process_image.set_pixel(im, x, y + i, 0, 1)
            process_image.set_pixel(im, x, y + i, 1, 0)
            process_image.set_pixel(im, x, y + i, 2, 1)


def mark_corners(im, descriptors):
    for descriptor in descriptors:
        mark_spot(im, descriptor['x'], descriptor['y'])


def describe_index(original_im, x, y, half_window_size=5):
    descriptor = {'x': x, 'y': y, 'data': []}
    for c in range(original_im['c']):
        central_value = process_image.get_pixel(original_im, x, y, c)
        for dy in range(-half_window_size, half_window_size + 1):
            for dx in range(-half_window_size, half_window_size + 1):
                pixel_value = process_image.get_pixel(original_im, x + dx, y + dy, c)
                descriptor['data'].append(central_value - pixel_value)
    return descriptor


# Calculate the structure matrix of an image.
def structure_matrix(im, sigma):
    # IxIx on channel 0
    # IyIy on channel 1
    # IxIy on channel 2
    S = process_image.make_image(im['w'], im['h'], 3)
    sobel_gx = filter_image.make_gx_filter()
    sobel_gy = filter_image.make_gy_filter()
    gaussian = filter_image.make_gaussian_filter(sigma)
    Ix = filter_image.convolve_image(im, sobel_gx, 1)
    Iy = filter_image.convolve_image(im, sobel_gy, 1)
    for y in range(S['h']):
        for x in range(S['w']):
            Ix_pixel = process_image.get_pixel(Ix, x, y, 0)
            Iy_pixel = process_image.get_pixel(Iy, x, y, 0)
            IxIx = Ix_pixel * Ix_pixel
            IyIy = Iy_pixel * Iy_pixel
            IxIy = Ix_pixel * Iy_pixel
            process_image.set_pixel(S, x, y, 0, IxIx)
            process_image.set_pixel(S, x, y, 1, IyIy)
            process_image.set_pixel(S, x, y, 2, IxIy)
    S_smoothed = filter_image.convolve_image(S, gaussian, 1)
    return S_smoothed


# Estimate the cornerness of each pixel given a structure matrix S.
def cornerness_response(S):
    R = process_image.make_image(S['w'], S['h'], 1)
    alpha = 0.06
    # We'll use formulation det(S) - alpha * trace(S)^2, alpha = .06.
    for y in range(S['h']):
        for x in range(S['w']):
            IxIx = process_image.get_pixel(S, x, y, 0)
            IyIy = process_image.get_pixel(S, x, y, 1)
            IxIy = process_image.get_pixel(S, x, y, 2)
            value = (IxIx * IyIy - (IxIy ** 2)) - alpha * (IxIx + IyIy) ** 2
            process_image.set_pixel(R, x, y, 0, value)
    return R


# Perform non-max suppression on an image of feature responses.
def nms_image(im, w):
    r = process_image.copy_image(im)
    for y in range(r['h']):
        for x in range(r['w']):
            current_pixel = process_image.get_pixel(im, x, y, 0)
            is_biggest = True
            for sub_y in range(y - w, y + w + 1):
                for sub_x in range(x - w, x + w + 1):
                    sub_pixel = process_image.get_pixel(im, sub_x, sub_y, 0)
                    if current_pixel < sub_pixel:
                        is_biggest = False
            if not is_biggest:
                process_image.set_pixel(r, x, y, 0, 0)
    return r


def harris_corner_detector(im, sigma, thresh, nms):
    S = structure_matrix(im, sigma)
    R = cornerness_response(S)
    Rnms = nms_image(R, nms)
    count = 0
    thresh = min(200, thresh * 4)
    for y in range(Rnms['h']):
        for x in range(Rnms['w']):
            pixel_value = process_image.get_pixel(Rnms, x, y, 0)
            if pixel_value * 255 < thresh:
                process_image.set_pixel(Rnms, x, y, 0, 0)
            else:
                count += 1
    n = count
    descriptors = []
    for y in range(Rnms['h']):
        for x in range(Rnms['w']):
            pixel_value = process_image.get_pixel(Rnms, x, y, 0)
            if pixel_value > 0:
                descriptor = describe_index(im, x, y)
                descriptors.append(descriptor)
    return descriptors


# Find and draw corners on an image.
def detect_and_draw_corners(im, sigma, thresh, nms):
    descriptors = harris_corner_detector(im, sigma, thresh, nms)
    mark_corners(im, descriptors)
