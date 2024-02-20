from src import process_image


def l1_normalize(im):
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


def make_box_filter(w):
    box_filter = process_image.make_image(w, w, 1)
    normal_factor = 1/(w**2)
    for y in range(box_filter['h']):
        for x in range(box_filter['w']):
            process_image.set_pixel(box_filter, x, y, 0, normal_factor)
    return box_filter


def convolve_image(im, inc_filter, preserve):
    kernel_center_x = inc_filter['w'] // 2
    kernel_center_y = inc_filter['h'] // 2
    if inc_filter['c'] == im['c']:
        new_img = process_image.make_image(im['w'], im['h'], 1)
        for y in range(im['h']):
            for x in range(im['w']):
                q = 0
                for y_filter in range(inc_filter['h']):
                    for x_filter in range(inc_filter['w']):
                        image_y = y + y_filter - kernel_center_y
                        image_x = x + x_filter - kernel_center_x
                        pixel_img = process_image.get_pixel(im, image_x, image_y, 0)
                        pixel_kernel = process_image.get_pixel(inc_filter, x_filter, y_filter, 0)
                        q += pixel_img * pixel_kernel
                process_image.set_pixel(new_img, x, y, 0, q)

    elif preserve == 1:
        new_img = process_image.make_image(im['w'], im['h'], im['c'])
        for y in range(im['h']):
            for x in range(im['w']):
                q_r = 0
                q_g = 0
                q_b = 0
                for y_filter in range(inc_filter['h']):
                    for x_filter in range(inc_filter['w']):
                        image_y = y + y_filter - kernel_center_y
                        image_x = x + x_filter - kernel_center_x
                        r_img = process_image.get_pixel(im, image_x, image_y, 0)
                        g_img = process_image.get_pixel(im, image_x, image_y, 1)
                        b_img = process_image.get_pixel(im, image_x, image_y, 2)
                        pixel_kernel = process_image.get_pixel(inc_filter, x_filter, y_filter, 0)
                        q_r += r_img * pixel_kernel
                        q_g += g_img * pixel_kernel
                        q_b += b_img * pixel_kernel
                process_image.set_pixel(new_img, x, y, 0, q_r)
                process_image.set_pixel(new_img, x, y, 1, q_g)
                process_image.set_pixel(new_img, x, y, 2, q_b)

    else:
        new_img = process_image.make_image(im['w'], im['h'], im['c'])

    return new_img


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
