from src import image_util, process_image, args

# 1. Getting and setting pixels
im_1 = image_util.load_image("data/dog.jpg")
total_num = 0
for row in range(im_1['h']):
    for col in range(im_1['w']):
        process_image.set_pixel(im_1, col, row, 0, 0)
image_util.save_image(im_1, "dog_no_red")


# 2. Copying an existing image.
im_2 = image_util.load_image('data/dog.jpg')
im_2_copy = process_image.copy_image(im_2)
image_util.save_image(im_2_copy, 'copy_dog')

# # 3. Grayscale image
im_3 = image_util.load_image("data/colorbar.png")
graybar = process_image.rgb_to_grayscale(im_3)
image_util.save_image(graybar, "graybar")


# 4. Shift LoadImage
im_4 = image_util.load_image("data/dog.jpg")
process_image.shift_image(im_4, 0, .4)
process_image.shift_image(im_4, 1, .4)
process_image.shift_image(im_4, 2, .4)
image_util.save_image(im_4, "overflow")


# 6-7. Colorspace and saturation
im_6 = image_util.load_image("data/dog.jpg")
process_image.rgb_to_hsv(im_6)
process_image.clamp_image(im_6)
process_image.shift_image(im_6, 1, .2)
process_image.hsv_to_rgb(im_6)
image_util.save_image(im_6, "dog_saturated")

# 8 Scale the saturation of an image.
im_7 = image_util.load_image('data/dog.jpg')
process_image.rgb_to_hsv(im_7)
process_image.scale_image(im_7, 2)
process_image.clamp_image(im_7)
process_image.hsv_to_rgb(im_7)
image_util.save_image(im_7, 'dog_scale_saturated')
