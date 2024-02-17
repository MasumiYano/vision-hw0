from src import prep_image, process_image, args

# # 1. Getting and setting pixels
# im_1 = prep_image.load_image("data/dog.jpg")
# total_num = 0
# for row in range(im_1['h']):
#     for col in range(im_1['w']):
#         process_image.set_pixel(im_1, col, row, 0, 0)
# prep_image.save_image(im_1, "dog_no_red")
#
#
# # 2. Copying an existing image.
# im_2 = prep_image.load_image('data/dog.jpg')
# im_2_copy = process_image.copy_image(im_2)
# prep_image.save_image(im_2_copy, 'copy_dog')
#
# # 3. Grayscale image
# im_3 = prep_image.load_image("data/colorbar.png")
# graybar = process_image.rgb_to_grayscale(im_3)
# prep_image.save_image(graybar, "graybar")


# 4. Shift LoadImage
im_4 = prep_image.load_image("data/dog.jpg")
process_image.shift_image(im_4, 0, .4)
process_image.shift_image(im_4, 1, .4)
process_image.shift_image(im_4, 2, .4)
prep_image.save_image(im_4, "overflow")

# # 5. Clamp LoadImage
# clamp_image(im_1)
# save_image(im_1, "doglight_fixed")
#
# # 6-7. Colorspace and saturation
# im_1 = load_image("data/dog.jpg")
# rgb_to_hsv(im_1)
# shift_image(im_1, 1, .2)
# clamp_image(im_1)
# hsv_to_rgb(im_1)
# save_image(im_1, "dog_saturated")
