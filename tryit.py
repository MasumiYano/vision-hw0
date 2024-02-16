from src import prep_image, process_image, args

# 1. Getting and setting pixels
im = prep_image.load_image("data/dog.jpg")
print(f"(Height, Width, Channel): {im['data'].shape}")
print(im['c'])
# print(im['data'])
total_num = 0
for row in range(im['h']):
    for col in range(im['w']):
        total_num += 1
        print(im['data'][row, col])
#         process_image.set_pixel(im, col, row, 0, 0)
# prep_image.save_image(im, "dog_no_red")


# # 3. Grayscale image
# im = load_image("data/colorbar.png")
# graybar = rgb_to_grayscale(im)
# save_image(graybar, "graybar")
#
# # 4. Shift LoadImage
# im = load_image("data/dog.jpg")
# shift_image(im, 0, .4)
# shift_image(im, 1, .4)
# shift_image(im, 2, .4)
# save_image(im, "overflow")
#
# # 5. Clamp LoadImage
# clamp_image(im)
# save_image(im, "doglight_fixed")
#
# # 6-7. Colorspace and saturation
# im = load_image("data/dog.jpg")
# rgb_to_hsv(im)
# shift_image(im, 1, .2)
# clamp_image(im)
# hsv_to_rgb(im)
# save_image(im, "dog_saturated")
