from src import image_util, resize_image, filter_image, process_image

################################ IMAGE RESIZING ################################
# Enlarge
im_dog_sml = image_util.load_image('data/dogsmall.jpg')
im_width = im_dog_sml['w']
im_height = im_dog_sml['h']
a = resize_image.bilinear_resize(im_dog_sml, im_width*4, im_height*4)
image_util.save_image(a, 'dog4x-bl')
b = resize_image.nn_resize(im_dog_sml, im_width*4, im_height*4)
image_util.save_image(b, 'dog4x-nn')

# Shrink
im_dog = image_util.load_image('data/dog.jpg')
im_width = im_dog['w']
im_height = im_dog['h']
c = resize_image.bilinear_resize(im_dog, im_width // 7, im_height // 7)
image_util.save_image(c, 'dog7th-bl')

################################ IMAGE FILTERING WITH CONVOLUTIONS ################################
im = image_util.load_image('data/dog.jpg')
f = filter_image.make_box_filter(7)
blur = filter_image.convolve_image(im, f, 1)
image_util.save_image(blur, "dog-box7")

thumb = resize_image.nn_resize(blur, blur['w']//7, blur['h']//7)
image_util.save_image(thumb, 'dog_thumb')

highpass = filter_image.make_highpass_filter()
highpass_img = filter_image.convolve_image(im, highpass, 0)
image_util.save_image(highpass_img, "highpass_dog")

sharpen = filter_image.make_sharpen_filter()
print(f"Image channel: {im['c']}")
print(f"Filter channel: {sharpen['c']}")
sharpen_img = filter_image.convolve_image(im, sharpen, 1)
image_util.save_image(sharpen_img, 'sharpen_dog')

emboss = filter_image.make_emboss_filter()
emboss_img = filter_image.convolve_image(im, emboss, 1)
image_util.save_image(emboss_img, 'emboss_dog')

gaussian = filter_image.make_gaussian_filter(2)
blur_gaussian = filter_image.convolve_image(im, gaussian, 1)
image_util.save_image(blur_gaussian, 'dog-gaussian2')

f = filter_image.make_gaussian_filter(2)
lfreq = filter_image.convolve_image(im, f, 1)
hfreq = filter_image.sub_image(im, lfreq)
reconstruct = filter_image.add_image(lfreq, hfreq)
image_util.save_image(lfreq, "low-frequency")
image_util.save_image(hfreq, "high-frequency")
image_util.save_image(reconstruct, "reconstruct")
