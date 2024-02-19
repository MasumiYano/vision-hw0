from src import image_util, resize_image, args

im_dog_sml = image_util.load_image('data/dogsmall.jpg')
im_width = im_dog_sml['w']
im_height = im_dog_sml['h']
a = resize_image.bilinear_resize(im_dog_sml, im_width*4, im_height*4)
image_util.save_image(a, 'dog4x-nn')

