from src import image_util, resize_image, args

################################ IMAGE RESIZING ################################
# Enlarge
# im_dog_sml = image_util.load_image('data/dogsmall.jpg')
# im_width = im_dog_sml['w']
# im_height = im_dog_sml['h']
# a = resize_image.bilinear_resize(im_dog_sml, im_width*4, im_height*4)
# image_util.save_image(a, 'dog4x-bl')
# b = resize_image.nn_resize(im_dog_sml, im_width*4, im_height*4)
# image_util.save_image(b, 'dog4x-nn')

# Shrink
# im_dog = image_util.load_image('data/dog.jpg')
# im_width = im_dog['w']
# im_height = im_dog['h']
# c = resize_image.bilinear_resize(im_dog, im_width // 7, im_height // 7)
# image_util.save_image(c, 'dog7th-bl')

################################ IMAGE FILTERING WITH CONVOLUTIONS ################################
