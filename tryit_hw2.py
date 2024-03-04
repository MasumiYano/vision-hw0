from src import image_util, harris_image

im = image_util.load_image("data/dog.jpg")
harris_image.detect_and_draw_corners(im, 2, 50, 3)
image_util.save_image(im, "corners_dog")
