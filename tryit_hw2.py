from src import image_util, harris_image, panorama_image

# im = image_util.load_image("data/dog.jpg")
# harris_image.detect_and_draw_corners(im, 2, 50, 3)
# image_util.save_image(im, "corners_dog")

a = image_util.load_image("data/Rainier1.png")
b = image_util.load_image("data/Rainier2.jpg")
m = panorama_image.find_and_draw_matches(a, b, 2, 50, 3)
image_util.save_image(m, "matches")
