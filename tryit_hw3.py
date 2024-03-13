from src import image_util, flow_image, flow_image_fast

# a = image_util.load_image("data/dog_a.jpg")
# b = image_util.load_image("data/dog_b.jpg")
# flow = flow_image.optical_flow_images(b, a, 15, 8)
# flow = flow_image_fast.optical_flow_images(b, a, 15, 8)
# flow_image_fast.draw_flow(a, flow, 8)
# flow_image.draw_flow(a, flow, 8)
# image_util.save_image(a, "lines")

flow_image_fast.optical_flow_webcam(15, 4, 8)
# flow_image.optical_flow_webcam(15, 4, 8)
