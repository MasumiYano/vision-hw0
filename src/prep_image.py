import numpy as np
from PIL import Image
import os


def make_empty_image(w, h, c):
    return {'data': np.zeros((h, w, c), dtype=np.float32), 'h': h, 'w': w, 'c': c}


def make_image(w, h, c):
    image = make_empty_image(w, h, c)
    image['data'] = np.zeros((h, w, c), dtype=np.float32)
    return image


def save_image_stb(im, name):
    buff = f"created_data/{name}.jpg"
    data = np.clip(im['data'] * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(data)
    img.save(buff)
    print(f"Image saved to {buff}")


def save_image(im, name):
    save_image_stb(im, name)


def load_image_stb(filename, channels=0):
    img = Image.open(filename)
    if channels:
        if channels == 1:
            img = img.convert('L')
        elif channels == 3:
            img = img.convert('RGB')
        else:  # channels == 4 for RGBA
            img = img.convert('RGBA')
    else:
        img = img.convert('RGB')
    w, h = img.size
    c = len(img.getbands())
    data = np.asarray(img, dtype=np.float32) / 255.0
    if channels and c == 4 and channels != 4:  # Removing alpha channel if not needed
        data = data[:, :, :3]
        c = 3
    return {'data': data, 'h': h, 'w': w, 'c': c}


def load_image(filename):
    return load_image_stb(filename, 0)


def free_image(im):
    del im['data']
