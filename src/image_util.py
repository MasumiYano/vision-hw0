import numpy as np
from PIL import Image


def make_empty_image(w: int, h: int, c: int) -> dict:
    return {'data': np.zeros((h, w, c), dtype=np.float32), 'h': h, 'w': w, 'c': c}


def make_image(w: int, h: int, c: int) -> dict:
    image = make_empty_image(w, h, c)
    image['data'] = np.zeros((h, w, c), dtype=np.float32)
    return image


def save_image_stb(im: dict, name: str) -> None:
    buff = f"created_data/{name}.jpg"
    data = np.clip(im['data'] * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(data)
    img.save(buff)
    print(f"Image saved to {buff}")


def save_image(im: dict, name: str) -> None:
    save_image_stb(im, name)


def load_image_stb(filename: str, channels=0) -> dict:
    img = Image.open(filename)
    if channels:
        if channels == 1:
            img = img.convert('L')
        elif channels == 3:
            img = img.convert('RGB')
            print(img)
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
    return {'data': data, 'w': w, 'h': h, 'c': c}


def load_image(filename: str) -> dict:
    return load_image_stb(filename, 0)
