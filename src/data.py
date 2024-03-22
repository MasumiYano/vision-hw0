import numpy as np
from typing import List, Dict
from PIL import Image


# Helper function to read lines from a file
def get_lines(filename: str) -> List[str]:
    with open(filename, 'r') as file:
        lines = [line.rstrip('\n') for line in file]
    return lines


# Load an image and convert to grayscale if necessary
def load_image(path: str, channels=0) -> np.ndarray:
    with Image.open(path) as img:
        if channels:
            if channels == 1:
                img = img.convert('L')
            elif channels == 3:
                img = img.convert('RGB')
            else:
                img = img.convert('RGBA')
        else:
            img = img.convert('RGB')

        img_data = np.asarray(img, dtype=np.float32) / 255.0
    return img_data


# Randomly sample a batch of data
def random_batch(d: Dict[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
    indices = np.random.choice(d['X'].shape[0], n, replace=False)
    return {'X': d['X'][indices], 'y': d['y'][indices]}


# Load classification data
def load_classification_data(images: str, label_file: str, bias: bool) -> Dict[str, np.ndarray]:
    image_list = get_lines(images)
    label_list = get_lines(label_file)
    labels = list(set(label_list))  # Get unique labels
    k = len(labels)

    n = len(image_list)
    cols = 0
    X = []
    y = []
    for path in image_list:
        im = load_image(path)
        if not cols:
            cols = len(im)
            if bias:
                cols += 1
        if bias:
            im = np.append(im, 1)  # Add bias term if necessary
        X.append(im)

        label_vector = np.zeros(k)
        for i, label in enumerate(labels):
            if label in path:
                label_vector[i] = 1
        y.append(label_vector)

    return {'X': np.array(X), 'y': np.array(y)}
