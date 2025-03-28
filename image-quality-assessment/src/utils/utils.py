
import os
import json
import tensorflow as tf
import numpy as np


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, target_file):
    with open(target_file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def random_crop(img, crop_dims):
    h, w = img.shape[0], img.shape[1]
    ch, cw = crop_dims[0], crop_dims[1]
    assert h >= ch, 'image height is less than crop height'
    assert w >= cw, 'image width is less than crop width'
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y:(y+ch), x:(x+cw), :]


def random_horizontal_flip(img):
    assert len(img.shape) == 3, 'input tensor must have 3 dimensions (height, width, channels)'
    assert img.shape[2] == 3, 'image not in channels last format'
    if np.random.random() < 0.5:
        img = img.swapaxes(1, 0)
        img = img[::-1, ...]
        img = img.swapaxes(0, 1)
    return img

def load_image(img_file, target_size=None):
    try:
        # Read image file
        img = tf.io.read_file(img_file)

        # Decode image
        img = tf.image.decode_image(img, channels=3)

        # Resize image if target_size is provided
        if target_size:
            img = tf.image.resize(img, target_size)

        # Convert to numpy array
        return np.asarray(img)

    except Exception as e:
        print(f"Error loading image: {img_file}. Exception: {e}")
        return None

def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist*np.arange(1, 11)).sum()


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def spread_scores(labels, factor=5):
    """
    Applies a power transformation to spread scores.
    Lower scores become lower, higher scores become higher.
    """
    rescaled = np.power(labels, factor)
    return rescaled / np.sum(rescaled)