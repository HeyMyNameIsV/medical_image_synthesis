import os
import numpy as np
from tensorflow import keras
from keras import utils

def load_images_from_folder(folder, target_size=(128, 128)):
    """
    Load images from a given folder and preprocess them.

    Args:
    - folder (str): The path to the folder containing images.
    - target_size (tuple): The target size of the images (width, height).

    Returns:
    - numpy.ndarray: Array of preprocessed images.
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            try:
                img = utils.load_img(img_path, color_mode='grayscale', target_size=target_size)
                img_array = utils.img_to_array(img)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images)

def preprocess_images(images):
    """
    Normalize image pixel values to the range [0, 1].

    Args:
    - images (numpy.ndarray): Array of images.

    Returns:
    - numpy.ndarray: Array of normalized images.
    """
    return images / 255.0