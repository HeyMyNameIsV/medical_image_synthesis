import os
import tensorflow as tf
from tensorflow import keras
from keras import utils
import numpy as np
from utils.image_utils import load_images_from_folder, preprocess_images

def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = utils.load_img(img_path, color_mode='grayscale', target_size=target_size)
            img_array = utils.img_to_array(img)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images)

if __name__ == "__main__":
    brain_tumor_images = load_images_from_folder('data/brain_tumor')
    non_brain_tumor_images = load_images_from_folder('data/non_brain_tumor')

    brain_tumor_images = preprocess_images(brain_tumor_images)  # Normalize images to [0, 1]
    non_brain_tumor_images = preprocess_images(non_brain_tumor_images)  # Normalize images to [0, 1]

    np.save('data/brain_tumor_images.npy', brain_tumor_images)
    np.save('data/non_brain_tumor_images.npy', non_brain_tumor_images)