import numpy as np
import os
from keras import preprocessing

def load_images_from_folder(folder, target_size=(128, 128), label=0):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = preprocessing.load_img(img_path, color_mode='grayscale', target_size=target_size)
            img_array = preprocessing.img_to_array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

def load_data():
    tumor_images, tumor_labels = load_images_from_folder('data/brain_tumor', label=1)
    non_tumor_images, non_tumor_labels = load_images_from_folder('data/non_brain_tumor', label=0)

    images = np.concatenate((tumor_images, non_tumor_images), axis=0)
    labels = np.concatenate((tumor_labels, non_tumor_labels), axis=0)

    # Normalize images to [0, 1]
    images = images / 255.0
    return images, labels