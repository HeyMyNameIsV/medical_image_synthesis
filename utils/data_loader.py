import numpy as np

def load_data():
    brain_tumor_images = np.load('data/brain_tumor_images.npy')
    non_brain_tumor_images = np.load('data/non_brain_tumor_images.npy')
    return brain_tumor_images, non_brain_tumor_images