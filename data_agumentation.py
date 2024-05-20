import tensorflow as tf
from utils.data_loader import load_data

def augment_images(images):
    augmented_images = []
    for img in images:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        augmented_images.append(img)
    return tf.stack(augmented_images)

# Apply augmentation during data loading
brain_tumor_images, _ = load_data()
augmented_brain_tumor_images = augment_images(brain_tumor_images)