import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_discriminator(input_shape=(128, 128, 1)):
    model = tf.keras.Sequential([
        layers.Conv2D(64, (4, 4), strides=2, padding='same', input_shape=input_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (4, 4), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(256, (4, 4), strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid'),  # Use sigmoid activation for binary classification
    ])
    return model