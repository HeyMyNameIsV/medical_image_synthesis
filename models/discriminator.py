import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers


def build_discriminator(input_shape=(128, 128, 1), num_classes=2):
    image_input = layers.Input(shape=input_shape)
    label_input = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, np.prod(input_shape))(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape(input_shape)(label_embedding)

    model_input = layers.concatenate([image_input, label_embedding])
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(model_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model([image_input, label_input], output)
    return model