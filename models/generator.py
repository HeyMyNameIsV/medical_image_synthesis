import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_generator(noise_dim=100, num_classes=2):
    noise_input = layers.Input(shape=(noise_dim,))
    label_input = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, noise_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    model_input = layers.multiply([noise_input, label_embedding])
    x = layers.Dense(256 * 8 * 8, activation='relu')(model_input)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation='relu')(x)
    output = layers.Conv2DTranspose(1, (4, 4), strides=2, padding='same', activation='sigmoid')(x)

    model = tf.keras.Model([noise_input, label_input], output)
    return model