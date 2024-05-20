import tensorflow as tf
import tensorflow_gan as tfgan
from keras import applications


def calculate_fid(real_images, generated_images):
    # Load InceptionV3 model pre-trained on ImageNet
    inception_model = applications.inception_v3.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Preprocess images
    real_images = tf.image.resize(real_images, (299, 299))
    generated_images = tf.image.resize(generated_images, (299, 299))
    real_images = applications.inception_v3.preprocess_input(real_images)
    generated_images = applications.inception_v3.preprocess_input(generated_images)

    # Calculate embeddings
    real_embeddings = inception_model(real_images)
    generated_embeddings = inception_model(generated_images)

    # Calculate FID
    fid = tfgan.eval.frechet_classifier_distance_from_activations(real_embeddings, generated_embeddings)
    return fid