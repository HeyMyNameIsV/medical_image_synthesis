import numpy as np
import tensorflow_gan as tfgan

def calculate_inception_score(generated_images, num_splits=10):
    inception_score = tfgan.eval.classifier_score_from_logits(generated_images, num_splits=num_splits)
    return inception_score