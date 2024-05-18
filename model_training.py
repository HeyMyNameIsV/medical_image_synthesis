import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras import losses
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.data_loader import load_data

class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=False)

    def compile(self, generator_optimizer, discriminator_optimizer):
        super(GAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

if __name__ == "__main__":
    # Load and preprocess data
    brain_tumor_images, _ = load_data()  # Only use brain tumor images for training

    # Initialize models
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile GAN
    gan = GAN(generator, discriminator)
    generator_optimizer = optimizers.Adam(1e-4)
    discriminator_optimizer = optimizers.Adam(1e-4)
    gan.compile(generator_optimizer, discriminator_optimizer)

    # Train GAN
    gan.fit(brain_tumor_images, epochs=50, batch_size=64)
    
    generator.save_weights('generator_w.weights.h5')