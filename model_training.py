import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras import losses
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.data_loader import load_data
from fid import calculate_fid
from inception_score import calculate_inception_score
from data_agumentation import augment_images
from visualization import plot_generated_images

class ConditionalGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(ConditionalGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=False)

    def compile(self, generator_optimizer, discriminator_optimizer):
        super(ConditionalGAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    @tf.function
    def train_step(self, data):
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator([noise, labels], training=True)

            real_output = self.discriminator([real_images, labels], training=True)
            fake_output = self.discriminator([generated_images, labels], training=True)

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
    images, labels = load_data()

    # Apply augmentation
    augmented_images = augment_images(images)

    # Initialize models
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile GAN
    gan = ConditionalGAN(generator, discriminator)
    generator_optimizer = optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = optimizers.Adam(1e-4, beta_1=0.5)
    gan.compile(generator_optimizer, discriminator_optimizer)

    # Train GAN
    epochs = 50
    batch_size = 64
    for epoch in range(epochs):
        gan.fit((augmented_images, labels), epochs=1, batch_size=batch_size)
        
        # Plot generated images
        plot_generated_images(generator, epoch)
        
        # Calculate and print FID and IS
        if epoch % 10 == 0 or epoch == epochs - 1:
            fid = calculate_fid(images, generator(tf.random.normal([len(images), 100]), training=False))
            inception_score = calculate_inception_score(generator(tf.random.normal([len(images), 100]), training=False))
            print(f'Epoch {epoch}: FID = {fid}, IS = {inception_score}')
    
    generator.save_weights('generator_w.weights.h5')