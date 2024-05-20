import matplotlib.pyplot as plt
import tensorflow as tf

def plot_generated_images(generator, epoch, noise_dim=100, examples=10):
    noise = tf.random.normal([examples, noise_dim])
    generated_images = generator(noise, training=False)

    fig, axes = plt.subplots(1, examples, figsize=(examples, 1))
    for i in range(examples):
        axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
    plt.savefig(f'generated_images_epoch_{epoch}.png')
    plt.show()