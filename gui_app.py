import tkinter as tk
from tkinter import filedialog
from tkinter import Canvas
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from models.generator import build_generator

class ImageGeneratorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("GAN Image Generator")
        
        self.canvas = Canvas(master, width=128, height=128)
        self.canvas.pack()

        self.generate_button = tk.Button(master, text="Generate Image", command=self.generate_images)
        self.generate_button.pack()

        self.generator = build_generator()
        self.generator.load_weights('generator_w.weights.h5')

    def generate_images(self):
        random_noise = tf.random.normal([1, 100])
        generated_images = self.generator(random_noise, training=False)
        
        if generated_images.shape[1:] != (128, 128, 1):
            raise ValueError(f"Expected 128x128 images, but got {generated_images.shape[1:]}")
        
        generated_images = generated_images.numpy().reshape(128, 128)
        img = (generated_images * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        self.master.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()