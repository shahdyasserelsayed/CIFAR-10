import tensorflow as tf
import os
from PIL import Image
import numpy as np

# Absolute output folder path
output_dir = os.path.join(os.path.dirname(__file__), "static", "samples")
os.makedirs(output_dir, exist_ok=True)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 class names
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Save one image per class
for i, class_name in enumerate(class_names):
    idx = np.where(y_test.flatten() == i)[0][0]
    img = Image.fromarray(x_test[idx])
    file_path = os.path.join(output_dir, f"{class_name}.png")
    img.save(file_path)

print(f"Saved CIFAR-10 sample images in: {output_dir}")
