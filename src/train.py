import tensorflow as tf
import tensorflow_datasets as tfds
from model import create_cifar10_model

# Load CIFAR-10 dataset
(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Normalize images
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(normalize_img).shuffle(50000).batch(64).prefetch(tf.data.AUTOTUNE)
ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y))
ds_test = ds_test.map(normalize_img).batch(64).prefetch(tf.data.AUTOTUNE)

# Create model
model = create_cifar10_model()

# Add callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('../saved_model/cifar10_best_model.h5',
                                                monitor='val_accuracy',
                                                save_best_only=True)

# Train model
history = model.fit(
    ds_train,
    epochs=30,  # more epochs for better learning
    validation_data=ds_test,
    callbacks=[early_stop, checkpoint]
)

# Save model
model.save('../saved_model/cifar10_model.h5')
print("Model trained and saved successfully!")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()
