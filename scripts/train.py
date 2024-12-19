import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Reshape, Conv3D, Flatten, Dense
import keras
import psutil

print("GPUs available:", len(tf.config.list_physical_devices('GPU')))
def print_memory_usage(stage):
    print(f"{stage} Memory Usage: {psutil.virtual_memory().percent}%")


# Define a generator for lazy loading
def lazy_data_generator(file_path, batch_size, is_train=True):
    data = np.load(file_path, mmap_mode='r')
    if is_train:
        X = data['X_train']
        y = data['y_train']
    else:
        X = data['X_test']
        y = data['y_test']

    num_samples = len(X)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        yield X[start_idx:end_idx].astype(np.float16), keras.utils.to_categorical(y[start_idx:end_idx], num_classes=10)


# Print initial memory usage
print_memory_usage("Initial")

# Define model
model = Sequential([
    Input(shape=(30, 30, 30)),  # Define the input explicitly
    Reshape((30, 30, 30, 1)),  # Add a channel dimension
    Conv3D(16, kernel_size=6, strides=2, activation='relu'),
    Conv3D(64, kernel_size=5, strides=2, activation='relu'),
    Conv3D(64, kernel_size=5, strides=2, activation='relu'),
    Flatten(),
    Dense(10, activation='softmax')  # Output layer for 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training parameters
batch_size = 32
epochs = 30
data_path = '../data/modelnet10.npz'

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # Training generator
    train_gen = lazy_data_generator(data_path, batch_size, is_train=True)
    for X_batch, y_batch in train_gen:
        model.train_on_batch(X_batch, y_batch)

    # Validation generator
    val_gen = lazy_data_generator(data_path, batch_size, is_train=False)
    val_loss, val_acc = 0, 0
    val_steps = 0
    for X_batch, y_batch in val_gen:
        metrics = model.test_on_batch(X_batch, y_batch)
        val_loss += metrics[0]
        val_acc += metrics[1]
        val_steps += 1

    val_loss /= val_steps
    val_acc /= val_steps
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

# Save the model
model.save('./models/model.h5')

# Print final memory usage
print_memory_usage("Final")
