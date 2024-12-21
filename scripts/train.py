import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv3D
from sklearn.utils import shuffle
import keras
import matplotlib.pyplot as plt

# Load data
#data = np.load('./data/modelnet10.npz')
data = np.load('/content/drive/MyDrive/convert_3d_off_file/data/modelnet10.npz') #training on ggcolab
X, y = shuffle(data['X_train'], data['y_train'])
X_test, y_test = shuffle(data['X_test'], data['y_test'])

# Convert labels to one-hot encoding
y = keras.utils.to_categorical(y, num_classes=10)

# Define model
model = Sequential([
    Reshape((36, 36, 36, 1), input_shape=(36, 36, 36)),
    Conv3D(16, kernel_size=6, strides=2, activation='relu'),
    Conv3D(64, kernel_size=5, strides=2, activation='relu'),
    Conv3D(64, kernel_size=5, strides=2, activation='relu'),
    Flatten(),
    Dense(10, activation='softmax')
])
print("Shape of X:", X.shape)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and save the history
history = model.fit(X, y, batch_size=256, epochs=30, validation_split=0.2, shuffle=True)

# Save the model
model.save('/content/drive/MyDrive/convert_3d_off_file/models/model.keras')

# Ensure the directory for plots exists
plot_dir = '/content/drive/MyDrive/convert_3d_off_file/plots'
os.makedirs(plot_dir, exist_ok=True)

# Save and display loss graph
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
loss_plot_path = os.path.join(plot_dir, 'loss_plot.png')
plt.savefig(loss_plot_path)
plt.show()

# Save and display accuracy graph
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
accuracy_plot_path = os.path.join(plot_dir, 'accuracy_plot.png')
plt.savefig(accuracy_plot_path)
plt.show()

print(f"Loss plot saved at: {loss_plot_path}")
print(f"Accuracy plot saved at: {accuracy_plot_path}")
