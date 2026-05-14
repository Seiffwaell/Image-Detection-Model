import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

print("Loading CIFAR-10 data...")
(X_train_orig, y_train_orig), (X_test_orig, y_test_orig) = cifar10.load_data()

# Use the full dataset for real training
sample_size = 50000 
X_full = np.concatenate((X_train_orig, X_test_orig), axis=0)[:sample_size]
y_full = np.concatenate((y_train_orig, y_test_orig), axis=0)[:sample_size].flatten()

X_dl = X_full.astype('float32') / 255.0
y_dl = to_categorical(y_full, num_classes=10)

X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_dl, test_size=0.2, stratify=y_full, random_state=42)

print("Building CNN model from notebook...")
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model... (this may take a minute)")
model_cnn.fit(
    X_train_dl, y_train_dl,
    validation_split=0.15,
    epochs=15,
    batch_size=128,
    verbose=1
)

model_path = 'cifar10_model.h5'
model_cnn.save(model_path)
print(f"Model saved successfully to {model_path}!")
