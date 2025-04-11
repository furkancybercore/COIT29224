# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:25:46 2024

@author: 12223508
"""

# Part A: Building a CNN Model for Age and Gender Estimation using UTKFace Dataset

# -----------------------------
# Import necessary libraries
# -----------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import warnings
from tqdm import tqdm
from IPython.display import SVG
from google.colab import files  # Remove or comment out this line for local running
from tensorflow.keras.utils import model_to_dot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16


# Suppress Warnings and TensorFlow Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
tf.get_logger().setLevel('FATAL')         # Suppress TensorFlow warnings
warnings.filterwarnings("ignore")         # Suppress other warnings

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# -----------------------------
# Step 1: Data Loading and Preprocessing
# -----------------------------

# Path to the UTKFace dataset
dataset_path = '/content/UTKFace' # Change to your local dataset path, for example: dataset_path = 'C:/path_to_dataset/UTKFace'

# Initialize lists to store data and labels
images = []
age_list = []
gender_list = []

# Define parameters                # Recomended parameters for local;
IMG_HEIGHT, IMG_WIDTH = 200, 200   # 32x32
batch_size = 32                    # 8  
epoch = 10                         # 5  

# Count total number of jpg files
total_images = sum(1 for img_name in os.listdir(dataset_path) if img_name.endswith('.jpg'))

# Load images and extract labels with progress bar
print("Loading and preprocessing images...")
for img_name in tqdm(os.listdir(dataset_path), total=total_images, desc="Processing"):
    if img_name.endswith('.jpg'):
        try:
            # Parse filename to get age and gender
            age, gender = map(int, img_name.split('_')[:2])

            # Load and preprocess the image
            img = cv2.imread(os.path.join(dataset_path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            age_list.append(age)
            gender_list.append(gender)
        except Exception as e:
            print(f"Skipped {img_name}: {e}")

# Convert lists to numpy arrays
print("Converting to numpy arrays...")
images = np.array(images) / 255.0  # Normalize images
age_list = np.array(age_list)
gender_list = np.array(gender_list)

# -----------------------------
# Step 2: Label Preprocessing
# -----------------------------

# Categorize age into 5 groups
def categorize_age(age):
    if age <= 24:
        return 0
    elif age <= 49:
        return 1
    elif age <= 74:
        return 2
    elif age <= 99:
        return 3
    else:
        return 4

print("Categorizing ages...")
age_categories = np.array([categorize_age(age) for age in age_list])

# Encode gender labels
print("Encoding gender labels...")
gender_encoder = LabelBinarizer()
gender_labels = to_categorical(gender_encoder.fit_transform(gender_list))

# Encode age categories
print("Encoding age categories...")
age_encoder = LabelBinarizer()
age_labels = to_categorical(age_categories)

# -----------------------------
# Step 3: Split Data into Training and Testing Sets
# -----------------------------

print("Splitting data into training and testing sets...")
X_train, X_test, gender_train, gender_test, age_train, age_test = train_test_split(
    images, gender_labels, age_labels, test_size=0.2, random_state=42)

# -----------------------------
# Step 4: Data Augmentation
# -----------------------------

print("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
    height_shift_range=0.2, horizontal_flip=True, fill_mode="nearest")

# Prepare the data generator
print("Preparing data generator...")

# Create a custom data generator that yields augmented images and labels
def train_data_generator(X, gender_y, age_y, batch_size):
    while True:
        idx = np.random.randint(0, len(X), batch_size)
        X_batch = X[idx]
        gender_y_batch = gender_y[idx]
        age_y_batch = age_y[idx]

        # Apply augmentation to X_batch
        augmented_images = np.zeros_like(X_batch)
        for i in range(len(X_batch)):
            augmented_images[i] = datagen.random_transform(X_batch[i])

        yield augmented_images, {'gender_output': gender_y_batch, 'age_output': age_y_batch}

train_generator = train_data_generator(X_train, gender_train, age_train, batch_size)

# Calculate the number of steps per epoch
steps_per_epoch = len(X_train) // batch_size

print(f"Data generator ready. Steps per epoch: {steps_per_epoch}")

# -----------------------------
# Step 5: Build and Compile the CNN Model
# -----------------------------

# Using VGG16 Model for Transfer Learning
def build_vgg16_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False  # Freeze the layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # Gender output branch
    gender_output = Dense(2, activation='softmax', name='gender_output')(x)

    # Age output branch
    age_output = Dense(5, activation='softmax', name='age_output')(x)

    model = Model(inputs=base_model.input, outputs=[gender_output, age_output])
    return model

print("Building and compiling the model...")
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
model = build_vgg16_model(input_shape)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss={'gender_output': 'categorical_crossentropy', 'age_output': 'categorical_crossentropy'},
    metrics={'gender_output': 'accuracy', 'age_output': 'accuracy'}
)

# Visualize and save the model architecture
print("Saving model architecture...")

try:
    # Display the model architecture within the notebook
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
    # Save the model architecture to a file
    plot_model(model, to_file='vgg16_model_architecture.png', show_shapes=True)
    files.download('vgg16_model_architecture.png') # Remove or comment out this line for local running
    print("Model architecture saved successfully.")
except Exception as e:
    print("An error occurred while trying to plot the model:")
    print(str(e))
    print("Printing model summary instead.")
    print(model.summary())

# -----------------------------
# Step 6: Train the Model
# -----------------------------

print("Training the model...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_test, {'gender_output': gender_test, 'age_output': age_test}),
    epochs=epoch,
    verbose=1
)

# -----------------------------
# Step 7: Evaluate the Model
# -----------------------------

print("Plotting accuracy curves...")
# Plot accuracy curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['gender_output_accuracy'], label='Train Gender Acc')
plt.plot(history.history['val_gender_output_accuracy'], label='Val Gender Acc')
plt.legend(); plt.title('Gender Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['age_output_accuracy'], label='Train Age Acc')
plt.plot(history.history['val_age_output_accuracy'], label='Val Age Acc')
plt.legend(); plt.title('Age Accuracy')
plt.show()

# -----------------------------
# Step 8: Make Sample Predictions
# -----------------------------

print("Making sample predictions...")
# Select 10 random samples
indices = np.random.choice(len(X_test), 10, replace=False)
sample_images = X_test[indices]
sample_genders = gender_test[indices]
sample_ages = age_test[indices]

# Predict on samples
gender_pred, age_pred = model.predict(sample_images)

# Define label mappings
gender_labels_list = ['Male', 'Female']
age_labels_list = ['0-24', '25-49', '50-74', '75-99', '100-124']

# Display predictions
for i in range(10):
    plt.imshow(sample_images[i])
    plt.axis('off')
    true_gender = gender_labels_list[np.argmax(sample_genders[i])]
    true_age = age_labels_list[np.argmax(sample_ages[i])]
    pred_gender = gender_labels_list[np.argmax(gender_pred[i])]
    pred_age = age_labels_list[np.argmax(age_pred[i])]
    plt.title(f"Actual: {true_gender}, {true_age}\nPredicted: {pred_gender}, {pred_age}")
    plt.show()

# -----------------------------
# End of Script
# -----------------------------

print("--- Image classification neural network processes finished ---\n")