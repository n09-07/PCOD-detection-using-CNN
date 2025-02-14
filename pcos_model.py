import os
import numpy as np # type: ignore
from PIL import Image # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Define paths
train_path = "/content/drive/My Drive/pcos_dataset/train"
test_path = "/content/drive/My Drive/pcos_dataset/test"

# Function to validate image files
def validate_images(directory_path):
    """
    Removes invalid or corrupted images from the specified directory.
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Check if the image is valid
            except (IOError, SyntaxError):
                print(f"Invalid image file detected and removed: {file_path}")
                os.remove(file_path)

# Validate images in train and test directories
validate_images(train_path)
validate_images(test_path)

# Data augmentation for training and testing
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Loading the data
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb'  # Ensure consistent color handling
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb'
)

# Load VGG16 pre-trained model without the top layers
vgg_base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freeze the base model layers
vgg_base.trainable = False

# Build the full model
model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: PCOS detection
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=test_data, epochs=10)

# Save the trained model
model.save("pcos_detection_model.h5")

print("Model training complete and saved successfully.")
