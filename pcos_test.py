import numpy as np  # type: ignore
import cv2  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

# Load the trained model
model_path = "/mnt/c/Users/navya/Documents/pcos_detection_model.h5"  # Ensure the correct path
model = load_model(model_path)

# Load and preprocess the image
image_path = "/mnt/c/Users/navya/OneDrive/Desktop/pcos_detection/test/infected/img_0_245.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))  # Resize to match model input size
img = img / 255.0  # Normalize pixel values (0 to 1)
img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 224, 224, 3)

# Make a prediction
prediction = model.predict(img)


# Interpret the result
threshold = 0.5  # Adjust based on training

# Check if prediction needs to be flipped
if prediction[0][0] >= threshold:
    result = "Not Infected (No PCOS detected)"
else:
    result = "Infected (PCOS detected)"

print("Final Classification:", result)
