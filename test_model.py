from tensorflow.keras.models import load_model # type: ignore

# Define the model path
model_path = "/home/navya/pcos_detection_model.h5"

# Load the trained model
model = load_model(model_path)

# Check if model loads successfully
print("Model loaded successfully!")

import numpy as np # type: ignore

# Create a dummy input (adjust the shape based on your model's input size)
dummy_input = np.random.rand(1, 224, 224, 3)  # Example for an image-based model

# Make a prediction
prediction = model.predict(dummy_input)

print("Model output:", prediction)

model.summary()