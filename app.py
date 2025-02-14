import os
import numpy as np  # type: ignore
import cv2  # type: ignore
from flask import Flask, render_template, request, send_from_directory  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "/mnt/c/Users/navya/Documents/pcos_detection_model.h5"  # Adjust if needed
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")  # Dynamic path
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Home page with upload form
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Preprocess image and make prediction
            img = preprocess_image(filepath)
            prediction = model.predict(img)
            raw_output = prediction[0][0]

            print(f"Raw model output: {raw_output}")  # Debugging line

            # Flip the condition if necessary
            threshold = 0.5  # Adjust based on training
            result = "Not Infected (No PCOS detected)" if raw_output >= threshold else "Infected (PCOS detected)"

            return render_template("result.html", filename=filename, result=result, probability=raw_output)
    
    return render_template("index.html")

# Route to serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
