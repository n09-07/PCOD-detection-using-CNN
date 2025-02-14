Requirements:

1.Check Python version:

python - -version
Or
python3 - -version
2. Install Backend Dependencies (Flask, Pandas, TensorFlow, OpenCV, etc.):
For Windows (cmd/PowerShell)
pip install flask 
pip install pandas
pip install numpy 
pip install tensorflow 
pip intsall keras
pip install opencv-python 
pip install pillow
pip install flask-cors

For Mac/Linux (Terminal)
pip3 install flask 
pip3 install pandas 
pip3 install numpy 
pip3 install tensorflow 
pip3 install keras 
pip3 install opencv-python
pip3 install  pillow 
pip3 install flask-cors

commands to run the codes:
for app.py:-
python app.py
or 
python3 app.py
for pcos_test.py:-
python3 pcos_test.py
or 
python pcos_test.py
for test_model.py:-
python3 test_model.py
or 
python test_model.py
to train the model in colab for pcos_model.py:-
python3 pcos_model.py
or 
python pcos_model.py


pcos_detection/
│── static/                # Stores CSS, JavaScript, and images
│   ├── styles.css         # Your CSS file
│── templates/             # Stores HTML templates
│   ├── index.html         # Main page with upload form
│   ├── result.html        # Page to display results
│── uploads/               # Stores uploaded images (created dynamically)
│── pcos_model.h5          # Your trained model(not stored under same root folder)
│── app.py                 # Flask backend
