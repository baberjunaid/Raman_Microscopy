from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from werkzeug.utils import secure_filename
import joblib  # For loading the SVM model
import resnet_model
from svm_model import svm_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained models
resnet_model = resnet_model.model
# svm_model = joblib.load('svm_model.pkl')

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.get('input_data')
    
    if input_data:
        input_data = np.fromstring(input_data, sep=',').reshape(1600, 2)
        input_data = np.expand_dims(input_data, axis=0)

        # ResNet prediction
        resnet_prediction = resnet_model.predict(input_data)
        resnet_predicted_class = np.argmax(resnet_prediction)

        # SVM prediction
        input_data_flat = input_data.reshape(1, -1)
        svm_predicted_class = svm_model.predict(input_data_flat)
        class_names = ['control_Nucleous', 'MV_DATA', 'SARS_Nucleous']

        return render_template('result.html', resnet_result=resnet_predicted_class, svm_result=svm_predicted_class[0],class_names=class_names)
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('files[]')
    predictions = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1])
        input_data = df.values.reshape(1, 1600, 2)

        # ResNet prediction
        resnet_prediction = resnet_model.predict(input_data)
        resnet_predicted_class = np.argmax(resnet_prediction)

        # SVM prediction
        input_data_flat = input_data.reshape(1, -1)
        svm_predicted_class = svm_model.predict(input_data_flat)

        predictions.append((filename, resnet_predicted_class, svm_predicted_class[0]))
    class_names = ['control_Nucleous', 'MV_DATA', 'SARS_Nucleous']
    return render_template('result.html', results=predictions, class_names=class_names)

if __name__ == '__main__':
    app.run(debug=True)
