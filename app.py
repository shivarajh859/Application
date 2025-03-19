from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = 'models/my_model.h5'
model = load_model(MODEL_PATH)

# Function to predict malaria

# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img = image.img_to_array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
#
#     preds = model.predict(img)
#     print("Raw Predictions:", preds)  # Debugging output
#
#     if preds.shape[-1] == 1:  # Sigmoid (Binary Classification)
#         return "Parasitized" if preds[0][0] > 0.5 else "Uninfected"
#     else:  # Softmax (Multiclass Classification)
#         return "Parasitized" if np.argmax(preds) == 1 else "Uninfected"

# Function to predict malaria
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    preds = model.predict(img_array)
    return "Parasitized" if preds[0][0] < 0.5 else "Uninfected"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            prediction = model_predict(file_path, model)
            return render_template('result.html', prediction=prediction, image_path=file_path)

    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
