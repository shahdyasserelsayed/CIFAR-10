from flask import Flask, render_template, request, send_from_directory
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Safe model path detection
base_dir = os.path.dirname(__file__)
saved_model_dir = os.path.join(base_dir, '..', 'saved_model')
keras_model = os.path.join(saved_model_dir, 'cifar10_model.keras')
h5_model = os.path.join(saved_model_dir, 'cifar10_model.h5')

if os.path.exists(keras_model):
    model_path = keras_model
elif os.path.exists(h5_model):
    model_path = h5_model
else:
    model_path = None
    print('No model found in saved_model/. The app will run but predictions will fail until you add a model.')

model = None
if model_path:
    model = tf.keras.models.load_model(model_path)
    print('Loaded model from', model_path)

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

UPLOAD_FOLDER = os.path.join(base_dir, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((32,32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    confidence = None
    img_rel_path = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and model is not None:
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            print('Saved upload to', save_path)
            img_rel_path = os.path.join('static','uploads', filename)
            x = preprocess_image(save_path)
            preds = model.predict(x)[0]
            class_id = int(np.argmax(preds))
            prediction = CLASS_NAMES[class_id]
            confidence = round(float(preds[class_id])*100,2)
        elif file and model is None:
            return render_template('index.html', error='No model available. Train or add model to saved_model/', prediction=None)
    return render_template('index.html', prediction=prediction, confidence=confidence, img_path=img_rel_path)

if __name__ == '__main__':
    app.run(debug=True)
