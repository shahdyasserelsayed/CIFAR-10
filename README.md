# CIFAR-10 Image Classification (TensorFlow / Keras)

## Overview
This project trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and provides a simple Flask web app to upload images and get predictions.

## Included
- `src/model.py` : CNN architecture factory
- `src/train.py` : Training script (uses `tensorflow_datasets` for reliable downloads)
- `src/evaluate.py` : Evaluation script (uses TFDS)
- `src/app.py` : Flask web app (upload images, predict class + confidence)
- `src/templates/index.html` : Simple web UI template
- `saved_model/` : Place to store saved model (empty placeholder)
- `requirements.txt` : Python dependencies

## How to run (local)
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   ```
2. Train the model (optional, will download CIFAR-10 via TFDS):
   ```bash
   cd src
   python train.py
   ```
   This saves the model as `../saved_model/cifar10_model.keras` (preferred) or `.h5`.
3. Evaluate:
   ```bash
   python evaluate.py
   ```
4. Run the Flask web app (requires the saved model in `saved_model/`):
   ```bash
   python app.py
   ```
   Open http://127.0.0.1:5000 in your browser.
