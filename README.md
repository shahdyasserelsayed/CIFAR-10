# CIFAR-10 Image Classification (TensorFlow / Keras)

## 🧠 Overview
This project trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and provides **two interactive apps** for inference:
- A **Flask Web App** for uploading and classifying images.
- A **Streamlit Dashboard** for interactive visualization and model insights.

---

## 📁 Included
- `src/model.py` : CNN model architecture
- `src/train.py` : Training script using TensorFlow Datasets (TFDS)
- `src/evaluate.py` : Model evaluation script
- `src/app.py` : Flask web app (upload images, predict class + confidence)
- `src/app_streamlit.py` : Streamlit dashboard for visualization and predictions
- `src/templates/index.html` : HTML template for Flask UI
- `saved_model/` : Folder to store the trained model
- `requirements.txt` : Project dependencies
- `main.py` : main execution script

---

## ⚙️ How to Run (Locally)

### 1️⃣ Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2️⃣ Train the Model (Optional)
```bash
cd src
python train.py
```
This will train the CNN on the CIFAR-10 dataset and save the model as:
```
../saved_model/cifar10_model.keras
```

### 3️⃣ Evaluate the Model
```bash
python evaluate.py
```

### 4️⃣ Run the Flask App
```bash
python app.py
```
Open the app in your browser at: **http://127.0.0.1:5000**

### 5️⃣ Run the Streamlit Dashboard
```bash
streamlit run app_streamlit.py
```

---

## 🎨 App Previews

### 🌐 Flask Web App
<img src="images_output\flask-output.png" width="600" alt="Flask Web Interface">

### 📊 Streamlit Dashboard
<img src="images_output\streamlit-output1.png" width="600" alt="Streamlit Dashboard Interface">

---

## 🧩 Technologies Used
- TensorFlow / Keras
- TensorFlow Datasets (TFDS)
- NumPy, Matplotlib, Seaborn
- Flask & Streamlit
- Pillow (PIL) for image processing

---

## 💾 Project Structure
```
CIFAR-10 Project/
│
├── saved_model/
│   └── cifar10_model.keras
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── app.py 
│   ├── app_streamlit.py
│   └── templates/
│       └── index.html
│
├── images_output/
│   ├── flask_app.png
│   └── streamlit_dashboard.png
│
├── main.py
├── README.md
└── requirements.txt
```

---

## ✨ Author
Developed by **Shahd Yasser**  
For DEPI TensorFlow Assignments (CIFAR-10 Project)

---
