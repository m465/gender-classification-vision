# Gender Classification from Pedestrian Images

This project focuses on **binary gender classification** (male vs. female) from pedestrian images using a hybrid approach of classical and deep learning techniques.

## 📁 Dataset Structure
```
dataset/
├── MIT-IB/
│   ├── female/
│   │   ├── img1.jpg
│   ├── male1/
│   │   ├── img2.jpg
```

---

## 🔧 Project Pipeline

### 1. Data Preparation
- Augmentation (rotation, flipping) for minority class (female)
- Image preprocessing and enhancement

### 2. Feature Engineering
- **Low-Level Features**: HOG, LBP
- **High-Level Features**: FC7 Layer from VGG19

### 3. Feature Fusion & Dimensionality Reduction
- Serial feature fusion
- PCA to reduce feature dimensions

### 4. Classification
- Linear SVM classifier
- 10-fold cross-validation
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

### 5. Testing on New Images
- Upload image using Google Colab interface
- Predict gender using the trained model

---

## 📊 Technologies Used
- Python
- OpenCV
- Scikit-learn
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn

---

## ▶️ How to Run

1. Clone this repo:
```bash
git clone https://github.com/yourusername/gender-classification-vision.git
cd gender-classification-vision
```

2. Open the notebook in **Google Colab** and upload your dataset.

3. Run each cell sequentially to:
   - Augment & preprocess data
   - Extract features
   - Train model
   - Save/load model
   - Upload and predict on new images

---

## 📷 Predict from Image
Use the following in Colab:
```python
from google.colab import files
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img = load_img(img_path, target_size=(224, 224))
print(predict_image_pil(img))
```

---

## ✅ Results
- Achieved high accuracy with a balanced dataset using hybrid features
- Reduced overfitting on minority class via targeted augmentation
- Real-time prediction support for new uploads

---

## 📌 License
This project is licensed under the MIT License.

---

## 🤝 Acknowledgements
- MIT-IB pedestrian dataset
- VGG19 model (Keras applications)
