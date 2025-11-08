<h1 align="center">💰 Fake Currency Detection using Digital Image Processing & Machine Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Image_Processing-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/scikit--learn-Machine_Learning-orange?logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/github/stars/YOUR_GITHUB_USERNAME/FAKE_CURRENCY_DETECTION?style=social" alt="Stars">
</p>

---

### 📸 **Project Overview**

**Fake Currency Detection** is a **Machine Learning-based Computer Vision project** designed to identify counterfeit banknotes using **Digital Image Processing** techniques.  
By analyzing **texture patterns** and **local visual features**, the system can differentiate *real* currency notes from *fake* ones with high accuracy.

---

### 🎯 **Objective**

> To build an automated system that accurately detects counterfeit currency using image-based texture analysis and machine learning.

---

### 🧰 **Tech Stack**

| Category | Tools Used |
|-----------|-------------|
| **Programming Language** | Python |
| **Libraries** | OpenCV, NumPy, Pandas, scikit-image, scikit-learn, joblib, matplotlib |
| **Algorithm** | Local Binary Pattern (LBP) for texture extraction |
| **Model** | Random Forest Classifier (best performing model) |
| **Environment** | Jupyter Notebook |

---

### ⚙️ **Project Workflow**

```mermaid
graph TD;
A[Input Currency Image] --> B[Preprocessing: Grayscale and Resize];
B --> C[Feature Extraction using LBP];
C --> D[Feature Scaling with Standard Scaler];
D --> E[Model Prediction using Random Forest];
E --> F{Real or Fake?};

🧩 Implementation Steps
1️⃣ Image Preprocessing
Convert RGB image to grayscale

Resize for uniformity

Apply noise reduction

2️⃣ Feature Extraction
Extract Local Binary Pattern (LBP) features

Compute histogram of LBP values representing note texture

3️⃣ Model Training
Train and compare Random Forest, SVM, and Logistic Regression models

Perform GridSearchCV for hyperparameter tuning

4️⃣ Evaluation
Confusion Matrix

Accuracy, Precision, Recall, and ROC-AUC metrics

5️⃣ Model Saving
Save best model as rf_currency_detector.pkl

Save scaler as scaler_currency.pkl

📊 Results
Model	Accuracy	Remarks
Logistic Regression	89%	Baseline model
SVM	93%	Better generalization
Random Forest	97%	Best accuracy & robustness

✅ Final model used: Random Forest Classifier

💻 Sample Output
Input Image	Predicted Result
✅ Real Currency
❌ Fake Currency

🌍 Real-world Applications
🔹 Integration in ATMs and cash counting machines

🔹 Bank and retail cash verification systems

🔹 Forensic analysis of counterfeit notes

🔹 Educational demonstration for ML + DIP synergy

🚀 Future Enhancements
Implement Deep Learning (CNN) for real-time detection

Develop a web or mobile app interface for image upload & detection

Expand dataset for multiple denominations and lighting variations

Add explainable AI layer to visualize feature importance

📁 Project Structure
bash
Copy code
FAKE_CURRENCY_DETECTION/
│
├── FAKE_CURRENCY_DETECTION.docx
├── FAKE_CURRENCY_DETECTION.pdf
├── Digital-Image-Processing-Project-Counterfeit-Currency-Detection.pptx
├── Untitled.ipynb                  # Main notebook
├── rf_currency_detector.pkl        # Trained model
├── scaler_currency.pkl             # Feature scaler
└── dataset/                        # Real & fake currency images
🧠 Concept Behind LBP (Local Binary Pattern)
LBP encodes texture by comparing each pixel with its neighborhood.
If neighboring pixels are brighter, it’s assigned 1; otherwise 0.
The resulting binary pattern represents surface texture — real notes have smoother, consistent patterns, while fake notes show irregularities.

<p align="center"> <img src="https://miro.medium.com/v2/resize:fit:800/format:webp/1*jXz2tT5XeStZMCzSr1m4gQ.png" width="400" alt="LBP Illustration"/> </p> ```
