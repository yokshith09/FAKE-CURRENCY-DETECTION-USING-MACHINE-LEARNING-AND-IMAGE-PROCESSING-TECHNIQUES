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
A[Input Currency Image] --> B[Preprocessing: Grayscale + Resize];
B --> C[Feature Extraction using LBP];
C --> D[Feature Scaling (StandardScaler)];
D --> E[Model Prediction (Random Forest)];
E --> F{Real or Fake?};

