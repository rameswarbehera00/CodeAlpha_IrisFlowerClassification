# Iris Flower Classification — CodeAlpha Data Science Internship

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning project developed for the **CodeAlpha Data Science Internship**. This project implements supervised classification algorithms to classify iris flowers into three species based on morphological measurements.

---

## 📌 Project Overview

The objective of this project is to train and evaluate supervised classification models to predict the species of an Iris flower given four physical measurements:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

### Target Classes
1. *Iris-setosa*
2. *Iris-versicolor*
3. *Iris-virginica*

---

## 📁 Repository Structure

```text
CodeAlpha_IrisFlowerClassification/
├── data/
│   └── Iris.csv                   # Raw dataset containing 150 instances
├── models/
│   ├── iris_lr_model.pkl          # Trained Logistic Regression model
│   ├── scaler.pkl                 # Fitted StandardScaler object
│   └── label_encoder.pkl          # Fitted LabelEncoder for class decoding
├── notebooks/
│   └── iris_pairplot.png          # Feature relationship pairplot visualization
├── src/
│   ├── explore.py                 # Initial data inspection and summary stats
│   ├── visualize.py               # Pairplot generation script
│   ├── train.py                   # Preprocessing, training, and model export
│   └── predict.py                 # Real-time inference script for new inputs
├── .gitignore                     # Ignores virtual environment and cached files
├── LICENSE                        # MIT License
├── README.md                      # Comprehensive project documentation
└── requirements.txt               # List of project dependencies

```

---

## 🔬 Exploratory Data Analysis & Insights

* **Distribution:** The dataset contains 150 samples with zero missing values, perfectly balanced across all 3 classes (50 samples each).
* **Separability:** Feature visualization via pairplots demonstrates that *Iris-setosa* is linearly separable using petal dimensions alone.
* **Overlap:** *Iris-versicolor* and *Iris-virginica* exhibit slight overlap across sepal measurements, necessitating standardized scaling and multi-feature decision boundaries.

---

## ⚙️ Model Evaluation & Performance

The dataset was split using an **80/20 stratified split** (120 training samples, 30 unseen test samples) with `StandardScaler` applied to prevent scale dominance.

| Model | Evaluation Metric | Score |
| --- | --- | --- |
| **Logistic Regression** | Test Accuracy | **93.33%** |
| **K-Nearest Neighbors ($k=3$)** | Test Accuracy | **93.33%** |

### Classification Report (Logistic Regression)

```text
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       0.90      0.90      0.90        10
 Iris-virginica       0.90      0.90      0.90        10

       accuracy                           0.93        30
      macro avg       0.93      0.93      0.93        30
   weighted avg       0.93      0.93      0.93        30

```

---

## 🚀 Getting Started (Windows Setup)

### 1. Clone the Repository

```cmd
git clone [https://github.com/](https://github.com/)<your-username>/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification

```

### 2. Set Up Virtual Environment & Dependencies

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Run Data Exploration

```cmd
python src\explore.py

```

### 4. Generate Visualizations

```cmd
python src\visualize.py

```

### 5. Train Models & Export Artifacts

```cmd
python src\train.py

```

### 6. Run Inference on Custom Inputs

```cmd
python src\predict.py

```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

