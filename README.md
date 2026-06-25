# Phishing URL Detection using Machine Learning
## Project Overview
This project aims to build a Machine Learning model that can detect whether a given URL is **phishing or legitimate** based on extracted structural, lexical, and host-based features.

Phishing attacks are one of the most common cybersecurity threats today, where attackers create fake websites or URLs to steal sensitive user information such as passwords, credit card data, and personal identity.

The goal of this project is to help mitigate such threats using a data-driven machine learning approach.

---

## Objectives
- Analyze phishing URL dataset
- Perform Exploratory Data Analysis (EDA)
- Preprocess and clean dataset
- Build multiple machine learning models
- Compare model performance
- Identify key features contributing to phishing detection

---

## Team Members
- Member 1: Jury Maxwell-103132400023
- Member 2: Muhammad 'Izzuddin Nabil-103132400032
- Member 3: Immanuel Raditya Deo-103132430001

---

## Dataset Source
- Dataset Name: PhiUSIIL Phishing URL Dataset
- Source: UCI Machine Learning Repository
- Type: Binary Classification
- Target Variable: `label`
  - `1` → Legitimate URL
  - `0` → Phishing URL

---

## 📁 Project Structure
phishing-link-detection-ml/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_eda.ipynb
│ └── 02_modeling.ipynb
│
├── src/
│ ├── data/
│ ├── models/
│ ├── utils/
│ └── pipeline/
│
├── artifacts/
├── requirements.txt
├── README.md
└── main.py

## Workflow
1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Engineering (if needed)
5. Model Training
6. Model Evaluation
7. Results Analysis

---

## Machine Learning Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting / XGBoost

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## Key Insight Goal
This project focuses on minimizing **false negatives**, because failing to detect phishing URLs can lead to real security risks.

---

## How to Run Project
```bash
pip install -r requirements.txt
python main.py