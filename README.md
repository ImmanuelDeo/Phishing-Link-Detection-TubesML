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

# Dataset Documentation - PhiUSIIL Phishing URL Dataset
The dataset used in this project is the **PhiUSIIL Phishing URL Dataset**, which contains extracted features from URLs and their corresponding classification labels. This dataset is designed for detecting phishing websites using machine learning techniques.

---

## Dataset Details
- Total Samples: **235,795**
- Total Features: **56**
- Task Type: Binary Classification

---

## Target Variable
- `label`
  - `1` → Legitimate URL
  - `0` → Phishing URL

---

## Feature Categories

### 1. URL-based Features
These features are extracted directly from the URL string:
- URLLength
- NoOfLettersInURL
- LetterRatioInURL
- NoOfDigitsInURL
- DigitRatioInURL
- NoOfSpecialCharsInURL
- URLSimilarityIndex

---

### 2. Domain-based Features
Information about domain structure:
- DomainLength
- IsDomainIP
- TLD
- NoOfSubDomain
- TLDLength
- TLDLegitimateProb

---

### 3. Security & Behavior Features
Indicators of suspicious behavior:
- IsHTTPS
- HasObfuscation
- NoOfRedirects
- HasPasswordField
- HasHiddenFields
- HasSubmitButton

---

### 4. Content-based Features
Extracted from page content:
- HasTitle
- Title
- HasDescription
- HasFavicon
- HasSocialNet
- HasCopyrightInfo

---

### 5. HTML/Structure Features
- NoOfImage
- NoOfCSS
- NoOfJS
- NoOfiFrame
- NoOfPopup

---

### 6. External Link Features
- NoOfExternalRef
- NoOfSelfRef
- NoOfEmptyRef

---

## Data Quality
- No missing values (clean dataset)
- Large-scale dataset suitable for ML training
- Balanced binary classification setup (to be verified during EDA)

---

## Notes
- Dataset is already feature-engineered (no raw HTML needed)
- Some features may require normalization
- Feature importance analysis is recommended after modeling

---

## Source
UCI Machine Learning Repository - PhiUSIIL Phishing URL Dataset
https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset

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
