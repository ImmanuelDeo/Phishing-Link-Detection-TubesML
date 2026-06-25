
# Dataset Documentation - PhiUSIIL Phishing URL Dataset

## Dataset Overview
The dataset used in this project is the **PhiUSIIL Phishing URL Dataset**, which contains extracted features from URLs and their corresponding classification labels.

This dataset is designed for detecting phishing websites using machine learning techniques.

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
