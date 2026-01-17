# 🧠 Feature Engineering Portfolio — Healthcare Datasets

**Real-World Feature Engineering • Exploratory Data Analysis • Statistical Validation • ML-Ready Pipelines**

**Python • Pandas • NumPy • Scikit-learn • Jupyter Notebook**  
**License: MIT**

---

## 📌 Repository Overview
This repository is a **feature-engineering portfolio** focused on transforming real-world healthcare datasets into **machine-learning-ready** formats.

Each notebook demonstrates how raw medical data is systematically processed through:

- **Exploratory Data Analysis (EDA)**
- **Data cleaning & missing-value handling**
- **Domain-driven feature engineering**
- **Statistical testing**
- **Outlier handling**
- **Feature scaling**
- **Final ML-ready dataset preparation**

> Rather than only training models, this repo emphasizes the most critical stage of any ML system: **feature engineering**.

---

## 📂 Projects in this Repository

| Project | Notebook | Problem Type | Focus | Link |
|---|---:|:---:|---|---|
| 🏥 **Insurance Cost Feature Engineering** | `Insurance.ipynb` | Regression | **Risk factors & cost-driven features** | [Open Insurance.ipynb](https://github.com/gee-46/feature-engineering/blob/main/Insurance.ipynb) |
| ❤️ **Heart Disease Feature Engineering** | `Heart.ipynb` | Classification | **Medical risk indicators & heart-health features** | [Open Heart.ipynb](https://github.com/gee-46/feature-engineering/blob/main/Heart.ipynb) |

---

## 🏥 Project 1 — Insurance Cost Analysis & Feature Engineering
**Goal:** Transform raw insurance records into a structured dataset optimized for regression models predicting medical charges.

**Dataset:** Per-person medical insurance data including:
`age, sex, bmi, children, smoker, region → charges (target)`

**Key work done**
- **Comprehensive EDA** (distributions, outliers, correlations)
- **Feature engineering**
  - Binary health indicators (e.g., **smoker** flag)
  - **One-hot encoding** for region
  - **BMI risk categorization** (underweight, normal, overweight, obese)
- **Statistical testing**
  - Pearson correlation (numeric relationships)
  - Chi-square tests (categorical dependencies)
- **Data standardization** using `StandardScaler`

**Outcome:** A cleaned and engineered dataset suitable for:
- **Linear Regression**
- **Random Forest**
- **Gradient Boosting / XGBoost**

---

## ❤️ Project 2 — Heart Disease Feature Engineering Pipeline
**Goal:** Engineer medically meaningful features to improve heart disease classification models.

**Dataset:** Clinical heart-health attributes including:
`Age, RestingBP, Cholesterol, MaxHR, FastingBS, Oldpeak, ECG results, chest pain type, exercise angina → HeartDisease (target)`

**Key work done**
- **Data cleaning & missing-value handling**
- **Boolean and numerical feature processing**
- **Medical domain-driven feature creation**
  - **High blood pressure** indicator
  - **High cholesterol** indicator
  - **Low maximum heart rate** flag
  - **Heart stress index**
  - **Age risk banding**
- **Outlier handling** using IQR
- **Feature scaling** using `StandardScaler`
- **Feature selection** using statistical methods

**Outcome:** A structured dataset optimized for:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machines**
- **Neural Networks**

---

## 🛠 Core Techniques Demonstrated
- **Exploratory Data Analysis (EDA)**
- **Feature encoding** (binary, one-hot, binning)
- **Domain-based feature engineering**
- **Missing-value imputation**
- **Outlier treatment (IQR)**
- **Feature scaling**
- **Statistical validation**
- **Feature selection**
- **ML-ready pipeline design**

---

## 🗂 Suggested Repository Structure
```
Feature-Engineering/
│
├── Insurance.ipynb
├── Heart.ipynb
├── data/
│   ├── insurance.csv
│   └── heart.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ How to Run Locally
```bash
git clone https://github.com/gee-46/feature-engineering.git
cd feature-engineering

# Create environment & install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# Launch notebook server
jupyter notebook
```

Open either:
- **Insurance.ipynb**
- **Heart.ipynb**

---

## 🎯 Why this repository?
This repo demonstrates the real foundation of machine learning systems — **turning messy, real-world medical data into reliable, interpretable, machine-learning-ready features**.

It is intended as a practical reference for:
- **ML beginners** learning preprocessing
- **Students** building strong project portfolios
- **Healthcare-focused ML experimentation**

---

## 🚀 Future Work
- **End-to-end** model training & evaluation notebooks
- **Cross-validation & hyperparameter tuning**
- **Feature-importance & SHAP analysis**
- **Deployment-ready preprocessing pipelines**
- **Streamlit demo apps**

---

## 👤 Author
**Gautam N Chipkar**  
AI & Data Science Engineering Student  
**Python • Data Analysis • Machine Learning**

---

## 📜 License
This project is released under the **MIT License**. See the `LICENSE` file for details.
