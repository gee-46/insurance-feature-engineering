# 🏥 Insurance Cost Analysis & Feature Engineering
Real-World Exploratory Data Analysis • Statistical Insights • ML-Ready Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

A focused data-analysis project that inspects factors influencing medical insurance charges, applies feature engineering, performs statistical testing, and produces a cleaned, ML-ready dataset for [...]

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [What I Built](#what-i-built)
- [Notebook & Project Structure](#notebook--project-structure)
- [How to Run (Locally)](#how-to-run-locally)
- [Key Methods & Techniques](#key-methods--techniques)
- [Key Findings & Insights](#key-findings--insights)
- [Deliverables](#deliverables)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)

---

## Project Overview

This project represents the foundational stage of a machine learning pipeline. The notebook transforms raw insurance data into a structured, ML-ready format through advanced EDA, feature engineering, statistical validation, and standardization. These steps are critical for building reliable predictive models. The resulting dataset is optimized for training regression algorithms to estimate medical insurance charges, positioning this project as a practical bridge between data analysis and machine learning deployment.

Insurance costs reflect demographic, lifestyle, and regional factors. This project investigates a real-world insurance dataset to:
- Discover patterns and distributions
- Engineer features that better capture health & risk signals
- Validate relationships with statistical tests
- Produce a cleaned, scaled dataset ready for regression models

Use case: Preprocessing step for predicting insurance charges using machine learning.

---

## Dataset
Source file: `insurance.csv`

Short answer — it's a per-person medical-insurance dataset used to build and evaluate models that predict insurance charges. Each row is an individual and the columns are demographic / health features (age, sex, BMI, number of children, smoker status, region) plus the target charges. Typical uses:

Exploratory data analysis to understand distributions and relationships
Feature engineering (e.g., BMI bins, smoker flag, one-hot regions)
Statistical tests (correlations, chi-square)
Training regression models to predict individual insurance cost (linear regression, random forest, XGBoost, etc.)
Teaching / demos because it's small and interpretable

Columns:
- `age` — age of the individual
- `sex` — male / female
- `bmi` — body mass index
- `children` — number of dependents
- `smoker` — yes / no
- `region` — residential region
- `charges` — insurance cost (target)

---

## What I Built
- Comprehensive Exploratory Data Analysis (EDA)
  - Distributions, boxplots, and countplots for key variables
- Feature engineering
  - Binary encodings: `is_female`, `is_smoker`
  - Region one-hot encoding: `region_northwest`, `region_southeast`, `region_southwest`, `region_northeast` (or equivalent)
  - BMI categorical bins: `underweight`, `normal`, `overweight`, `obese`
- Data cleaning & transformation
  - Removed redundant categorical columns and kept model-friendly numeric features
- Feature scaling
  - StandardScaler applied to numeric features (age, bmi, children, charges)
- Statistical analysis
  - Pearson correlation (numeric relationships)
  - Chi-square tests (categorical dependencies)
- Final ML-ready dataset prepared for regression experiments

---

## Notebook & Project Structure
```
Insurance-Analysis/
├─ Insurance.ipynb        # Main Jupyter notebook (EDA, FE, analysis)
├─ insurance.csv          # Source dataset
└─ README.md              # This file
```

Notebook sections:
1. Data loading & inspection
2. EDA (plots and descriptive stats)
3. Feature engineering & encoding
4. Data cleaning & scaling
5. Statistical testing (Pearson, Chi-square)
6. Final dataset export and summary

---

## How to Run (Locally)

1. Clone the repo
   ```bash
   git clone https://github.com/gee-46/ML-2-.git
   cd ML-2-
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS / Linux
   venv\Scripts\activate       # Windows

   pip install -r requirements.txt
   ```
   If there is no requirements file, install the typical stack:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
   ```

3. Start Jupyter and open the notebook
   ```bash
   jupyter notebook Insurance.ipynb
   ```

4. Inspect or re-run cells to reproduce preprocessing and export the final dataset.

Quick snippet to preview the dataset:
```python
import pandas as pd
df = pd.read_csv("insurance.csv")
df.head()
```

---

## Key Methods & Techniques
- EDA: histograms, KDEs, boxplots, countplots
- Feature engineering: binary flags, one-hot encoding, BMI bucketing
- Scaling: StandardScaler for numeric features
- Statistical testing:
  - Pearson correlation matrix for numeric features vs `charges`
  - Chi-square test for categorical feature dependency on binned charges
- Output: cleaned DataFrame with only numeric/encoded columns + target

---

## Key Findings & Insights (Summary)
- Smoking status is typically one of the strongest predictors of charges.
- Age and BMI exhibit positive relationships with insurance charges; higher age/BMI commonly associates with higher charges.
- Region and sex show smaller but potentially meaningful effects after encoding.
- Engineered features (BMI categories, smoker flag, region one-hot) improve interpretability and model input quality.

Note: Specific numeric results, charts, and p-values are available in `Insurance.ipynb` for reproducibility.

---

## Deliverables
- `Insurance.ipynb` — full analysis, visualizations, and transformations
- `insurance.csv` — original dataset
- A final cleaned, ML-ready DataFrame exported (e.g., `insurance_cleaned.csv`) — check the notebook for the exact filename

---

## Future Work
- Train regression models: Linear Regression, Random Forest, XGBoost
- Hyperparameter tuning & cross-validation
- Model evaluation: R², MAE, RMSE
- Feature importance and SHAP analysis
- Build a Streamlit app for interactive predictions and demo

---

## Author
Gautam N Chipkar  
AI & Data Science Engineering Student — Python | Data Analysis | Machine Learning

---

## License
This project is released under the MIT License. See [LICENSE](./LICENSE) for details.
