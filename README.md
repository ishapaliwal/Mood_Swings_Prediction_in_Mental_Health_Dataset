# Mood Swings Prediction in Mental Health Dataset

This project explores the use of machine learning techniques to predict **mood swings** using a comprehensive **Mental Health dataset** from Kaggle. It aims to bridge the gap between subjective diagnosis and objective, data-driven mental health insights.

---

## Problem Statement

Mood swings are a core symptom of several mental health conditions like **bipolar disorder**, **depression**, and **borderline personality disorder**. Accurate, timely prediction is crucial for effective intervention — yet current diagnostic methods rely heavily on subjective evaluation.

**Goal:** Develop a machine learning model that can **predict mood swing levels** using empirical mental health data.

---

## Dataset

- **Source**: [Kaggle - Mental Health Dataset](https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset)
- **Key Features**:
  - Demographics: `Gender`, `Country`, `Occupation`, `Self-employed`
  - Indicators: `Mood_Swings`, `Coping_Struggles`, `Growing_Stress`, `Social_Weakness`
  - Treatment History & Mental Health Background

---

## Methodology

### Data Preprocessing
- Removed duplicates and irrelevant columns
- Handled missing values (e.g., imputed `self_employed`)
- Used `LabelEncoding`, `OneHotEncoding`, and `LeaveOneOutEncoding`

### EDA
- Visualized distributions and unique values
- Explored correlations with mood swings

### Model Development
- Used `MultiOutputClassifier` for predicting multiple outputs
- Base Models Evaluated:
  - `LinearSVC`
  - `Logistic Regression`
  - `Naive Bayes`
  - `Random Forest`
  - **`XGBoost` (Selected)**

### Evaluation
- Metrics: Accuracy, Precision, Recall, ROC AUC
- Cross-validation used to ensure generalization

---

## Results

- **Best Model**: `XGBoost`
- High accuracy in classifying multiple mood swing levels
- Validated with unseen test data

You can view the project results in the following formats:
- [Jupyter Notebook](https://github.com/ishapaliwal/Mood_Swings_Prediction_in_Mental_Health_Dataset/blob/master/Notebooks/Final%20Project%20Submission_Paliwal_Isha.ipynb) (in the `notebooks/` directory) – includes data exploration, model training, and evaluation
- Project Presentation – [Mood Swings Prediction Project Presentation (PPTX)](https://github.com/ishapaliwal/Mood_Swings_Prediction_in_Mental_Health_Dataset/blob/master/Mood%20Swings%20Prediction%20Project%20Presentation.pptx)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/mood-swings-prediction.git
cd mood-swings-prediction
pip install -r requirements.txt
```

---

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- xgboost
- category_encoders
- matplotlib / seaborn (for EDA)
- jupyter (optional, for notebooks)

---

## Future Work

- Expand feature set with real-time data (e.g., wearables)
- Apply SHAP or LIME for model interpretability
- Build a live monitoring dashboard
- Consider clinical deployment with secure data integration
