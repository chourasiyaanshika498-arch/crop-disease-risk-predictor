# 🌾 Crop Disease Risk Predictor

An explainable Machine Learning system that predicts **crop disease outbreak risk (Low /Medium / High)** for major Indian crops using soil and weather conditions.

The project compares multiple ML algorithms, selects the best-performing model using **GridSearchCV**, explains every prediction using **SHAP**, and deploys the solution through an interactive **Streamlit** web application.

---

# Features

- Predict disease risk for 10+ Indian crops
- District-specific predictions
- Weather + soil based inference
- XGBoost model optimized using GridSearchCV
- SHAP explanations for every prediction
- Farmer-friendly Streamlit interface
- Confidence score for each prediction

---

# Workflow

User Input
(Crop + District + Season + Soil + Weather)

↓

Preprocessing

↓

Model Prediction (XGBoost)

↓

Risk Classification

↓

SHAP Explainability

↓

Prediction + Confidence + Important Features

---

# Tech Stack

- Python
- XGBoost
- SHAP
- Scikit-learn
- Pandas
- NumPy
- Streamlit

---

# Dataset

Since no publicly available dataset directly maps soil and weather measurements to crop disease outbreak risk across multiple Indian crops, this project generates a synthetic dataset based on agronomic relationships.

The dataset incorporates realistic assumptions such as:

- Higher humidity increases fungal disease risk
- Poor N-P-K levels reduce crop resistance
- Soil pH deviation stresses crops
- Excess rainfall raises infection probability

Randomized noise is added to improve diversity while preserving realistic relationships.

**Dataset Size**

- 5,000 samples
- 10+ crops
- 10 districts
- Soil attributes
- Weather attributes

---

# Model Comparison

| Model | Accuracy |
|---------|----------|
| Logistic Regression | 73% |
| Random Forest | 84% |
| Support Vector Machine | 86% |
| **XGBoost** | **90%** |

XGBoost achieved the highest validation accuracy and better performance on the minority **High Risk** class after hyperparameter tuning using **GridSearchCV**, making it the final deployed model.

---

# Explainable AI

Instead of acting as a black box, the model explains every prediction using **SHAP (SHapley Additive Explanations)**.

Users can see:

- Which features increased disease risk
- Which features lowered disease risk
- Relative contribution of each feature

This improves transparency and trust in model predictions.

---

# Project Structure

```
crop-disease-risk-predictor/
│
├── generate_dataset.py
├── train.py
├── app.py
├── crop_disease_data.csv
├── models/
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone <repo-url>
cd crop-disease-risk-predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Generate dataset

```bash
python generate_dataset.py
```

Train model

```bash
python train.py
```

Launch Streamlit app

```bash
streamlit run app.py
```

---

# Future Improvements

- Train on real ICAR and IMD datasets
- Satellite imagery integration
- Time-series disease forecasting
- Cloud deployment
- Mobile application
- Disease-specific recommendations

---

# Built With

- Python
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Pandas

AI • Machine Learning • Explainable AI • Agritech
