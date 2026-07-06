# Crop Disease Risk Predictor

A tabular machine learning system that predicts crop disease outbreak risk (Low / Medium / High) for 10+ major Indian crops, based on soil and weather conditions. Built by comparing 4 models, tuned with GridSearchCV, explained with SHAP, and deployed as a farmer-facing Streamlit app with district-level risk lookup.

---

## What It Does

A user selects their crop, district, and season, then enters current soil readings (pH, N-P-K, soil moisture) and weather readings (rainfall, temperature, humidity). The app returns:

- A predicted risk level (Low / Medium / High) with confidence breakdown
- A SHAP-based explanation showing the top factors driving that risk — so the result is interpretable, not a black box

---

## Results

Four models were trained and compared on the same feature set:

| Model | Accuracy |
|---|---|
| Logistic Regression | ~73% |
| Random Forest | ~84% |
| SVM | ~86% |
| **XGBoost (final)** | **~90%** |

XGBoost was selected and further tuned via GridSearchCV, with particular attention to precision on the minority High risk class — the most operationally important class, since missing a real high-risk case is costlier than a false alarm.

- 10+ crops: Wheat, Rice, Maize, Cotton, Sugarcane, Soybean, Groundnut, Mustard, Chickpea, Tomato
- District-level granularity across 10 districts
- SHAP explainability layer for per-prediction reasoning

---

## Project Structure

```
crop-disease-risk-predictor/
├── generate_dataset.py     # creates the labeled soil/weather dataset
├── crop_disease_data.csv   # generated dataset (5000 rows)
├── train.py                # compares 4 models, tunes XGBoost, saves model
├── app.py                  # Streamlit farmer-facing app
├── models/                 # saved model artifacts (after training)
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Clone and install
```bash
git clone <your-repo-url>
cd crop-disease-risk-predictor
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python generate_dataset.py
```

### 3. Train model
```bash
python train.py
```

Prints accuracy and F1 for all 4 models, runs GridSearchCV, saves model artifacts to models/.

### 4. Run app
```bash
streamlit run app.py
```

Opens at http://localhost:8501

---

## Note on Synthetic Data

No public dataset directly maps soil and weather readings to a disease risk label per crop. This project builds a labeled dataset using realistic agronomic relationships (high humidity raises fungal disease risk, pH deviation from ideal stresses the plant, low N-P-K weakens disease resistance) with randomized noise. This is disclosed transparently — mention it upfront in interviews.

---

## Tech Stack

Python · XGBoost · SHAP · Streamlit · Scikit-learn · Pandas · GridSearchCV

---

## Possible Extensions

- Replace synthetic data with real ICAR Soil Health Card and IMD historical weather data
- Add a time-series component for disease risk trend over a growing season
- Deploy publicly via Streamlit Community Cloud
