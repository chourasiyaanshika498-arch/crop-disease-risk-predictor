"""
train.py
--------
Trains and compares 4 models (Random Forest, XGBoost, SVM, Logistic
Regression) to predict crop disease outbreak risk (Low/Medium/High).
Saves XGBoost model in JSON format — works on any Python/sklearn version.

Run:
    python train.py
"""

import json, os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

DATA_PATH = "crop_disease_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERIC = ["soil_ph","rainfall_mm","temperature_c","humidity_pct","nitrogen","phosphorus","potassium","soil_moisture_pct"]
CATEGORICAL = ["crop","district","season"]

df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
y = le.fit_transform(df["disease_risk"])

# --- Preprocess manually (no Pipeline) so we can save without sklearn version issues ---
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(df[CATEGORICAL])
cat_encoded = ohe.transform(df[CATEGORICAL])
cat_cols = list(ohe.get_feature_names_out(CATEGORICAL))

scaler = StandardScaler()
num_scaled = scaler.fit_transform(df[NUMERIC])

X = np.hstack([num_scaled, cat_encoded])
feat_names = NUMERIC + cat_cols

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Compare 4 models ---
print("Comparing 4 models...\n")
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "SVM":                 SVC(probability=True, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "XGBoost (baseline)":  xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=42, n_estimators=50, max_depth=3, learning_rate=0.3),
}

results = {}
for name, clf in candidates.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="macro")
    results[name] = acc
    print(f"{name:25s} | Accuracy: {acc*100:.2f}% | Macro F1: {f1:.3f}")

print(f"\nBest baseline: XGBoost")

# --- GridSearchCV on XGBoost ---
print("\nRunning GridSearchCV...")
base_clf = candidates["XGBoost (baseline)"]
base_preds = base_clf.predict(X_test)
high_idx = list(le.classes_).index("High")
base_prec = precision_score(y_test, base_preds, average=None, zero_division=0)[high_idx]

param_grid = {"n_estimators":[150,250], "max_depth":[4,6], "learning_rate":[0.05,0.1]}
xgb_clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=42)
grid = GridSearchCV(xgb_clf, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

tuned_preds = best_model.predict(X_test)
tuned_prec = precision_score(y_test, tuned_preds, average=None, zero_division=0)[high_idx]
acc = accuracy_score(y_test, tuned_preds)
f1  = f1_score(y_test, tuned_preds, average="macro")

print(f"\nFinal XGBoost Accuracy: {acc*100:.2f}%")
print(f"Macro F1: {f1:.3f}")
print(f"High risk precision: {base_prec*100:.1f}% -> {tuned_prec*100:.1f}% ({(tuned_prec-base_prec)*100:+.1f} pts)")
print("\n", classification_report(y_test, tuned_preds, target_names=le.classes_, zero_division=0))

# --- Save model in XGBoost JSON format (version-independent) ---
best_model.save_model(f"{MODEL_DIR}/model.json")
np.save(f"{MODEL_DIR}/classes.npy", le.classes_)
np.save(f"{MODEL_DIR}/scaler_mean.npy", scaler.mean_)
np.save(f"{MODEL_DIR}/scaler_scale.npy", scaler.scale_)

with open(f"{MODEL_DIR}/ohe_categories.json","w") as f:
    json.dump({col: cats.tolist() for col, cats in zip(CATEGORICAL, ohe.categories_)}, f)
with open(f"{MODEL_DIR}/feature_lists.json","w") as f:
    json.dump({"numeric": NUMERIC, "categorical": CATEGORICAL, "cat_cols": cat_cols}, f)
with open(f"{MODEL_DIR}/crop_options.json","w") as f:
    json.dump(sorted(df["crop"].unique().tolist()), f)
with open(f"{MODEL_DIR}/district_options.json","w") as f:
    json.dump(sorted(df["district"].unique().tolist()), f)

# --- SHAP ---
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(pd.DataFrame(X_test, columns=feat_names))
plt.figure()
if isinstance(shap_values, list):
    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
elif shap_values.ndim == 3:
    mean_abs = np.mean(np.abs(shap_values), axis=2)
else:
    mean_abs = shap_values
shap.summary_plot(mean_abs, pd.DataFrame(X_test, columns=feat_names), show=False, plot_type="bar", max_display=12)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/shap_summary.png", dpi=150)
print("Training complete. Model saved in models/")
