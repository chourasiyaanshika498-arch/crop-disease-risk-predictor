"""
app.py
------
Farmer-facing Streamlit app for Crop Disease Risk Prediction.
Uses XGBoost JSON model — no sklearn version dependency.

Run:
    streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Disease Risk Predictor", layout="wide")

MODEL_DIR = "models"


@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(f"{MODEL_DIR}/model.json")
    classes = np.load(f"{MODEL_DIR}/classes.npy", allow_pickle=True)
    scaler_mean = np.load(f"{MODEL_DIR}/scaler_mean.npy")
    scaler_scale = np.load(f"{MODEL_DIR}/scaler_scale.npy")
    with open(f"{MODEL_DIR}/ohe_categories.json") as f:
        ohe_cats = json.load(f)
    with open(f"{MODEL_DIR}/feature_lists.json") as f:
        fl = json.load(f)
    with open(f"{MODEL_DIR}/crop_options.json") as f:
        crops = json.load(f)
    with open(f"{MODEL_DIR}/district_options.json") as f:
        districts = json.load(f)
    return model, classes, scaler_mean, scaler_scale, ohe_cats, fl, crops, districts


model, classes, scaler_mean, scaler_scale, ohe_cats, fl, crops, districts = load_artifacts()
NUMERIC = fl["numeric"]
CATEGORICAL = fl["categorical"]
CAT_COLS = fl["cat_cols"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
RISK_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
RISK_BG = {"Low": "#0d2618", "Medium": "#2a1f00", "High": "#2a0d0d"}


def preprocess(soil_ph, rainfall_mm, temperature_c, humidity_pct,
               nitrogen, phosphorus, potassium, soil_moisture_pct,
               crop, district, season):
    num = np.array([[soil_ph, rainfall_mm, temperature_c, humidity_pct,
                     nitrogen, phosphorus, potassium, soil_moisture_pct]])
    num_scaled = (num - scaler_mean) / scaler_scale
    cat_vals = [crop, district, season]
    cat_enc = []
    for i, col in enumerate(CATEGORICAL):
        for cat in ohe_cats[col]:
            cat_enc.append(1.0 if cat_vals[i] == cat else 0.0)
    return np.hstack([num_scaled, np.array([cat_enc])])


# --- UI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }
.hero { background: linear-gradient(135deg,#1a2332,#0d1b2a); border:1px solid #2d3748; border-radius:16px; padding:2rem 2.5rem; margin-bottom:2rem; }
.hero h1 { font-size:1.9rem; font-weight:700; color:#fff; margin:0 0 0.4rem 0; }
.hero p { color:#8892a4; font-size:0.92rem; margin:0; line-height:1.6; }
.sec { font-size:0.7rem; font-weight:600; color:#4a90d9; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.8rem; margin-top:1.2rem; }
.stButton > button { background:linear-gradient(135deg,#4a90d9,#357abd); color:white; border:none; border-radius:10px; padding:0.75rem 2.5rem; font-size:0.95rem; font-weight:600; width:100%; }
label { color:#a0aec0 !important; font-size:0.84rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Crop Disease Risk Predictor</h1>
  <p>District-level disease outbreak risk for 10 major Indian crops based on soil composition and weather conditions. Powered by XGBoost with SHAP-based explainability.</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="sec">Crop & Location</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: crop = st.selectbox("Crop", crops)
    with c2: district = st.selectbox("District", districts)
    with c3: season = st.selectbox("Season", SEASONS)

    st.markdown('<div class="sec">Soil Conditions</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        soil_ph = st.slider("Soil pH", 4.5, 9.0, 6.7, 0.1)
        nitrogen = st.slider("Nitrogen N", 0, 100, 45)
        phosphorus = st.slider("Phosphorus P", 0, 80, 25)
    with s2:
        potassium = st.slider("Potassium K", 0, 80, 25)
        soil_moisture_pct = st.slider("Soil Moisture %", 5, 90, 35)

    st.markdown('<div class="sec">Weather Conditions</div>', unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        rainfall_mm = st.number_input("Rainfall (mm)", 0, 2500, 900)
        temperature_c = st.slider("Temperature (C)", 10, 45, 28)
    with w2:
        humidity_pct = st.slider("Humidity %", 15, 100, 60)

    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("Analyze Disease Risk")

with col_right:
    if btn:
        X = preprocess(soil_ph, rainfall_mm, temperature_c, humidity_pct,
                       nitrogen, phosphorus, potassium, soil_moisture_pct,
                       crop, district, season)
        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]
        color = RISK_COLORS.get(pred_label, "#fff")
        bg = RISK_BG.get(pred_label, "#1a2332")

        st.markdown('<div class="sec">Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{bg};border:1px solid #2d3748;border-left:4px solid {color};
                    border-radius:12px;padding:1.5rem 2rem;margin-bottom:1rem">
            <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:2px;color:{color}">Disease Risk Level</div>
            <div style="font-size:2rem;font-weight:700;color:#fff">{pred_label} Risk</div>
            <div style="color:#8892a4;font-size:0.85rem">{probs[pred_idx]*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec">Probability Breakdown</div>', unsafe_allow_html=True)
        for cls, prob in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
            c = RISK_COLORS.get(cls, "#4a90d9")
            st.markdown(f"""
            <div style="background:#1a2332;border:1px solid #2d3748;border-radius:10px;
                        padding:0.8rem 1.2rem;margin-bottom:0.5rem;
                        display:flex;justify-content:space-between">
                <span style="color:#e2e8f0;font-size:0.9rem">{cls} Risk</span>
                <span style="color:{c};font-weight:600">{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob))

        st.markdown('<div class="sec">SHAP Explainability — Why This Prediction?</div>', unsafe_allow_html=True)
        feat_names = NUMERIC + CAT_COLS
        xdf = pd.DataFrame(X, columns=feat_names)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(xdf)
        if isinstance(sv, list): sv = sv[pred_idx][0]
        elif sv.ndim == 3: sv = sv[0, :, pred_idx]
        else: sv = sv[0]
        contrib = pd.Series(sv, index=feat_names).sort_values(key=abs, ascending=False).head(7)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#1a2332")
        ax.set_facecolor("#1a2332")
        bar_colors = [color if v > 0 else "#4a5568" for v in contrib.values[::-1]]
        ax.barh(contrib.index[::-1], contrib.values[::-1], color=bar_colors, height=0.6)
        ax.set_xlabel("SHAP Impact", color="#8892a4", fontsize=9)
        ax.tick_params(colors="#a0aec0", labelsize=9)
        for spine in ax.spines.values(): spine.set_color("#2d3748")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Colored bars push toward predicted risk level. Grey bars push toward lower risk.")
    else:
        st.markdown("""
        <div style="background:#1a2332;border:1px dashed #2d3748;border-radius:12px;
                    padding:4rem 2rem;text-align:center;margin-top:2rem">
            <div style="color:#4a5568;font-size:0.9rem;line-height:2">
                Configure soil and weather parameters on the left<br>
                then click <strong style="color:#4a90d9">Analyze Disease Risk</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.78rem;
            margin-top:3rem;border-top:1px solid #2d3748;padding-top:1.5rem">
    Built with XGBoost + SHAP — Compared against Random Forest, SVM, Logistic Regression — 10 major Indian crops
</div>
""", unsafe_allow_html=True)
