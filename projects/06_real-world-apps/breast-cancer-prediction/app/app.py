
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---- LOAD MODEL AND FEATURE ORDER SAFELY ----
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, '..', 'models', 'xgboost_model.pkl')
feature_path = os.path.join(BASE_DIR, '..', 'models', 'feature_order.pkl')

try:
    model = joblib.load(model_path)
    feature_order = joblib.load(feature_path)
except Exception as e:
    st.error(f"❌ Failed to load model or feature list:
{e}")
    st.stop()

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="Breast Cancer Mortality Risk Prediction", layout="centered")
st.title("🔬 METABRIC Breast Cancer Mortality Risk Prediction")
st.markdown("This tool estimates the **mortality risk** for breast cancer patients based on clinical and gene expression data.")

# ---- CLINICAL INPUTS ----
st.subheader("📋 Clinical Information")

chemotherapy = st.selectbox("Received chemotherapy?", [0, 1])
hormone_therapy = st.selectbox("Received hormone therapy?", [0, 1])
radio_therapy = st.selectbox("Received radiotherapy?", [0, 1])
grade = st.selectbox("Histologic tumor grade", [1, 2, 3])
tumor_stage = st.selectbox("Tumor stage", ['I', 'II', 'III', 'IV'])

# ---- GENE EXPRESSION INPUTS ----
st.subheader("🧬 Gene Expression Levels (Z-score normalized)")

tp53 = st.slider("TP53", -3.0, 3.0, 0.0)
esr1 = st.slider("ESR1", -3.0, 3.0, 0.0)
brca1 = st.slider("BRCA1", -3.0, 3.0, 0.0)

# ---- PREPARE INPUT DICT ----
input_dict = {
    "chemotherapy": chemotherapy,
    "hormone_therapy": hormone_therapy,
    "radio_therapy": radio_therapy,
    "neoplasm_histologic_grade": grade,
    "tumor_stage_II": int(tumor_stage == 'II'),
    "tumor_stage_III": int(tumor_stage == 'III'),
    "tumor_stage_IV": int(tumor_stage == 'IV'),
    "TP53": tp53,
    "ESR1": esr1,
    "BRCA1": brca1
}

# ---- FORMAT FINAL INPUT ----
X_input = pd.DataFrame([input_dict])
for col in feature_order:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[feature_order]

# ---- PREDICT ----
if st.button("💡 Predict Mortality Risk"):
    try:
        proba = model.predict_proba(X_input)[0][1]
        st.success(f"📉 Estimated Mortality Risk: {proba*100:.2f}%")
        st.info(f"🫀 Estimated Survival Probability: {(1 - proba)*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed:
{e}")

