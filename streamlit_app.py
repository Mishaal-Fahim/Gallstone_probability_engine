import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------
# Load pre-trained model
# ---------------------------
# Your pipeline + model is inside the model folder
pipe = joblib.load("model/gallstone_pipeline.pkl")  

# SHAP explainer
explainer = shap.TreeExplainer(pipe.named_steps["xgb"])

# ---------------------------
# App title
# ---------------------------
st.title("Gallstone Risk Index Calculator 🩺")

# ---------------------------
# Sidebar: Patient input
# ---------------------------
st.sidebar.header("Enter Patient Data")

# Automatically get feature names from your model
feature_names = pipe.named_steps["prep"].transformers_[0][2]  # numeric features
# If you have categorical features, add them here as well

# Collect user input for each feature
patient_input = {}
for feature in feature_names:
    # example: you can customize ranges based on your dataset
    patient_input[feature] = st.sidebar.number_input(
        feature, 
        value=float(0),  # default value
        step=1.0
    )

patient_df = pd.DataFrame([patient_input])

# ---------------------------
# Predict Gallstone Risk Score
# ---------------------------
score_01 = pipe.predict_proba(patient_df)[:, 1][0]
score_100 = score_01 * 100

# Determine risk band
if score_100 < 33:
    risk = "Low"
elif score_100 < 66:
    risk = "Medium"
else:
    risk = "High"

st.subheader("Gallstone Risk Score")
st.write(f"Score (0-100): **{score_100:.2f}**")
st.write(f"Risk Band: **{risk}**")

# ---------------------------
# SHAP explanations
# ---------------------------
st.subheader("Feature Contribution (SHAP)")

# Preprocess input for SHAP
X_preprocessed = pipe.named_steps["prep"].transform(patient_df)
shap_values = explainer.shap_values(X_preprocessed)

# Summary plot for this patient
fig, ax = plt.subplots(figsize=(6,3))
shap.bar_plot(shap_values[0], feature_names=feature_names)
st.pyplot(fig)
