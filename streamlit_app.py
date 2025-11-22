# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------
# Load pre-trained model and weights
# ----------------------------------------
# Model folder contains: gallstone_model.pkl and top_features.pkl
model = joblib.load("model/gallstone_model.pkl")
top_features = joblib.load("model/top_features.pkl")  # DataFrame with Feature + Weight

# List of top features and their weights
top_feature_names = list(top_features['Feature'])
weights = top_features.set_index('Feature')['Weight']

# ----------------------------------------
# Helper functions
# ----------------------------------------
def normalize_feature(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def compute_score(input_df, top_features, weights, min_vals, max_vals):
    scaled = pd.DataFrame()
    for feature in top_features:
        scaled[feature] = normalize_feature(input_df[feature], min_vals[feature], max_vals[feature])
    scaled['Gallstone_Score_0_1'] = scaled[top_features].dot(weights)
    scaled['Gallstone_Score_0_100'] = scaled['Gallstone_Score_0_1'] * 100
    
    # Risk Band
    def risk_band(score):
        if score < 33:
            return "Low"
        elif score < 66:
            return "Medium"
        else:
            return "High"
    scaled['Risk_Band'] = scaled['Gallstone_Score_0_100'].apply(risk_band)
    return scaled

def digital_twin(patient_df, top_features, weights, min_vals, max_vals, top_n=3, percent_change=0.1):
    sim_data = patient_df.copy()
    # Top N features by weight
    top_n_features = weights.sort_values(ascending=False).head(top_n).index
    changes = {}
    for feature in top_n_features:
        delta = (max_vals[feature] - min_vals[feature]) * percent_change
        changes[feature] = sim_data[feature].values[0] - delta
        sim_data[feature] = changes[feature]
    sim_score = compute_score(sim_data, top_features, weights, min_vals, max_vals)
    return sim_score, changes

# ----------------------------------------
# Streamlit App
# ----------------------------------------
st.set_page_config(page_title="Gallstone Probability Engine", layout="wide")

st.title("🩺 Gallstone Probability Engine")
st.markdown("""
Predict your **Gallstone Risk Score** (0–100) and visualize how changes in key factors affect your risk.
""")

# ----------------------------------------
# Sidebar: Patient Input
# ----------------------------------------
st.sidebar.header("Patient Input")

# Create a sample input based on min/max values from training dataset
min_vals = pd.Series({f: 0 for f in top_feature_names})  # Replace with real min from dataset
max_vals = pd.Series({f: 100 for f in top_feature_names})  # Replace with real max from dataset

patient_input = {}
for feature in top_feature_names:
    # You can adjust the min/max values as per your dataset
    patient_input[feature] = st.sidebar.slider(feature, float(min_vals[feature]), float(max_vals[feature]), float((min_vals[feature]+max_vals[feature])/2))

patient_df = pd.DataFrame([patient_input])

# ----------------------------------------
# Compute Original Score
# ----------------------------------------
original_score_df = compute_score(patient_df, top_feature_names, weights, min_vals, max_vals)
original_score = original_score_df['Gallstone_Score_0_100'].values[0]
original_risk = original_score_df['Risk_Band'].values[0]

st.subheader("Original Gallstone Risk")
st.metric(label="Score (0-100)", value=f"{original_score:.2f}", delta=None)
st.write(f"Risk Band: **{original_risk}**")

# ----------------------------------------
# Digital Twin Simulation
# ----------------------------------------
st.subheader("Digital Twin Simulation")
top_n = st.sidebar.slider("Number of features to simulate change", 1, min(5, len(top_feature_names)), 3)
percent_change = st.sidebar.slider("Percentage change for simulation", 1, 50, 10) / 100.0

sim_score_df, applied_changes = digital_twin(patient_df, top_feature_names, weights, min_vals, max_vals, top_n, percent_change)
sim_score = sim_score_df['Gallstone_Score_0_100'].values[0]
sim_risk = sim_score_df['Risk_Band'].values[0]

st.metric(label="Simulated Score (after changes)", value=f"{sim_score:.2f}")
st.write(f"Simulated Risk Band: **{sim_risk}**")

# ----------------------------------------
# Plot Comparison
# ----------------------------------------
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(['Original', 'Simulated'], [original_score, sim_score], color=['blue', 'green'])
ax.set_ylabel("Gallstone Score (0-100)")
ax.set_title("Digital Twin: Original vs Simulated Score")
st.pyplot(fig)

# ----------------------------------------
# Show Applied Changes
# ----------------------------------------
st.subheader("Applied Changes in Features")
st.write(applied_changes)

# ----------------------------------------
# Optional: Show top feature weights
# ----------------------------------------
st.subheader("Top Feature Contributions (Weights)")
st.dataframe(top_features)

