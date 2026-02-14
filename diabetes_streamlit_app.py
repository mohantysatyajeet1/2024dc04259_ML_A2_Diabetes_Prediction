import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Users\91956\Desktop\ML2_Diabetes_Prediction\best_model.pkl")
    model_name = "Random Forest"
    return model

model = load_model()

# The feature order must match how you trained your model
FEATURE_COLUMNS = [
    'Gender',
    'AGE',
    'Urea',
    'Cr',
    'HbA1c',
    'Chol',
    'TG', 
    'HDL',
    'LDL',
    'VLDL',
    'BMI'
]

st.title("🏥 Diabetes Prediction App")

st.write(
    "Enter patient details on the left and click **Predict** to see whether you are Diabetic."
)

st.sidebar.header("Patient Details")

# -----------------------------
# Inputs
# -----------------------------
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

age = st.sidebar.number_input(
    "Patient Age (in year)", min_value=20.0, value=80.0, step=1.0
)
urea = st.sidebar.number_input(
    "urea (in mg/dl)", min_value=0.5, value=40.0, step=0.10
)
cr = st.sidebar.number_input(
    "Cr (in mg/dl)", min_value=1.0, value=900.0, step=1.0,
)
hba1c = st.sidebar.number_input(
    "HbA1c (in %)", min_value=0.5, value=20.0, step=0.10
)
chol = st.sidebar.number_input(
    "Chol (in mg/dl)", min_value=0.0, value=15.0, step=0.10
)
tg = st.sidebar.number_input(
    "Tg (in mg/dl)", min_value=0.0, value=15.0, step=0.10
)
hdl = st.sidebar.number_input(
    "HDL (in mg/dl)", min_value=0.0, value=10.0, step=0.10
)
ldl = st.sidebar.number_input(
    "LDL (in mg/dl)", min_value=0.0, value=10.0, step=0.10
)
vldl = st.sidebar.number_input(
    "VLDL (in mg/dl)", min_value=1.0, value=40.0, step=0.10
)
bmi = st.sidebar.number_input(
    "BMI (in kg/m2)", min_value=18.0, value=50.0, step=0.10
)

# -----------------------------
# Feature engineering helpers
# -----------------------------
def safe_log(x):
    # avoid log(0) or log of negative
    return np.log(max(x, 1e-3))

def build_feature_vector():
   
    # FORCE numeric encoding
    if gender == "Male":
        gender_val = 1
    else:
        gender_val = 0

    data = {
        'Gender': gender_val,
        'AGE': int(age),
        'Urea': float(urea),
        'Cr': float(cr),
        'HbA1c': float(hba1c),
        'Chol': float(chol),
        'TG': float(tg),
        'HDL': float(hdl),
        'LDL': float(ldl),
        'VLDL': float(vldl),
        'BMI': float(bmi)
     }

    X = pd.DataFrame([data])

    return X

    # Ensure correct column order
    X_input = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    return X_input

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Am I Diabetic")

if st.button("🔍 Predict"):
    X_input = build_feature_vector()

    # Model prediction: assumes 1 = Yes, 0 = No
    pred = model.predict(X_input)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0][1]  # probability of class 1
    else:
        proba = None

    if pred == 1:
        st.success("🚨 **YES**")
    else:
        st.error("💚 **NO**")

    if proba is not None:
        st.write(f"**probability is:** {proba*100:.1f}%")

    st.write("### Input Features Sent to Model")
    st.dataframe(build_feature_vector())
else:
    st.info("Fill the details on the left and click **Predict**.")
