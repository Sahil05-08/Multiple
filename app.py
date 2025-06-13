import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaled.joblib")

# App title
st.title("Income Prediction App")
st.write("Enter the values for the features below to predict income.")

# Input fields for features
crim = st.number_input("Crime Rate (crim)", value=0.0)
zn = st.number_input("Zoning Proportion (zn)", value=0.0)
indus = st.number_input("Industrial Proportion (indus)", value=0.0)
chas = st.selectbox("Bounds River (chas)", options=[0, 1])  # Binary
nox = st.number_input("Nitric Oxide Concentration (nox)", value=0.0)
rm = st.number_input("Average Rooms (rm)", value=0.0)
age = st.number_input("Property Age (age)", value=0.0)
dis = st.number_input("Distance to Employment Centers (dis)", value=0.0)
rad = st.number_input("Accessibility to Highways (rad)", value=0.0)
tax = st.number_input("Property Tax Rate (tax)", value=0.0)
ptratio = st.number_input("Pupil-Teacher Ratio (ptratio)", value=0.0)
b = st.number_input("Proportion of Black Population (b)", value=0.0)
istat = st.number_input("Index Statistic (istat)", value=0.0)

# Collect inputs
features = np.array([[crim, zn, indus, chas, nox, rm, age, dis,
                      rad, tax, ptratio, b, istat]])

# Predict
if st.button("Predict Income"):
    scaled_input = scaler.transform(features)
    prediction = model.predict(scaled_input)
    st.success(f"Predicted Income: ${prediction[0]:,.2f}")
