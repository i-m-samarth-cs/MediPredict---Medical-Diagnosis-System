import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load Models
models = {
    "Heart Disease": "heart_disease_model.pkl",
    "Diabetes": "diabetes_model.pkl",
    "Parkinson's": "parkinsons_model.pkl"
}

# Load Disease Information
disease_info = {
    "Heart": "Heart disease includes conditions like blocked blood vessels, arrhythmias, and congenital heart defects.",
    "Diabetes": "Diabetes is a chronic condition that affects how your body turns food into energy.",
    "Parkinson's": "Parkinson’s disease is a brain disorder that leads to shaking, stiffness, and difficulty with balance and coordination."
}

# Load Model Function
def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# UI Styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://source.unsplash.com/1600x900/?medical,hospital');
        background-size: cover;
        color: white;
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar - Disease Selection
st.sidebar.title("AI-Powered Medical Diagnosis")
selected_disease = st.sidebar.selectbox("Choose Disease to Diagnose", list(models.keys()))

# Load Selected Model
model_path = models[selected_disease]
model = load_model(model_path)

# Display Disease Info
st.title(f"{selected_disease} Diagnosis")
st.info(disease_info[selected_disease.split()[0]])  

# Input Fields
if selected_disease == "Heart Disease":
    st.subheader("Enter the following details:")
    age = st.number_input("Age", min_value=1, max_value=120, help="Patient's age in years")
    sex = st.selectbox("Sex", ["Male", "Female"], help="Biological sex of the patient")
    chest_pain = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="Type of chest pain experienced")
    bp = st.number_input("Resting Blood Pressure", help="Blood pressure in mmHg at rest")
    cholesterol = st.number_input("Serum Cholesterol", help="Cholesterol level in mg/dL")
    sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], help="Indicates if fasting blood sugar is above 120 mg/dL")
    
    # Prediction
    input_data = np.array([[age, 1 if sex == "Male" else 0, chest_pain, bp, cholesterol, sugar]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'}")

elif selected_disease == "Diabetes":
    st.subheader("Enter the following details:")
    glucose = st.number_input("Glucose Level", help="Blood sugar level in mg/dL")
    blood_pressure = st.number_input("Blood Pressure", help="Diastolic blood pressure in mmHg")
    skin_thickness = st.number_input("Skin Thickness", help="Thickness of skin in mm")
    insulin = st.number_input("Insulin Level", help="Insulin level in μU/mL")
    bmi = st.number_input("BMI", help="Body Mass Index")
    age = st.number_input("Age", min_value=1, max_value=120, help="Patient's age in years")

    # Prediction
    input_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi, age]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {'Diabetes Detected' if prediction[0] == 1 else 'No Diabetes'}")

elif selected_disease == "Parkinson's":
    st.subheader("Enter the following details:")
    jitter = st.number_input("Jitter (%)", help="Variation in voice frequency")
    shimmer = st.number_input("Shimmer (dB)", help="Variation in voice amplitude")
    hnr = st.number_input("Harmonics-to-Noise Ratio", help="Ratio of harmonic sound to noise")
    spread1 = st.number_input("Spread 1", help="Voice signal spread parameter")
    spread2 = st.number_input("Spread 2", help="Second spread parameter of voice")
    
    # Prediction
    input_data = np.array([[jitter, shimmer, hnr, spread1, spread2]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {'Parkinson’s Detected' if prediction[0] == 1 else 'No Parkinson’s'}")

