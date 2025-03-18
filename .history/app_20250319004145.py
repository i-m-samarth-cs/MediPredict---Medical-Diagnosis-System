import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# Streamlit Page Config
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide")

# Background Image
background_img = """
    <style>
    .stApp {
        background: url("https://wallpaperaccess.com/full/4112935.png");
        background-size: cover;
        background-position: center;
    }
    </style>
"""
st.markdown(background_img, unsafe_allow_html=True)

# Hide Streamlit menu
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Dataset paths
datasets = {
    "diabetes": "Datasets/diabetes_data.csv",
    "heart_disease": "Datasets/heart_disease_data.csv",
    "parkinsons": "Datasets/parkinson_data.csv",
    "lung_cancer": "Datasets/preprocessed_lungs_data.csv",
    "thyroid": "Datasets/preprocessed_hypothyroid.csv"
}

# Function to load and train models
def train_model(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
        df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
        
        target_column = df.columns[-1]  # Assume last column is target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model, scaler, list(X.columns)
    except Exception as e:
        st.warning(f"Error loading dataset {dataset_path}: {e}")
        return None, None, []

# Train models
models, scalers, features = {}, {}, {}
for disease, path in datasets.items():
    if os.path.exists(path):
        model, scaler, feature_names = train_model(path)
        if model:
            models[disease] = model
            scalers[disease] = scaler
            features[disease] = feature_names
    else:
        st.warning(f"Dataset not found: {path}")

# Sidebar Menu
selected = st.sidebar.selectbox(
    '🔹 Select a Disease to Predict',
    ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Lung Cancer Prediction', 'Hypo-Thyroid Prediction']
)

# Extracting disease key
disease_key = selected.replace(" Prediction", "").lower().replace(" ", "_")

# Disease Information
disease_info = {
    "diabetes": "A chronic condition that affects how your body processes blood sugar (glucose).",
    "heart_disease": "A range of heart conditions affecting heart function, commonly due to blocked blood vessels.",
    "parkinsons": "A progressive nervous system disorder affecting movement, often causing tremors.",
    "lung_cancer": "A type of cancer that starts in the lungs, usually caused by smoking or pollution.",
    "thyroid": "A condition where the thyroid gland does not produce enough hormones, leading to fatigue and weight gain."
}

# Display Disease Info
if selected:
    st.markdown(f"### 🏥 {selected}")
    st.info(disease_info.get(disease_key, "No information available for this disease."))

    # Check if model is trained
    if disease_key in models:
        model = models[disease_key]
        scaler = scalers[disease_key]
        feature_names = features[disease_key]

        # User Inputs
        user_inputs = []
        for feat in feature_names:
            user_inputs.append(st.number_input(f"{feat}", key=feat))

        # Prediction Button
        if st.button(f"🔍 Predict {selected.replace(' Prediction', '')}"):
            try:
                input_array = np.array(user_inputs).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)
                result = "⚠️ The person has the disease!" if prediction[0] == 1 else "✅ The person does not have the disease."
                st.success(result)
            except Exception as e:
                st.error(f"Error in prediction: {e}")
    else:
        st.warning("Model not trained due to missing or invalid dataset.")