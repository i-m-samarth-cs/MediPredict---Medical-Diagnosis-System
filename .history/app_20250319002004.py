import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# Change Name & Logo
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️")

# Hiding Streamlit add-ons
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load datasets
datasets = {
    "diabetes": "Datasets/diabetes_data.csv",
    "heart_disease": "Datasets/heart_disease_data.csv",
    "parkinsons": "Datasets/parkinson_data.csv",
    "lung_cancer": "Datasets/preprocessed_lungs_data.csv",
    "thyroid": "Datasets/preprocessed_hypothyroid.csv"
}

# Train models dynamically
def train_model(dataset_path, target_column):
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target variable

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, list(X.columns)  # Return model, scaler & feature names

# Train all models
models = {}
scalers = {}
features = {}

try:
    for disease, path in datasets.items():
        target = "Outcome" if disease == "diabetes" else "target"
        model, scaler, feature_names = train_model(path, target)
        models[disease] = model
        scalers[disease] = scaler
        features[disease] = feature_names
except Exception as e:
    st.error(f"Error loading datasets: {e}")

# Sidebar Menu
selected = st.sidebar.selectbox(
    'Select a Disease to Predict',
    ['Diabetes Prediction',
     'Heart Disease Prediction',
     'Parkinsons Prediction',
     'Lung Cancer Prediction',
     'Hypo-Thyroid Prediction']
)

# Input Display Function
def display_input(label, tooltip, key):
    return st.number_input(label, key=key, help=tooltip)

# Prediction function
def predict_disease(disease_key, input_values):
    try:
        model = models[disease_key]
        scaler = scalers[disease_key]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        return prediction[0]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Disease-Specific Input Forms
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    user_inputs = [display_input(feat, f"Enter {feat}", feat) for feat in features['diabetes']]
    if st.button('Predict Diabetes'):
        result = predict_disease('diabetes', user_inputs)
        st.success("The person is diabetic" if result == 1 else "The person is not diabetic")

elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    user_inputs = [display_input(feat, f"Enter {feat}", feat) for feat in features['heart_disease']]
    if st.button('Predict Heart Disease'):
        result = predict_disease('heart_disease', user_inputs)
        st.success("The person has heart disease" if result == 1 else "The person does not have heart disease")

elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction")
    user_inputs = [display_input(feat, f"Enter {feat}", feat) for feat in features['parkinsons']]
    if st.button("Predict Parkinson's Disease"):
        result = predict_disease('parkinsons', user_inputs)
        st.success("The person has Parkinson's disease" if result == 1 else "The person does not have Parkinson's disease")

elif selected == "Lung Cancer Prediction":
    st.title("Lung Cancer Prediction")
    user_inputs = [display_input(feat, f"Enter {feat}", feat) for feat in features['lung_cancer']]
    if st.button("Predict Lung Cancer"):
        result = predict_disease('lung_cancer', user_inputs)
        st.success("The person has lung cancer" if result == 1 else "The person does not have lung cancer")

elif selected == "Hypo-Thyroid Prediction":
    st.title("Hypo-Thyroid Prediction")
    user_inputs = [display_input(feat, f"Enter {feat}", feat) for feat in features['thyroid']]
    if st.button("Predict Hypo-Thyroid"):
        result = predict_disease('thyroid', user_inputs)
        st.success("The person has hypothyroidism" if result == 1 else "The person does not have hypothyroidism")
