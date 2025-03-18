import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Streamlit Page Config
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://wallpaperaccess.com/full/4112935.png");
        background-size: cover;
        background-position: center;
    }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

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
        target_column = df.columns[-1]  # Assume last column is target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model, scaler, list(X.columns), target_column
    except Exception as e:
        st.error(f"Error loading dataset {dataset_path}: {e}")
        return None, None, None, None

# Train models
models, scalers, features = {}, {}, {}
for disease, path in datasets.items():
    model, scaler, feature_names, target_col = train_model(path)
    if model:
        models[disease] = model
        scalers[disease] = scaler
        features[disease] = feature_names

# Sidebar Menu
selected = st.sidebar.selectbox(
    '🔹 Select a Disease to Predict',
    ['Diabetes', 'Heart Disease', 'Parkinsons', 'Lung Cancer', 'Hypothyroid']
)

disease_key = selected.lower().replace(" ", "_")

# Disease Explanations
disease_info = {
    "diabetes": "A chronic condition affecting how your body processes blood sugar.",
    "heart_disease": "A range of heart conditions that affect heart function.",
    "parkinsons": "A progressive nervous system disorder affecting movement.",
    "lung_cancer": "A type of cancer that starts in the lungs, often linked to smoking.",
    "thyroid": "A condition affecting thyroid hormone production, leading to metabolic issues."
}

# Input Field Information
input_fields = {
    "diabetes": [
        ("Pregnancies", "Number of times pregnant"),
        ("Glucose", "Plasma glucose concentration"),
        ("Blood Pressure", "Diastolic blood pressure"),
        ("Skin Thickness", "Triceps skinfold thickness"),
        ("Insulin", "2-Hour serum insulin"),
        ("BMI", "Body Mass Index"),
        ("Diabetes Pedigree Function", "Diabetes likelihood based on family history"),
        ("Age", "Age of the person")
    ],
    "heart_disease": [
        ("Age", "Age of the patient"),
        ("Sex", "1 = Male, 0 = Female"),
        ("Chest Pain Type", "Type of chest pain experienced"),
        ("Resting BP", "Resting blood pressure"),
        ("Cholesterol", "Serum cholesterol level"),
        ("Fasting Blood Sugar", "Blood sugar levels after fasting"),
        ("Resting ECG", "Electrocardiographic results"),
        ("Max Heart Rate", "Maximum heart rate achieved"),
        ("Exercise-Induced Angina", "Pain caused by exercise")
    ],
    "parkinsons": [
        ("MDVP:Fo(Hz)", "Fundamental frequency of voice"),
        ("MDVP:Jitter(%)", "Variation in fundamental frequency"),
        ("MDVP:Shimmer", "Amplitude variation in voice"),
        ("NHR", "Noise-to-Harmonics ratio"),
        ("HNR", "Harmonics-to-Noise ratio"),
        ("RPDE", "Dynamical complexity of voice"),
        ("DFA", "Signal fractal scaling exponent")
    ],
    "lung_cancer": [
        ("Smoking", "1 = Yes, 0 = No"),
        ("Yellow Fingers", "1 = Yes, 0 = No"),
        ("Anxiety", "1 = Yes, 0 = No"),
        ("Chronic Disease", "1 = Yes, 0 = No"),
        ("Fatigue", "1 = Yes, 0 = No"),
        ("Allergy", "1 = Yes, 0 = No"),
        ("Wheezing", "1 = Yes, 0 = No")
    ],
    "thyroid": [
        ("TSH", "Thyroid-Stimulating Hormone level"),
        ("T3", "Triiodothyronine level"),
        ("TT4", "Total thyroxine level"),
        ("T4U", "Thyroxine utilization"),
        ("FTI", "Free Thyroxine Index")
    ]
}

# Display UI
st.markdown(f"### 🏥 {selected} Prediction")
st.info(disease_info.get(disease_key, "No information available for this disease."))

# Collect user inputs
user_inputs = [st.number_input(feat, help=desc, key=feat) for feat, desc in input_fields[disease_key]]

# Prediction Function
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

# Prediction Button
if st.button(f"🔍 Predict {selected}"):
    result = predict_disease(disease_key, user_inputs)
    if result == 1:
        st.success(f"⚠️ The person has {selected}!")
    else:
        st.success(f"✅ The person does not have {selected}.")
