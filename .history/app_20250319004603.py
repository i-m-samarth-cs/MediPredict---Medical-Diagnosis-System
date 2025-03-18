import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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

# Modified function to load and train models with numeric features only
def train_model(dataset_path):
    df = pd.read_csv(dataset_path)
    target_column = df.columns[-1]  # Assume last column is target
    
    # Keep only numeric columns for features
    X = df.drop(columns=[target_column])
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    X = X[numeric_features]
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, list(X.columns), target_column

# Train models
models, scalers, features = {}, {}, {}
try:
    for disease, path in datasets.items():
        model, scaler, feature_names, target_col = train_model(path)
        models[disease] = model
        scalers[disease] = scaler
        features[disease] = feature_names
except Exception as e:
    st.error(f"Error loading datasets: {e}")

# Sidebar Menu
selected = st.sidebar.selectbox(
    '🔹 Select a Disease to Predict',
    ['Diabetes Prediction',
     'Heart Disease Prediction',
     'Parkinsons Prediction',
     'Lung Cancer Prediction',
     'Hypo-Thyroid Prediction']
)

# Fix the disease key extraction
disease_key = selected.replace(" Prediction", "").lower().replace(" ", "_")

# Disease Explanations
disease_info = {
    "diabetes": "A chronic condition that affects how your body processes blood sugar (glucose).",
    "heart_disease": "A range of heart conditions affecting heart function, commonly due to blocked blood vessels.",
    "parkinsons": "A progressive nervous system disorder affecting movement, often causing tremors.",
    "lung_cancer": "A type of cancer that starts in the lungs, usually caused by smoking or pollution.",
    "thyroid": "A condition where the thyroid gland does not produce enough hormones, leading to fatigue and weight gain."
}

# Dynamically generate input fields based on available features
def get_input_fields(disease_key):
    if disease_key not in features:
        return []
    
    # Original input field descriptions
    field_descriptions = {
        # Diabetes
        "pregnancies": "Number of times pregnant",
        "glucose": "Plasma glucose concentration",
        "bloodpressure": "Diastolic blood pressure",
        "skinthickness": "Triceps skinfold thickness",
        "insulin": "2-Hour serum insulin",
        "bmi": "Body Mass Index",
        "diabetespedigreefunction": "Diabetes likelihood based on family history",
        "age": "Age of the person",
        
        # Heart Disease
        "sex": "1 = Male, 0 = Female",
        "chestpaintype": "Type of chest pain experienced",
        "restingbp": "Resting blood pressure",
        "cholesterol": "Serum cholesterol level",
        "fastingbloodsugar": "Blood sugar levels after fasting",
        "restingecg": "Electrocardiographic results",
        "maxheartrate": "Maximum heart rate achieved",
        "exerciseinducedangina": "Pain caused by exercise",
        
        # Parkinsons (selected numeric features)
        "mdvp:fo(hz)": "Fundamental frequency of voice",
        "mdvp:jitter(%)": "Variation in fundamental frequency",
        "mdvp:shimmer": "Amplitude variation in voice",
        "nhr": "Noise-to-Harmonics ratio",
        "hnr": "Harmonics-to-Noise ratio",
        "rpde": "Dynamical complexity of voice",
        "dfa": "Signal fractal scaling exponent",
        
        # Lung Cancer
        "smoking": "1 = Yes, 0 = No",
        "yellowfingers": "1 = Yes, 0 = No",
        "anxiety": "1 = Yes, 0 = No",
        "chronicdisease": "1 = Yes, 0 = No",
        "fatigue": "1 = Yes, 0 = No",
        "allergy": "1 = Yes, 0 = No",
        "wheezing": "1 = Yes, 0 = No",
        
        # Thyroid
        "tsh": "Thyroid-Stimulating Hormone level",
        "t3": "Triiodothyronine level",
        "tt4": "Total thyroxine level",
        "t4u": "Thyroxine utilization",
        "fti": "Free Thyroxine Index"
    }
    
    input_fields = []
    for feature in features[disease_key]:
        # Try to find a description, fallback to generic
        feature_key = feature.lower().replace(" ", "")
        description = field_descriptions.get(feature_key, "Numerical input value")
        input_fields.append((feature, description))
    
    return input_fields

# Input Function
def display_input(label, tooltip, key):
    return st.number_input(label, key=key, help=tooltip)

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

# Display Section
if selected:
    st.markdown(f"### 🏥 {selected}")

    # Display disease info
    if disease_key in disease_info:
        st.info(disease_info[disease_key])
    else:
        st.warning("No information available for this disease.")

    # Get dynamically generated input fields
    dynamic_input_fields = get_input_fields(disease_key)
    
    # Display input fields with descriptions
    user_inputs = []
    if dynamic_input_fields:
        for feat, desc in dynamic_input_fields:
            st.markdown(f"**{feat}** - {desc}")  # Display description above input
            user_inputs.append(display_input(feat, desc, feat))
    else:
        st.warning(f"No input fields available for {selected}")

    # Prediction Button
    if st.button(f"🔍 Predict {selected.replace(' Prediction', '')}"):
        if user_inputs:
            result = predict_disease(disease_key, user_inputs)
            if result == 1:
                st.success(f"⚠️ The person has {selected.replace(' Prediction', '')}!")
            else:
                st.success(f"✅ The person does not have {selected.replace(' Prediction', '')}.")
        else:
            st.error("Cannot make prediction without input fields.")