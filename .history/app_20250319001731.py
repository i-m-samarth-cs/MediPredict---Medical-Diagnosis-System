import streamlit as st
import pickle
import os

# Set up the Streamlit page
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️")

# Hide Streamlit default menu and footer
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Background Image
background_image_url = "https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg"
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-image: url({background_image_url});
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load trained models dynamically
models = {}
model_names = ['diabetes', 'heart_disease', 'parkinsons', 'lung_cancer', 'thyroid']
model_path = "models"

for model in model_names:
    model_file = os.path.join(model_path, f"{model}.pkl")
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            models[model] = pickle.load(f)
    else:
        st.error(f"Model file missing: {model_file}")

# Sidebar Menu
selected = st.sidebar.selectbox(
    'Select a Disease to Predict',
    ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Lung Cancer Prediction', 'Hypo-Thyroid Prediction']
)

# Function to collect input

def collect_inputs(labels, keys):
    return [st.number_input(label, key=key) for label, key in zip(labels, keys)]

# Prediction function
def make_prediction(model_key, inputs):
    if model_key in models:
        prediction = models[model_key].predict([inputs])
        return "The person has the disease" if prediction[0] == 1 else "The person does not have the disease"
    else:
        return "Model not available."

# Disease-specific Inputs and Predictions
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    labels = ['Pregnancies', 'Glucose Level', 'Blood Pressure', 'Skin Thickness', 'Insulin Level', 'BMI', 'Diabetes Pedigree Function', 'Age']
    keys = [f"diabetes_{i}" for i in range(len(labels))]
    inputs = collect_inputs(labels, keys)
    if st.button('Predict Diabetes'):
        st.success(make_prediction('diabetes', inputs))

elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    labels = ['Age', 'Sex (1=Male, 0=Female)', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting Blood Sugar (>120 mg/dl: 1, else 0)', 'Resting ECG', 'Max Heart Rate', 'Exercise-Induced Angina (1=Yes, 0=No)', 'ST Depression', 'Slope of ST', 'Major Vessels Colored', 'Thalassemia']
    keys = [f"heart_{i}" for i in range(len(labels))]
    inputs = collect_inputs(labels, keys)
    if st.button('Predict Heart Disease'):
        st.success(make_prediction('heart_disease', inputs))

elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction")
    labels = [f'Feature {i+1}' for i in range(22)]
    keys = [f"parkinsons_{i}" for i in range(len(labels))]
    inputs = collect_inputs(labels, keys)
    if st.button("Predict Parkinson's Disease"):
        st.success(make_prediction('parkinsons', inputs))

elif selected == "Lung Cancer Prediction":
    st.title("Lung Cancer Prediction")
    labels = [f'Feature {i+1}' for i in range(10)]
    keys = [f"lung_cancer_{i}" for i in range(len(labels))]
    inputs = collect_inputs(labels, keys)
    if st.button("Predict Lung Cancer"):
        st.success(make_prediction('lung_cancer', inputs))

elif selected == "Hypo-Thyroid Prediction":
    st.title("Hypo-Thyroid Prediction")
    labels = [f'Feature {i+1}' for i in range(15)]
    keys = [f"thyroid_{i}" for i in range(len(labels))]
    inputs = collect_inputs(labels, keys)
    if st.button("Predict Hypo-Thyroid"):
        st.success(make_prediction('thyroid', inputs))
