import streamlit as st
import pickle
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

# Adding Background Image
background_image_url = "https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg"

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url({background_image_url});
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stAppViewContainer"]::before {{
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background-color: rgba(0, 0, 0, 0.7);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the trained models
try:
    models = {
        'diabetes': pickle.load(open('models/diabetes.pkl', 'rb')),
        'heart_disease': pickle.load(open('models/heart_disease.pkl', 'rb')),
        'parkinsons': pickle.load(open('models/parkinsons.pkl', 'rb')),
        'lung_cancer': pickle.load(open('models/lung_cancer.pkl', 'rb')),
        'thyroid': pickle.load(open('models/thyroid.pkl', 'rb'))
    }
except Exception as e:
    st.error(f"Error loading models: {e}")

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
def display_input(label, tooltip, key, type="number"):
    return st.number_input(label, key=key, help=tooltip) if type == "number" else st.text_input(label, key=key, help=tooltip)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    st.write("Enter the following details to predict diabetes:")

    Pregnancies = display_input('Number of Pregnancies', 'Enter number of times pregnant', 'Pregnancies')
    Glucose = display_input('Glucose Level', 'Enter glucose level', 'Glucose')
    BloodPressure = display_input('Blood Pressure', 'Enter blood pressure value', 'BloodPressure')
    SkinThickness = display_input('Skin Thickness', 'Enter skin thickness value', 'SkinThickness')
    Insulin = display_input('Insulin Level', 'Enter insulin level', 'Insulin')
    BMI = display_input('BMI value', 'Enter Body Mass Index', 'BMI')
    DiabetesPedigreeFunction = display_input('Diabetes Pedigree Function', 'Enter function value', 'DiabetesPedigreeFunction')
    Age = display_input('Age', 'Enter age of the person', 'Age')

    if st.button('Predict Diabetes'):
        diab_prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        st.success(result)

# Heart Disease Prediction
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    st.write("Enter the following details to predict heart disease:")

    age = display_input('Age', 'Enter age', 'age')
    sex = display_input('Sex (1 = Male, 0 = Female)', 'Enter sex', 'sex')
    cp = display_input('Chest Pain Type (0-3)', 'Enter chest pain type', 'cp')
    trestbps = display_input('Resting Blood Pressure', 'Enter blood pressure', 'trestbps')
    chol = display_input('Serum Cholesterol (mg/dl)', 'Enter cholesterol', 'chol')
    fbs = display_input('Fasting Blood Sugar (>120 mg/dl: 1, else 0)', 'Enter fasting blood sugar', 'fbs')
    restecg = display_input('Resting ECG (0-2)', 'Enter ECG results', 'restecg')
    thalach = display_input('Max Heart Rate Achieved', 'Enter max heart rate', 'thalach')
    exang = display_input('Exercise-Induced Angina (1 = Yes, 0 = No)', 'Enter angina status', 'exang')
    oldpeak = display_input('ST Depression by Exercise', 'Enter ST depression', 'oldpeak')
    slope = display_input('Slope of ST Segment (0-2)', 'Enter slope', 'slope')
    ca = display_input('Major Vessels Colored (0-3)', 'Enter number of vessels', 'ca')
    thal = display_input('Thalassemia (0-2)', 'Enter thal value', 'thal')

    if st.button('Predict Heart Disease'):
        heart_prediction = models['heart_disease'].predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        result = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
        st.success(result)

# Parkinson's Prediction
elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction")
    st.write("Enter the following details to predict Parkinson's disease:")

    inputs = [display_input(f'Feature {i+1}', 'Enter value', f'feature_{i+1}') for i in range(22)]

    if st.button("Predict Parkinson's Disease"):
        parkinsons_prediction = models['parkinsons'].predict([inputs])
        result = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        st.success(result)

# Lung Cancer Prediction
elif selected == "Lung Cancer Prediction":
    st.title("Lung Cancer Prediction")
    st.write("Enter the following details to predict lung cancer:")

    inputs = [display_input(f'Feature {i+1}', 'Enter value', f'feature_{i+1}') for i in range(10)]

    if st.button("Predict Lung Cancer"):
        lung_prediction = models['lung_cancer'].predict([inputs])
        result = "The person has lung cancer" if lung_prediction[0] == 1 else "The person does not have lung cancer"
        st.success(result)

# Thyroid Prediction
elif selected == "Hypo-Thyroid Prediction":
    st.title("Hypo-Thyroid Prediction")
    st.write("Enter the following details to predict hypothyroidism:")

    inputs = [display_input(f'Feature {i+1}', 'Enter value', f'feature_{i+1}') for i in range(15)]

    if st.button("Predict Hypo-Thyroid"):
        thyroid_prediction = models['thyroid'].predict([inputs])
        result = "The person has hypothyroidism" if thyroid_prediction[0] == 1 else "The person does not have hypothyroidism"
        st.success(result)
