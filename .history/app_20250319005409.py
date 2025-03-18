import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import json

# Page Configuration
st.set_page_config(
    page_title="MediPredict | Disease Prediction", 
    page_icon="⚕️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    /* Main Page Styling */
    .stApp {
        background: linear-gradient(to bottom, #f0f5ff, #ffffff);
    }
    
    /* Custom Card */
    .css-card {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #1E40AF;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    /* Success Message */
    .stSuccess {
        background-color: #E8F5E9;
        padding: 16px;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Info Box */
    .stInfo {
        background-color: #E3F2FD;
        padding: 16px;
        border-radius: 6px;
        border-left: 4px solid #2196F3;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #F8FAFC;
    }
    
    /* Number Input */
    div[data-baseweb="input"] {
        border-radius: 6px !important;
    }
    
    /* Hide Hamburger Menu & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Loading Animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    
    /* Progress Bar */
    div.stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Load Lottie Animation
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Lottie Animations
lottie_medical = load_lottieurl("https://lottie.host/68175350-81b6-4594-958c-a4c351d70d26/EMQMi2cxSi.json")
lottie_prediction = load_lottieurl("https://lottie.host/27aea613-d36e-49c7-acdb-6b85419b71f5/yw10pJ27Pb.json")
lottie_result = load_lottieurl("https://lottie.host/55309046-8c70-46f1-a8dc-0e23d54c1b97/0uqKdJPOkF.json")

# Dataset paths
datasets = {
    "diabetes": "Datasets/diabetes_data.csv",
    "heart_disease": "Datasets/heart_disease_data.csv",
    "parkinsons": "Datasets/parkinson_data.csv",
    "lung_cancer": "Datasets/preprocessed_lungs_data.csv",
    "thyroid": "Datasets/preprocessed_hypothyroid.csv"
}

# Disease Icons and Colors
disease_icons = {
    "diabetes": "🥤",
    "heart_disease": "❤️",
    "parkinsons": "🧠",
    "lung_cancer": "🫁",
    "thyroid": "🦋"
}

disease_colors = {
    "diabetes": "#FF6B6B",
    "heart_disease": "#FF9E9E",
    "parkinsons": "#7579E7",
    "lung_cancer": "#9AB3F5",
    "thyroid": "#98DDCA"
}

# Function to create custom card container
def custom_card(title, content, icon="", color="#FFFFFF"):
    st.markdown(f"""
        <div style="background-color:{color}20; border-left:5px solid {color}; padding:15px; border-radius:5px; margin-bottom:20px;">
            <h3 style="color:{color}; margin-top:0;">{icon} {title}</h3>
            {content}
        </div>
    """, unsafe_allow_html=True)

# Function to load and train models
@st.cache_resource(show_spinner=False)
def train_model(dataset_path):
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

# --------- SIDEBAR ---------
with st.sidebar:
    st.title("🏥 MediPredict")
    st.markdown("### AI-Powered Disease Prediction")
    
    if lottie_medical:
        st_lottie(lottie_medical, height=200, key="sidebar_animation")
    
    st.markdown("---")
    
    # Disease Selection
    selected = st.selectbox(
        '🔹 Select a Disease to Predict',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinsons Prediction',
         'Lung Cancer Prediction',
         'Hypo-Thyroid Prediction']
    )
    
    st.markdown("---")
    
    # About Section
    with st.expander("ℹ️ About This App"):
        st.write("""
        **MediPredict** uses machine learning to predict the likelihood of various diseases based on your health parameters.
        
        **Note:** This tool is for educational purposes only and should not replace professional medical advice.
        """)
    
    # Tips Section
    with st.expander("💡 Tips for Accurate Results"):
        st.write("""
        - Enter your actual health metrics for accurate predictions
        - Use recent test results when available
        - Consult a healthcare professional for diagnosis
        """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ by Your Name")

# Fix the disease key extraction
disease_key = selected.replace(" Prediction", "").lower().replace(" ", "_")

# Disease Explanations
disease_info = {
    "diabetes": """
        Diabetes is a chronic condition affecting how your body processes glucose (blood sugar).
        
        **Risk Factors include:**
        - High blood sugar
        - Family history
        - Obesity
        - Physical inactivity
    """,
    "heart_disease": """
        Heart disease refers to various conditions affecting heart function, commonly due to narrowed or blocked blood vessels.
        
        **Risk Factors include:**
        - High blood pressure
        - High cholesterol
        - Smoking
        - Family history
        - Age
    """,
    "parkinsons": """
        Parkinson's disease is a progressive nervous system disorder affecting movement, causing tremors, stiffness, and balance problems.
        
        **Risk Factors include:**
        - Age (typically over 60)
        - Genetic factors
        - Environmental factors
        - Head injuries
    """,
    "lung_cancer": """
        Lung cancer is a type of cancer that begins in the lungs, usually caused by smoking or exposure to harmful substances.
        
        **Risk Factors include:**
        - Smoking
        - Exposure to secondhand smoke
        - Radon gas exposure
        - Family history
    """,
    "thyroid": """
        Hypothyroidism is a condition where the thyroid gland doesn't produce enough thyroid hormones, affecting metabolism.
        
        **Risk Factors include:**
        - Family history
        - Autoimmune diseases
        - Certain medications
        - Radiation therapy
    """
}

# Input Field Information
input_fields = {
    "diabetes": [
        ("Pregnancies", "Number of times pregnant", 0, 20),
        ("Glucose", "Plasma glucose concentration (mg/dL)", 70, 200),
        ("Blood Pressure", "Diastolic blood pressure (mm Hg)", 40, 120),
        ("Skin Thickness", "Triceps skinfold thickness (mm)", 5, 50),
        ("Insulin", "2-Hour serum insulin (mu U/ml)", 0, 500),
        ("BMI", "Body Mass Index (weight in kg/(height in m)^2)", 15, 50),
        ("Diabetes Pedigree Function", "Diabetes likelihood based on family history", 0.0, 2.5),
        ("Age", "Age of the person (years)", 18, 100),
    ],
    "heart_disease": [
        ("Age", "Age of the patient (years)", 20, 90),
        ("Sex", "1 = Male, 0 = Female", 0, 1),
        ("Chest Pain Type", "Type of chest pain (0-3)", 0, 3),
        ("Resting BP", "Resting blood pressure (mm Hg)", 90, 200),
        ("Cholesterol", "Serum cholesterol level (mg/dL)", 120, 400),
        ("Fasting Blood Sugar", "Blood sugar > 120 mg/dL (1 = true; 0 = false)", 0, 1),
        ("Resting ECG", "Electrocardiographic results (0-2)", 0, 2),
        ("Max Heart Rate", "Maximum heart rate achieved (bpm)", 70, 220),
        ("Exercise-Induced Angina", "Pain caused by exercise (1 = yes; 0 = no)", 0, 1),
    ],
    "parkinsons": [
        ("MDVP:Fo(Hz)", "Fundamental frequency of voice (Hz)", 80, 260),
        ("MDVP:Jitter(%)", "Variation in fundamental frequency (%)", 0, 1),
        ("MDVP:Shimmer", "Amplitude variation in voice", 0, 1),
        ("NHR", "Noise-to-Harmonics ratio", 0, 0.5),
        ("HNR", "Harmonics-to-Noise ratio (dB)", 8, 30),
        ("RPDE", "Dynamical complexity of voice", 0, 1),
        ("DFA", "Signal fractal scaling exponent", 0.5, 1),
    ],
    "lung_cancer": [
        ("Smoking", "1 = Yes, 0 = No", 0, 1),
        ("Yellow Fingers", "1 = Yes, 0 = No", 0, 1),
        ("Anxiety", "1 = Yes, 0 = No", 0, 1),
        ("Chronic Disease", "1 = Yes, 0 = No", 0, 1),
        ("Fatigue", "1 = Yes, 0 = No", 0, 1),
        ("Allergy", "1 = Yes, 0 = No", 0, 1),
        ("Wheezing", "1 = Yes, 0 = No", 0, 1),
    ],
    "thyroid": [
        ("TSH", "Thyroid-Stimulating Hormone level (μU/mL)", 0, 10),
        ("T3", "Triiodothyronine level (ng/dL)", 60, 200),
        ("TT4", "Total thyroxine level (μg/dL)", 5, 20),
        ("T4U", "Thyroxine utilization", 0.5, 1.5),
        ("FTI", "Free Thyroxine Index", 50, 200),
    ]
}

# Custom Input Function with slider and number input options
def display_input(label, tooltip, key, min_val, max_val):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write(f"**{label}:**")
    with col2:
        input_method = "slider"  # Default input method
        
        if input_method == "slider":
            value = st.slider(
                label, 
                min_value=float(min_val), 
                max_value=float(max_val), 
                key=f"slider_{key}",
                help=tooltip
            )
        else:
            value = st.number_input(
                label, 
                min_value=float(min_val), 
                max_value=float(max_val), 
                key=f"number_{key}",
                help=tooltip
            )
        
        return value

# Prediction Function
def predict_disease(disease_key, input_values):
    try:
        model = models[disease_key]
        scaler = scalers[disease_key]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        
        # Get probability for extra information
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        return prediction[0], prediction_proba
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Generate feature importance chart
def show_feature_importance(disease_key, model, feature_names):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    fig = px.bar(
        x=importances[sorted_indices],
        y=[feature_names[i] for i in sorted_indices],
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'},
        title='Feature Importance',
        color=importances[sorted_indices],
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        height=400,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(230, 230, 230, 0.8)'),
        yaxis=dict(showgrid=False),
        coloraxis_showscale=False
    )
    
    return fig

# Main App
def main():
    # Page Header
    st.title(f"{disease_icons.get(disease_key, '🏥')} {selected}")
    
    # Load Models in Background
    with st.spinner("Loading models..."):
        # Train models
        global models, scalers, features
        models, scalers, features = {}, {}, {}
        try:
            for disease, path in datasets.items():
                model, scaler, feature_names, target_col = train_model(path)
                models[disease] = model
                scalers[disease] = scaler
                features[disease] = feature_names
        except Exception as e:
            st.error(f"Error loading datasets: {e}")
    
    # Split layout into two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Disease information card
        if disease_key in disease_info:
            custom_card(
                f"About {selected.replace(' Prediction', '')}",
                disease_info[disease_key],
                icon="ℹ️",
                color=disease_colors.get(disease_key, "#3B82F6")
            )
        
        # Input Section
        st.markdown(f"### Enter Health Parameters")
        
        user_inputs = []
        if disease_key in input_fields:
            # Create responsive grid layout for inputs
            if len(input_fields[disease_key]) > 5:
                col_count = 2
            else:
                col_count = 1
                
            input_cols = st.columns(col_count)
            for i, (feat, desc, min_val, max_val) in enumerate(input_fields[disease_key]):
                with input_cols[i % col_count]:
                    user_inputs.append(display_input(feat, desc, feat, min_val, max_val))
        
        # Prediction Button
        st.markdown("")  # Spacing
        predict_clicked = st.button(f"🔍 Predict {selected.replace(' Prediction', '')}", use_container_width=True)
    
    with col2:
        # If we have model data, show feature importance
        if disease_key in models and disease_key in features:
            st.markdown("### Key Factors")
            fig = show_feature_importance(disease_key, models[disease_key], features[disease_key])
            st.plotly_chart(fig, use_container_width=True)
    
    # Results Section
    if predict_clicked:
        # Create a container for results
        results_container = st.container()
        
        with results_container:
            # Show loading animation
            with st.spinner("Analyzing health parameters..."):
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.01)
                    progress_bar.progress(i)
                
                # Get prediction
                result, probabilities = predict_disease(disease_key, user_inputs)
            
            # Clear progress bar
            progress_bar.empty()
            
            # Display result with animation
            st.markdown("### 📊 Prediction Results")
            
            if result is not None:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if lottie_result:
                        st_lottie(lottie_result, height=150, key="results_animation")
                
                with col2:
                    # Format the output nicely
                    if result == 1:
                        risk_level = "High"
                        st.markdown(f"""
                        <div style="background-color: #FFEBEE; padding: 15px; border-radius: 10px; border-left: 5px solid #D32F2F;">
                            <h3 style="color: #C62828; margin: 0;">⚠️ High Risk Detected</h3>
                            <p>Our analysis indicates that the person may have <b>{selected.replace(' Prediction', '')}</b>.</p>
                            <p><b>Recommendation:</b> Please consult with a healthcare professional for proper diagnosis.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        risk_level = "Low"
                        st.markdown(f"""
                        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; border-left: 5px solid #388E3C;">
                            <h3 style="color: #2E7D32; margin: 0;">✅ Low Risk Detected</h3>
                            <p>Our analysis indicates that the person likely does not have <b>{selected.replace(' Prediction', '')}</b>.</p>
                            <p><b>Recommendation:</b> Continue maintaining a healthy lifestyle.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show confidence gauge
                if probabilities is not None:
                    confidence = probabilities[1] if result == 1 else probabilities[0]
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Prediction Confidence", 'font': {'size': 24}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue" if confidence > 0.7 else "orange"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 40], 'color': '#E3F2FD'},
                                {'range': [40, 70], 'color': '#BBDEFB'},
                                {'range': [70, 100], 'color': '#90CAF9'}
                            ],
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor="white",
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Health tips based on prediction
                st.markdown("### 💡 Personalized Recommendations")
                
                if disease_key == "diabetes":
                    if result == 1:
                        st.markdown("""
                        - Monitor blood glucose levels regularly
                        - Follow a balanced diet low in sugar and simple carbohydrates
                        - Engage in regular physical activity
                        - Maintain a healthy weight
                        - Take medications as prescribed by your doctor
                        """)
                    else:
                        st.markdown("""
                        - Maintain a healthy diet rich in fruits, vegetables, and whole grains
                        - Exercise regularly (aim for 150 minutes per week)
                        - Maintain a healthy weight
                        - Limit alcohol consumption
                        - Have regular health check-ups
                        """)
                
                elif disease_key == "heart_disease":
                    if result == 1:
                        st.markdown("""
                        - Follow a heart-healthy diet low in saturated fats and sodium
                        - Engage in regular aerobic exercise
                        - Take medications as prescribed
                        - Manage stress through relaxation techniques
                        - Quit smoking and limit alcohol consumption
                        """)
                    else:
                        st.markdown("""
                        - Maintain a heart-healthy diet
                        - Exercise regularly
                        - Avoid smoking and limit alcohol
                        - Manage stress effectively
                        - Schedule regular heart health check-ups
                        """)
                        
                # Disclaimer
                st.markdown("""
                <div style="background-color: #FFF8E1; padding: 10px; border-radius: 5px; font-size: 0.8em; margin-top: 20px;">
                ⚠️ <b>Disclaimer:</b> This prediction is based on machine learning and is not a substitute for professional medical advice, diagnosis, or treatment.
                </div>
                """, unsafe_allow_html=True)

# Run the main app
if __name__ == "__main__":
    main()