import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys
import joblib

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%; 
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    # st.image("https://img.icons8.com/color/96/000000/customer.png", width=100)
    st.title("Telco Churn Prediction")
    
    selected = option_menu(
        menu_title="Navigation",
        # options=["Dashboard", "Data Analysis", "Model Prediction", "Model Comparison", "About"],
        options=["Model Prediction", "Model Comparison", "About"],
        icons=["house", "graph-up", "robot", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Model Prediction":
    st.title("🤖 Churn Prediction")
    
    model_choice = st.selectbox("Select Model you want to predict with", ["Random Forest", "Transformer-based"])
    st.write("""
    Use this section to predict customer churn based on their information.
    Fill in the customer details below to get a prediction.
    """)
    # Create a form for user input
    with st.form("prediction_form"):
            customer_id = st.text_input("Customer ID")
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.checkbox("Senior Citizen")
            Partner = st.checkbox("Has Partner")
            Dependents = st.checkbox("Has Dependents")
            tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72)
            PhoneService = st.checkbox("Phone Service")
            MultipleLines= st.selectbox("Multiple Lines", ["Yes", "No", "No Phone Service"])
            InternetService = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup  = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaymentMethod = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, format="%.2f")
            TotalCharges = st.number_input("Total Charges", min_value=0.0, format="%.2f")


        
            submitted = st.form_submit_button("Predict Churn")
        
            if submitted:

                input_data = pd.DataFrame([{
                    "gender": gender,
                    "SeniorCitizen": int(SeniorCitizen),
                    "Partner": "Yes" if Partner else "No",
                    "Dependents": "Yes" if Dependents else "No",
                    "tenure": tenure,
                    "PhoneService": "Yes" if PhoneService else "No",
                    "MultipleLines": MultipleLines,
                    "InternetService": InternetService,
                    "OnlineSecurity": OnlineSecurity,
                    "OnlineBackup": OnlineBackup,
                    "DeviceProtection": DeviceProtection,
                    "TechSupport": TechSupport,
                    "StreamingTV": StreamingTV,
                    "StreamingMovies": StreamingMovies,
                    "Contract": Contract,
                    "PaperlessBilling": PaperlessBilling,
                    "PaymentMethod": PaymentMethod,
                    "MonthlyCharges": MonthlyCharges,
                    "TotalCharges": TotalCharges
                }])

                input_data.replace(['No internet service','No phone service'], 'No', inplace=True)

                if model_choice == "Random Forest":
                    cat_cols = ['gender', 'InternetService','PaymentMethod','Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'OnlineSecurity', 
                                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract']
                    
                    input_data = pd.concat([input_data, pd.get_dummies(input_data[cat_cols], drop_first=True)], axis=1)
                    input_data = input_data.drop(columns=cat_cols)
                    condition = [((input_data.tenure >= 0)&(input_data.tenure <= 12)), ((input_data.tenure > 12)&(input_data.tenure <= 24)), 
                                    ((input_data.tenure > 24)&(input_data.tenure <= 36)),((input_data.tenure > 36)&(input_data.tenure <= 48)),
                                    ((input_data.tenure > 48)&(input_data.tenure <= 60)), (input_data.tenure > 60)]

                    #choice = ['0-1year','1-2years', '2-3years', '3-4years','4-5years','more than 5 years']
                    choice = [0,1, 2, 3, 4, 5]
                    input_data['tenure_range'] = np.select(condition, choice)

                    input_data['MonthlyCharges']=np.log1p(input_data['MonthlyCharges'])
                    input_data['TotalCharges']=np.log1p(input_data['TotalCharges'])
                    model = joblib.load("models/random_forest_model.joblib")
                    prediction = model.predict(input_data)[0]
                    if prediction == 1:
                        st.error("🚨 Prediction: Customer is likely to **Churn**.")
                    else:
                        st.success("✅ Prediction: Customer is **Not likely to Churn**.")

                elif model_choice == "Transformer-based":
                    print("Transformer-based model selected")


elif selected == "Model Comparison":
    st.title("📊 Model Comparison")
    st.write("""
    Compare the performance of different models:
    - Traditional Machine Learning (Logistic Regression/Random Forest)
    - Transformer-based Model
    """)
    
    # Placeholder for model comparison metrics
    st.info("Model comparison metrics and visualizations will be added here.")

else:  # About page
    st.title("ℹ️ About")
    st.write("""
    ## Telco Customer Churn Prediction
    
    This application helps predict customer churn for a telecommunications company using machine learning techniques.
    
    ### Features:
    - Interactive dashboard with key metrics
    - Detailed data analysis
    - Real-time churn prediction
    - Model comparison and evaluation
    
    ### Technologies Used:
    - Python
    - Streamlit
    - Scikit-learn
    - TensorFlow/PyTorch
    - Plotly
    
    ### Dataset:
    The project uses the Telco Customer Churn dataset from Kaggle, which includes customer demographics,
    account information, and service usage data.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Telco Customer Churn</p>
    </div>
""", unsafe_allow_html=True) 