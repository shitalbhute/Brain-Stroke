import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Ensure scaler.pkl exists or create one if missing
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    # Load dataset to create scaler if missing
    data = pd.read_csv("patient_data.csv")  # Replace with the path to your dataset

    # Encode categorical variables
    label_encoder_gender = LabelEncoder()
    label_encoder_smoking = LabelEncoder()
    data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
    data['Smoking Status'] = label_encoder_smoking.fit_transform(data['Smoking Status'])

    # Define features (X)
    X = data[['Age', 'Gender', 'Hypertension', 'Heart Disease', 'Smoking Status', 'BMI', 'Avg Glucose Level']]

    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(X)

    # Save the scaler for future use
    joblib.dump(scaler, "scaler.pkl")

# Load the trained model
model = joblib.load("stroke_prediction_model.pkl")

# Define mappings for gender and smoking status
gender_mapping = {"Male": 1, "Female": 0}
smoking_status_mapping = {"Never smoked": 2, "Former smoker": 1, "Smokes": 0}

# Streamlit app setup
st.set_page_config(page_title="Brain Stroke Prediction", page_icon="🧠", layout="centered")

# App Title
st.title("Early Detection of Brain Stroke")
st.write("This application predicts the probability of a brain stroke based on patient data.")

# Input fields
st.sidebar.header("Enter Patient Details")
age = st.sidebar.slider("Age", min_value=0, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
hypertension = st.sidebar.selectbox("Hypertension", options=["No", "Yes"])
heart_disease = st.sidebar.selectbox("Heart Disease", options=["No", "Yes"])
smoking_status = st.sidebar.selectbox("Smoking Status", options=["Never smoked", "Former smoker", "Smokes"])
bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=50.0, value=22.5, step=0.1)
avg_glucose_level = st.sidebar.slider("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0, step=0.1)

# Predict button
if st.sidebar.button("Predict Stroke Risk"):
    # Convert inputs to encoded values
    gender_encoded = gender_mapping[gender]
    smoking_status_encoded = smoking_status_mapping[smoking_status]
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0

    # Create input array
    input_data = np.array([[age, gender_encoded, hypertension_encoded, heart_disease_encoded,
                            smoking_status_encoded, bmi, avg_glucose_level]])
    input_data_scaled = scaler.transform(input_data)

    # Predict probability
    stroke_probability = model.predict_proba(input_data_scaled)[0][1] * 100

    # Display result
    st.markdown(f"### The predicted chance of having a brain stroke is **{stroke_probability:.2f}%**.")

# Footer with additional information
st.markdown("---")
st.markdown("Developed with ❤️ for healthcare advancement.")
st.markdown("[GitHub Repository](https://github.com/your-repo) | [Contact Developer](mailto:developer@example.com)")
