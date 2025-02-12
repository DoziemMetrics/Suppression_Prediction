import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load("model.pkl")

# Title of the Streamlit app
st.title("ARV Client Adherence Prediction")

# Input fields for user data
sex = st.selectbox("Sex", ["Male", "Female"])
target_group = st.selectbox("Target Group", ["FSW", "MSM", "PWID", "Others"])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
current_art_status = st.selectbox("Current ART Status", ["Active", "IIT"])
iit_count = st.number_input("IIT Count", min_value=0, step=1)
refill_count = st.number_input("Refill Count", min_value=0, step=1)
unsuppressed_count = st.number_input("Unsuppressed Count", min_value=0, step=1)
result_count = st.number_input("Result Count", min_value=0, step=1)
months_on_treatment = st.number_input("Months on Treatment", min_value=0, step=1)
months_since_last_pickup = st.number_input("Months Since Last Pickup", min_value=0, step=1)

# Convert categorical features to numerical
def encode_categorical(value, mapping):
    return mapping.get(value, -1)

sex_mapping = {"Male": 0, "Female": 1}
target_group_mapping = {"FSW": 0, "MSM": 1, "PWID": 2, "Others": 3}
current_art_status_mapping = {"Active": 0, "IIT": 1}

sex_encoded = encode_categorical(sex, sex_mapping)
target_group_encoded = encode_categorical(target_group, target_group_mapping)
current_art_status_encoded = encode_categorical(current_art_status, current_art_status_mapping)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    "Sex": [sex_encoded],
    "Target group": [target_group_encoded],
    "Age": [age],
    "Current ART Status": [current_art_status_encoded],
    "IIT Count": [iit_count],
    "Refill Count": [refill_count],
    "Unsuppressed Count": [unsuppressed_count],
    "Result_Count": [result_count],
    "Months on Treatment": [months_on_treatment],
    "Months Since Last Pickup": [months_since_last_pickup]
})

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Outcome: {prediction[0]}")

