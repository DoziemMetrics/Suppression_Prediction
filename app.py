import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("model21.pkl")

# App title
st.title("HIV Viral Load Prediction App")

# Introduction
st.write("""
This application predicts the likelihood of HIV viral load suppression based on key patient data.
By analyzing treatment history and lab results, the model provides insights to support clinical decision-making.

### Why is this important?
HIV viral load suppression is critical in preventing disease progression and reducing transmission risks. By leveraging key patient information, this model helps healthcare providers identify individuals who may need additional support to achieve viral suppression.
""")

# Feature explanations
st.sidebar.header("Feature Descriptions")
st.sidebar.write("""
Each feature plays a significant role in predicting viral load suppression:

- **Age** 🏥: The age of the patient in years. Age can impact immune response and adherence to treatment.
- **IIT Count** ❌: The number of times the patient has experienced an interruption in treatment (IIT). Frequent interruptions can lead to drug resistance and treatment failure.
- **Refill Count** 💊: The total number of ARV refills the patient has collected. More refills generally indicate better adherence to medication, improving suppression outcomes.
- **Unsuppressed Count** 📉: The number of times a patient’s viral load was unsuppressed. A high count suggests challenges with adherence, resistance, or ineffective treatment.
- **Result Count** 🧪: The total number of viral load test results available for the patient. Frequent monitoring ensures timely intervention when suppression is not achieved.
- **Months on Treatment** 📆: The duration (in months) that the patient has been on ART. Longer treatment periods usually lead to improved viral suppression, provided adherence is maintained.
- **Months Since Last Pickup** 🔄: The time elapsed (in months) since the patient last picked up their ARVs. A long gap could indicate non-adherence and a higher risk of viral rebound.
""")

# User input
age = st.number_input("Age", min_value=0, max_value=120, value=30)
iit_count = st.number_input("IIT Count", min_value=0, value=0)
refill_count = st.number_input("Refill Count", min_value=0, value=0)
unsuppressed_count = st.number_input("Unsuppressed Count", min_value=0, value=0)
result_count = st.number_input("Result Count", min_value=0, value=0)
months_on_treatment = st.number_input("Months on Treatment", min_value=0, value=0)
months_since_last_pickup = st.number_input("Months Since Last Pickup", min_value=0, value=0)

# Prediction button
if st.button("Predict Viral Load Suppression"):
    # Ensure feature names match model training
    feature_names = ["Age", "IIT Count", "Refill Count", "Unsuppressed Count", "Result Count", "Months on Treatment", "Months Since Last Pickup"]
    input_df = pd.DataFrame([[age, iit_count, refill_count, unsuppressed_count, result_count, months_on_treatment, months_since_last_pickup]], 
                            columns=feature_names)

    # Make the prediction
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success("The patient is likely to have a suppressed viral load.")
    else:
        st.warning("The patient may have an unsuppressed viral load. Consider further assessment and intervention.")
