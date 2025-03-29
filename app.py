import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("model21.pkl")

# Define expected feature names
expected_features = model.feature_names_in_

# Streamlit App Layout
st.set_page_config(page_title="HIV Viral Load Prediction", layout="wide")

# Sidebar: App Info
st.sidebar.title("🔍 About This App")
st.sidebar.write("""
This app predicts whether a patient is likely to have a **suppressed viral load** based on input data.

- Enter patient details in the form.
- Click **Predict** to see the prediction.
""")

st.sidebar.info("**Note:** This is a machine learning model and should not replace clinical judgment.")

# Sidebar: Explanation of Features
st.sidebar.title("📊 Feature Descriptions")
st.sidebar.write("""
Below is an explanation of each input feature used for prediction:
- **🧑 Age**: The patient's age in years.
- **🔁 IIT Count**: Number of times the patient had an **Interruption in Treatment (IIT)**.
- **💊 Total Refill Count**: Total number of **ARV medication refills** the patient has completed.
- **📉 Unsuppressed Count**: Number of times the patient had an **unsuppressed viral load**.
- **📑 Total Test Results Count**: Total number of viral load tests conducted for this patient.
- **🗓️ Months on Treatment**: Total number of **months since ART initiation**.
- **⏳ Months Since Last Pickup**: Time (in months) since the patient's last **ARV pickup**.
""")

# Main App Title
st.title("📊 HIV Viral Load Suppression Prediction")

st.write("""
This model helps predict whether a **patient's viral load is suppressed** based on their treatment history.  
**Fill in the required details and click 'Predict' to get results.**
""")

# User Inputs
st.subheader("📥 Enter Patient Information")

age = st.number_input("🧑 Age", min_value=0, max_value=120, value=30)
iit_count = st.number_input("🔁 IIT Count (Interruptions in Treatment)", min_value=0, value=0)
refill_count = st.number_input("💊 Total Refill Count", min_value=0, value=0)
unsuppressed_count = st.number_input("📉 Unsuppressed Count", min_value=0, value=0)
result_count = st.number_input("📑 Total Test Results Count", min_value=0, value=0)
months_on_treatment = st.number_input("🗓️ Months on Treatment", min_value=0, value=0)
months_since_last_pickup = st.number_input("⏳ Months Since Last Pickup", min_value=0, value=0)

# Prediction Button
if st.button("🔮 Predict Viral Load Suppression"):
    try:
        # Create DataFrame in correct order
        input_df = pd.DataFrame([[age, iit_count, refill_count, unsuppressed_count, result_count, months_on_treatment, months_since_last_pickup]], 
                                 columns=expected_features)  # Ensure column names match model training

        # Make Prediction
        prediction = model.predict(input_df)

        # Display Results
        st.subheader("📊 Prediction Result")
        if prediction[0] == 1:
            st.success("✅ The patient is likely to have a **suppressed viral load**.")
        else:
            st.warning("⚠️ The patient **may have an unsuppressed viral load**. Consider further assessment.")

    except ValueError as e:
        st.error(f"❌ Error in prediction: {e}")
