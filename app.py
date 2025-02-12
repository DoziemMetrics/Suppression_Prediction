import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit UI
st.title("HIV Treatment Prediction App")

st.write("Enter the details below to predict the treatment outcome:")

# User inputs
age = st.number_input("Enter Age", min_value=0, max_value=100, value=25)
iit_count = st.number_input("IIT Count", min_value=0, max_value=10, value=0)
refill_count = st.number_input("Refill Count", min_value=0, max_value=100, value=5)
unsuppressed_count = st.number_input("Unsuppressed Count", min_value=0, max_value=50, value=0)

# When user clicks the Predict button
if st.button("Predict"):
    data = pd.DataFrame([[age, iit_count, refill_count, unsuppressed_count]],
                        columns=['Age', 'IIT Count', 'Refill Count', 'Unsuppressed Count'])
    prediction = model.predict(data)
    st.success(f"Predicted Category: {prediction[0]}")
