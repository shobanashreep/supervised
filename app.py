import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------- LOAD MODEL --------------------
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
label_encoders = data["label_encoders"]
target_encoder = data["target_encoder"]
columns = data["columns"]

st.set_page_config(page_title="Telecom Churn Predictor", layout="centered")
st.title("📞 Telecom Customer Churn Prediction")
st.write("Provide customer details in the sidebar to predict churn probability.")

# -------------------- USER INPUT --------------------
st.sidebar.header("Enter Customer Details")

# Function to create input form dynamically
def user_input():
    input_data = {}
    
    for col in columns:
        # If column is categorical (has LabelEncoder)
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            value = st.sidebar.selectbox(f"{col}", options)
            input_data[col] = label_encoders[col].transform([value])[0]
        else:
            # Binary columns with only 0/1 values
            if col == "SeniorCitizen":
                value = st.sidebar.selectbox(f"{col}", [0, 1])
            else:
                # continuous numeric columns
                value = st.sidebar.number_input(f"{col}", value=0.0)
            input_data[col] = value
    
    return pd.DataFrame([input_data])

input_df = user_input()

# -------------------- SCALE INPUT --------------------
input_scaled = scaler.transform(input_df)

# -------------------- PREDICTION --------------------
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)
pred_label = target_encoder.inverse_transform(prediction)[0]

# -------------------- DISPLAY RESULTS --------------------
st.subheader("Prediction Result")
if pred_label == "Yes":
    st.markdown(f"⚠️ **Churn Prediction:** {pred_label}")
else:
    st.markdown(f"✅ **Churn Prediction:** {pred_label}")

st.subheader("Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=target_encoder.classes_)
st.dataframe(proba_df.style.format("{:.2f}"))
