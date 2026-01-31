import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==================== LOAD MODEL ====================
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
label_encoders = data["label_encoders"]
target_encoder = data["target_encoder"]
columns = data["columns"]

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Telecom Churn Prediction", layout="centered")
st.title("📞 Telecom Customer Churn Prediction")
st.write("Fill in customer details to predict whether the customer is likely to churn.")

# ==================== SIDEBAR INPUT ====================
st.sidebar.header("Customer Details")

def get_user_input():
    input_data = {}

    for col in columns:
        # Categorical columns
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            value = st.sidebar.selectbox(col, options)
            input_data[col] = label_encoders[col].transform([value])[0]

        # Binary numeric column
        elif col == "SeniorCitizen":
            value = st.sidebar.selectbox(col, [0, 1])
            input_data[col] = value

        # Continuous numeric columns
        else:
            value = st.sidebar.number_input(col, value=0.0)
            input_data[col] = value

    return pd.DataFrame([input_data])

input_df = get_user_input()

# ==================== SCALE INPUT ====================
input_scaled = scaler.transform(input_df)

# ==================== PREDICTION ====================
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

pred_label = target_encoder.inverse_transform(prediction)[0]

# ==================== OUTPUT ====================
st.subheader("Prediction Result")

if pred_label == "Yes":
    st.error("⚠️ Customer is likely to CHURN")
else:
    st.success("✅ Customer is likely to STAY")

st.subheader("Prediction Probability")
proba_df = pd.DataFrame(
    probability,
    columns=target_encoder.classes_
)

st.dataframe(proba_df.style.format("{:.2%}"))
