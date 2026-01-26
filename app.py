import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

# Title
st.title("📞 Telecom Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to **churn or stay** using Machine Learning.")

st.markdown("---")

# ===============================
# 👤 Customer Profile
# ===============================
st.header("👤 Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])

with col2:
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col3:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

# ===============================
# 📞 Services Used
# ===============================
st.header("📞 Services Used")

col4, col5, col6 = st.columns(3)

with col4:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col5:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col6:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# ===============================
# 💳 Contract & Billing
# ===============================
st.header("💳 Contract & Billing")

col7, col8, col9 = st.columns(3)

with col7:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col8:
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

with col9:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# ===============================
# 💰 Charges
# ===============================
st.header("💰 Charges")

col10, col11 = st.columns(2)

with col10:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

with col11:
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

st.markdown("---")

# ===============================
# 🔮 Prediction
# ===============================
if st.button("🔍 Predict Churn"):
    input_data = pd.DataFrame([{
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "MultipleLines": 1 if multiple_lines == "Yes" else 0,
        "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service],
        "OnlineSecurity": 1 if online_security == "Yes" else 0,
        "OnlineBackup": 1 if online_backup == "Yes" else 0,
        "DeviceProtection": 1 if device_protection == "Yes" else 0,
        "TechSupport": 1 if tech_support == "Yes" else 0,
        "StreamingTV": 1 if streaming_tv == "Yes" else 0,
        "StreamingMovies": 1 if streaming_movies == "Yes" else 0,
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "PaymentMethod": {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3
        }[payment_method],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    st.subheader("📊 Prediction Result")

    if prediction == 0:
        st.success(f"✅ Customer is likely to **STAY** (Probability: {probability:.2f})")
    else:
        st.error(f"⚠️ Customer is likely to **CHURN** (Probability: {probability:.2f})")
    import os
    import pickle

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

