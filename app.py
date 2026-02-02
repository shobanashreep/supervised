import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

st.title("📞 Telecom Customer Churn Prediction")
st.write("Fill in customer details to predict whether the customer is likely to churn.")

# --------------------------------------------------
# Load model
# --------------------------------------------------
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
label_encoders = data["label_encoders"]
columns = data["columns"]
target_encoder = data["target_encoder"]

# --------------------------------------------------
# Sidebar inputs (like your screenshot)
# --------------------------------------------------
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("gender", ["Female", "Male"])
senior = st.sidebar.selectbox("SeniorCitizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.number_input(
    "tenure", min_value=0, max_value=100, value=1
)

phone_service = st.sidebar.selectbox(
    "PhoneService", ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "MultipleLines",
    ["Yes", "No", "No phone service"]  # ✅ FIXED
)

internet_service = st.sidebar.selectbox(
    "InternetService",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "OnlineSecurity",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "OnlineBackup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "DeviceProtection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "TechSupport",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "StreamingTV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "StreamingMovies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "PaperlessBilling",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "PaymentMethod",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "MonthlyCharges", min_value=0.0, max_value=200.0, value=50.0
)

total_charges = st.sidebar.number_input(
    "TotalCharges", min_value=0.0, max_value=10000.0, value=100.0
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.sidebar.button("🔮 Predict Churn"):

    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Ensure column order
    input_df = input_df[columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    result = target_encoder.inverse_transform([pred])[0]

    st.subheader("Prediction Result")

    if result == "Yes":
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")

    st.write(f"**Churn Probability:** {prob:.2%}")
