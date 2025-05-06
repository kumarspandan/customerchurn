import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# Load trained model
model_path = "C:/Users/LENOVO/Desktop/MajorProject/Newfolder/logistic_churn_model.sav"
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Sidebar navigation with default Streamlit styling
with st.sidebar:
    st.header("Churn Prediction System")
    selected = option_menu("Navigation", ["Churn Prediction"], icons=["bar-chart"], menu_icon="graph-up", default_index=0)

# UI Function with default styling
def get_user_input():
    st.markdown("### 🎨 Fill Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender_select")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract_select")
        online_security = st.selectbox("Online Security", ["No", "Yes"], key="online_security_select")
        tech_support = st.selectbox("Tech Support", ["No", "Yes"], key="tech_support_select")

    with col2:
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"], key="senior_select")
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], key="multiple_lines_select")
        online_backup = st.selectbox("Online Backup", ["No", "Yes"], key="online_backup_select")
        device_protection = st.selectbox("Device Protection", ["No", "Yes"], key="device_protection_select")

    with col3:
        partner = st.selectbox("Has Partner", ["Yes", "No"], key="partner_select")
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless_select")
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="payment_select")
        tenure = st.number_input("Tenure (in months)", min_value=0, value=0, key="tenure_input")

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=0.0, key="monthly_charges_input")

    # Convert SeniorCitizen to int
    senior_val = 1 if senior_citizen == "Yes" else 0

    user_input = {
        "gender": gender,
        "SeniorCitizen": senior_val,
        "Partner": partner,
        "tenure": tenure,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges
    }

    return user_input

# Encoding function
def ordinal_encode(df, cat_cols):
    for col in cat_cols:
        ordered = sorted(df[col].unique())
        mapping = {k: v for v, k in enumerate(ordered)}
        df[col] = df[col].map(mapping)
    return df

# Prediction Page with default styling
if selected == "Churn Prediction":
    st.title("📉 Customer Churn Prediction")

    user_input = get_user_input()
    input_df = pd.DataFrame([user_input])

    categorical = [col for col in input_df.columns if input_df[col].dtype == 'O']
    input_encoded = ordinal_encode(input_df.copy(), categorical)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            # Add a progress bar for visual feedback
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)

            prediction = model.predict(input_encoded)[0]
            probability = model.predict_proba(input_encoded)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")
        st.write(f"**Churn Probability:** {probability*100:.2f}%")

        if prediction == 1:
            st.error("Customer is likely to churn ❌")
        else:
            st.success("Customer is likely to stay ✅")

        if probability > 0.5:
            st.subheader("🛠 Retention Suggestions")
            suggestions = [
                "- Offer **discounts or personalized plans** based on usage.",
                "- Recommend **annual or two-year contracts** to increase retention.",
                "- Improve **customer support & technical help**.",
                "- Suggest bundles with **DeviceProtection**.",
                "- Provide benefits for long tenure customers."
            ]
            for suggestion in suggestions:
                st.markdown(suggestion)

# Footer with default styling
st.markdown(
    """
    <div style='text-align: center; padding: 10px;'>
        © 2025 Customer Churn Prediction | Developed by Kumar Spandan 
    </div>
    """,
    unsafe_allow_html=True
)