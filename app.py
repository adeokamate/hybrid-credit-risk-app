import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Hybrid Credit Risk System",
    page_icon="💳",
    layout="wide"
)

st.title("Hybrid Credit Risk Prediction System")
st.caption("Model A: Loan Default Probability | Model B: Mobile Money Behavioral Risk")

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_artifacts():
    model_a = joblib.load("model_a.pkl")          # XGBoost model
    iso_model_b = joblib.load("iso_model_b.pkl")  # Isolation Forest
    scaler_b = joblib.load("scaler_b.pkl")        # MinMaxScaler for Model B inputs
    risk_scaler_b = joblib.load("risk_scaler_b.pkl")  # scaler for anomaly score -> 0-1
    return model_a, iso_model_b, scaler_b, risk_scaler_b

model_a, iso_model_b, scaler_b, risk_scaler_b = load_artifacts()

# -----------------------------
# Sidebar navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model A: Loan Default", "Model B: Behavioral Risk", "Final Hybrid Score"]
)

# -----------------------------
# Helper functions
# -----------------------------
def risk_band(score: float) -> str:
    if score < 0.20:
        return "Low Risk"
    elif score < 0.40:
        return "Moderate Risk"
    elif score < 0.60:
        return "High Risk"
    return "Very High Risk"

def normalize_anomaly_score(raw_score: np.ndarray, scaler) -> np.ndarray:
    # IsolationForest.decision_function: higher = more normal
    # Invert so higher = riskier
    inverted = -raw_score.reshape(-1, 1)
    normalized = scaler.transform(inverted).flatten()
    normalized = np.clip(normalized, 0, 1)
    return normalized

# -----------------------------
# Session state
# -----------------------------
if "default_probability" not in st.session_state:
    st.session_state.default_probability = None

if "behavior_risk" not in st.session_state:
    st.session_state.behavior_risk = None

# -----------------------------
# Overview page
# -----------------------------
if page == "Overview":
    st.subheader("System Overview")
    st.write(
        """
        This app demonstrates a hybrid credit risk framework:

        - **Model A** predicts loan default probability from structured loan features.
        - **Model B** estimates behavioral risk from mobile money activity patterns.
        - **Final Risk** combines both scores using a weighted formula.
        """
    )

    st.latex(r"FinalRisk = 0.7 \times DefaultProbability + 0.3 \times BehaviorRisk")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model A Weight", "0.7")
    c2.metric("Model B Weight", "0.3")
    c3.metric("Output Range", "0 to 1")

# -----------------------------
# Model A page
# -----------------------------
elif page == "Model A: Loan Default":
    st.subheader("Loan Default Prediction")

    with st.form("model_a_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=5000.0)
            term = st.selectbox("Term", [36, 60])
            int_rate = st.number_input("Interest Rate", min_value=0.0, value=12.0)
            installment = st.number_input("Installment", min_value=0.0, value=200.0)
            annual_inc = st.number_input("Annual Income", min_value=0.0, value=40000.0)
            dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0)

        with col2:
            delinq_2yrs = st.number_input("Delinquencies in 2 Years", min_value=0, value=0)
            inq_last_6mths = st.number_input("Inquiries Last 6 Months", min_value=0, value=1)
            open_acc = st.number_input("Open Accounts", min_value=0, value=6)
            total_acc = st.number_input("Total Accounts", min_value=0, value=20)
            revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=8000.0)
            revol_util = st.number_input("Revolving Utilization", min_value=0.0, value=45.0)

        with col3:
            emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=10, value=3)
            home_ownership = st.number_input("Home Ownership (encoded)", min_value=0, value=1)
            verification_status = st.number_input("Verification Status (encoded)", min_value=0, value=1)
            purpose = st.number_input("Purpose (encoded)", min_value=0, value=2)
            credit_age_days = st.number_input("Credit Age (days)", min_value=0, value=3000)

        submitted_a = st.form_submit_button("Predict Default Probability")

    if submitted_a:
        input_a = pd.DataFrame([{
            "loan_amnt": loan_amnt,
            "term": term,
            "int_rate": int_rate,
            "installment": installment,
            "annual_inc": annual_inc,
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "total_acc": total_acc,
            "revol_bal": revol_bal,
            "revol_util": revol_util,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "verification_status": verification_status,
            "purpose": purpose,
            "credit_age_days": credit_age_days
        }])

        default_probability = float(model_a.predict_proba(input_a)[:, 1][0])
        st.session_state.default_probability = default_probability

        st.metric("Default Probability", f"{default_probability:.3f}")
        st.success(f"Risk Band: {risk_band(default_probability)}")

# -----------------------------
# Model B page
# -----------------------------
elif page == "Model B: Behavioral Risk":
    st.subheader("Mobile Money Behavioral Risk")

    with st.form("model_b_form"):
        col1, col2 = st.columns(2)

        with col1:
            transaction_count = st.number_input("Transaction Count", min_value=1, value=10)
            total_amount = st.number_input("Total Amount", min_value=0.0, value=500000.0)
            avg_amount = st.number_input("Average Amount", min_value=0.0, value=50000.0)
            max_amount = st.number_input("Maximum Amount", min_value=0.0, value=120000.0)
            min_amount = st.number_input("Minimum Amount", min_value=0.0, value=5000.0)
            std_amount = st.number_input("Std of Amount", min_value=0.0, value=25000.0)

        with col2:
            avg_oldbalance = st.number_input("Average Old Balance", min_value=0.0, value=100000.0)
            avg_newbalance = st.number_input("Average New Balance", min_value=0.0, value=70000.0)
            unique_destinations = st.number_input("Unique Destinations", min_value=1, value=4)
            avg_step = st.number_input("Average Step", min_value=0.0, value=150.0)
            std_step = st.number_input("Std of Step", min_value=0.0, value=20.0)

        submitted_b = st.form_submit_button("Score Behavioral Risk")

    if submitted_b:
        amount_range = max_amount - min_amount
        balance_change = avg_oldbalance - avg_newbalance
        amount_per_transaction = total_amount / (transaction_count + 1)
        destination_ratio = unique_destinations / (transaction_count + 1)

        input_b = pd.DataFrame([{
            "transaction_count": transaction_count,
            "total_amount": total_amount,
            "avg_amount": avg_amount,
            "max_amount": max_amount,
            "min_amount": min_amount,
            "std_amount": std_amount,
            "avg_oldbalance": avg_oldbalance,
            "avg_newbalance": avg_newbalance,
            "unique_destinations": unique_destinations,
            "avg_step": avg_step,
            "std_step": std_step,
            "amount_range": amount_range,
            #"balance_change": balance_change,
            "amount_per_transaction": amount_per_transaction,
            "destination_ratio": destination_ratio
        }])

        input_b_scaled = scaler_b.transform(input_b)
        raw_score = iso_model_b.decision_function(input_b_scaled)
        behavior_risk = float(normalize_anomaly_score(raw_score, risk_scaler_b)[0])

        st.session_state.behavior_risk = behavior_risk

        st.metric("Behavioral Risk", f"{behavior_risk:.3f}")
        st.success(f"Risk Band: {risk_band(behavior_risk)}")

# -----------------------------
# Final score page
# -----------------------------
elif page == "Final Hybrid Score":
    st.subheader("Final Hybrid Credit Risk Score")

    default_probability = st.session_state.default_probability
    behavior_risk = st.session_state.behavior_risk

    c1, c2 = st.columns(2)
    c1.metric(
        "Default Probability",
        "Not computed" if default_probability is None else f"{default_probability:.3f}"
    )
    c2.metric(
        "Behavioral Risk",
        "Not computed" if behavior_risk is None else f"{behavior_risk:.3f}"
    )

    if default_probability is not None and behavior_risk is not None:
        final_risk = 0.7 * default_probability + 0.3 * behavior_risk
        st.metric("Final Risk", f"{final_risk:.3f}")
        st.success(f"Final Band: {risk_band(final_risk)}")
        st.progress(float(final_risk))
    else:
        st.warning("Please compute both Model A and Model B scores first.")