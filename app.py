import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessor
model = joblib.load("bank_marketing_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Page Configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Bank Marketing Prediction System")
st.write("Predict whether a customer will subscribe to a bank term deposit.")

st.divider()

st.header("Enter Customer Details")

# Customer Input Form
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 35)

        job = st.selectbox("Job", [
            "admin.", "blue-collar", "entrepreneur",
            "housemaid", "management", "retired",
            "self-employed", "services", "student",
            "technician", "unemployed", "unknown"
        ])

        marital = st.selectbox("Marital Status", [
            "married", "single", "divorced", "unknown"
        ])

        education = st.selectbox("Education", [
            "basic.4y", "basic.6y", "basic.9y",
            "high.school", "illiterate",
            "professional.course", "university.degree",
            "unknown"
        ])

        default = st.selectbox("Credit Default", [
            "no", "yes", "unknown"
        ])

        housing = st.selectbox("Housing Loan", [
            "no", "yes", "unknown"
        ])

        loan = st.selectbox("Personal Loan", [
            "no", "yes", "unknown"
        ])

    with col2:
        contact = st.selectbox("Contact Type", [
            "cellular", "telephone"
        ])

        month = st.selectbox("Month", [
            "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct",
            "nov", "dec"
        ])

        day_of_week = st.selectbox("Day of Week", [
            "mon", "tue", "wed", "thu", "fri"
        ])

        campaign = st.number_input(
            "Number of Campaign Contacts",
            min_value=1, value=2
        )

        pdays = st.number_input(
            "Days Since Previous Contact",
            min_value=0, value=999
        )

        previous = st.number_input(
            "Previous Contacts",
            min_value=0, value=0
        )

        poutcome = st.selectbox("Previous Campaign Outcome", [
            "nonexistent", "failure", "success"
        ])

    st.subheader("Economic Indicators")

    col3, col4, col5 = st.columns(3)

    with col3:
        emp_var_rate = st.number_input(
            "Employment Variation Rate", value=1.1
        )

    with col4:
        cons_price_idx = st.number_input(
            "Consumer Price Index", value=93.994
        )

    with col5:
        cons_conf_idx = st.number_input(
            "Consumer Confidence Index", value=-36.4
        )

    euribor3m = st.number_input(
        "Euribor 3 Month Rate", value=4.857
    )

    nr_employed = st.number_input(
        "Number of Employees", value=5191.0
    )

    submit = st.form_submit_button(
        "Predict Customer Subscription"
    )

# Prediction
if submit:

    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed
    }])

    try:
        processed_data = preprocessor.transform(input_data)

        prediction = model.predict(processed_data)

        st.divider()
        st.header("Prediction Result")

        if prediction[0] == 1:
            st.success(
                "YES! Customer is likely to subscribe to a term deposit."
            )
        else:
            st.warning(
                "NO! Customer is unlikely to subscribe to a term deposit."
            )

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(processed_data)[0]

            st.subheader("Prediction Probability")

            st.write(
                f"Subscription Probability: {probability[1] * 100:.2f}%"
            )

            st.progress(float(probability[1]))

    except Exception as e:
        st.error(f"Prediction Error: {e}")