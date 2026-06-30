import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load('model.pkl')
except:
    st.error("Model tidak ditemukan! Pastikan file model.pkl ada.")
    st.stop()

# =========================
# UI
# =========================
st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Masukkan data customer untuk prediksi churn")

# =========================
# INPUT USER
# =========================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Income", 1000, 100000, 5000)

with col2:
    score = st.number_input("Spending Score", 1, 100, 50)
    tenure = st.number_input("Tenure (bulan)", 1, 60, 12)

# =========================
# DATAFRAME
# =========================
input_df = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'SpendingScore': [score],
    'Tenure': [tenure]
})

st.subheader("Input Data")
st.write(input_df)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Predict"):
    try:
        result = model.predict(input_df)

        if result[0] == 1:
            st.error("❌ Customer berpotensi CHURN")
        else:
            st.success("✅ Customer kemungkinan STAY")

    except Exception as e:
        st.error("Terjadi error saat prediksi")
        st.text(e)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Info Model")
st.sidebar.write("""
Model digunakan untuk memprediksi customer churn
berdasarkan data:

- Age
- Income
- Spending Score
- Tenure
""")
