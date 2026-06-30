import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_churn.pkl")

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #aaa;
    margin-bottom: 30px;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan data customer untuk prediksi churn</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Income", 1000, 10000, 5000)

with col2:
    spending = st.slider("Spending Score", 1, 100, 50)
    tenure = st.slider("Tenure (bulan)", 1, 60, 20)

st.markdown('</div>', unsafe_allow_html=True)

input_df = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'SpendingScore': [spending],
    'Tenure': [tenure]
})

st.write("### 📋 Input Data")
st.dataframe(input_df)

if st.button("🔍 Predict"):
    result = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)

    st.write("### 📊 Hasil Prediksi")

    if result == 1:
        st.error("❌ Customer berpotensi CHURN")
    else:
        st.success("✅ Customer kemungkinan STAY")

    st.info(f"📈 Probabilitas Churn: {proba[0][1]:.2f}")
