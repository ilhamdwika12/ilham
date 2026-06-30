import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load('model.pkl')
except Exception:
    st.error("⚠️ Model tidak ditemukan! Pastikan file model.pkl ada di folder yang sama.")
    st.stop()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main-header {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    .input-card {
        background-color: #f8f9fc;
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #e6e8f0;
        margin-bottom: 1rem;
    }
    .result-card-churn {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 1px solid #f87171;
        text-align: center;
    }
    .result-card-stay {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #4ade80;
        text-align: center;
    }
    .result-card-churn h2, .result-card-stay h2 {
        margin: 0;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(79, 70, 229, 0.35);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="main-header">
    <h1>📊 Customer Churn Prediction</h1>
    <p>Masukkan data customer untuk memprediksi potensi churn secara instan</p>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
st.markdown("### 📝 Data Customer")

with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input("🎂 Age", 18, 100, 30)
    with col2:
        income = st.number_input("💰 Income", 1000, 100000, 5000)
    with col3:
        score = st.number_input("⭐ Spending Score", 1, 100, 50)
    with col4:
        tenure = st.number_input("📅 Tenure (bulan)", 1, 60, 12)
    st.markdown('</div>', unsafe_allow_html=True)

input_df = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'SpendingScore': [score],
    'Tenure': [tenure]
})

# Ringkasan singkat input dalam bentuk metric, lebih enak dilihat dari tabel mentah
m1, m2, m3, m4 = st.columns(4)
m1.metric("Age", f"{age} thn")
m2.metric("Income", f"${income:,}")
m3.metric("Spending Score", score)
m4.metric("Tenure", f"{tenure} bln")

with st.expander("Lihat data mentah (DataFrame)"):
    st.dataframe(input_df, use_container_width=True)

st.write("")

# =========================
# PREDIKSI
# =========================
predict_col = st.columns([1, 1, 1])[1]
with predict_col:
    predict_clicked = st.button("🔍 Predict Churn")

if predict_clicked:
    try:
        result = model.predict(input_df)

        # Coba ambil probabilitas jika model mendukungnya, untuk tampilan lebih informatif
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)[0]
            except Exception:
                proba = None

        st.write("")
        if result[0] == 1:
            st.markdown("""
            <div class="result-card-churn">
                <h2>❌ Customer berpotensi CHURN</h2>
                <p>Pertimbangkan strategi retensi untuk customer ini.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card-stay">
                <h2>✅ Customer kemungkinan STAY</h2>
                <p>Customer ini diprediksi akan tetap loyal.</p>
            </div>
            """, unsafe_allow_html=True)

        if proba is not None:
            st.write("")
            st.markdown("**Probabilitas Prediksi**")
            churn_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            st.progress(min(max(churn_prob, 0.0), 1.0))
            st.caption(f"Probabilitas churn: {churn_prob*100:.1f}%")

    except Exception as e:
        st.error("Terjadi error saat prediksi")
        st.text(e)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 📌 Info Model")
    st.info(
        "Model ini digunakan untuk memprediksi potensi **customer churn** "
        "berdasarkan beberapa fitur utama."
    )
    st.markdown("**Fitur yang digunakan:**")
    st.markdown("""
- 🎂 Age
- 💰 Income
- ⭐ Spending Score
- 📅 Tenure
""")
    st.divider()
    st.caption("Dibuat dengan ❤️ menggunakan Streamlit")
