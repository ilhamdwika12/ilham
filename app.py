import streamlit as st
import pandas as pd
import joblib
import math

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Churn Radar",
    page_icon="🛰️",
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
# DESIGN TOKENS
# =========================
BG = "#0B1120"
SURFACE = "#121B2E"
SURFACE_ALT = "#0F1729"
BORDER = "#22304A"
TEXT = "#E6EAF2"
MUTED = "#7C8AA5"
TEAL = "#22D3EE"
AMBER = "#FBBF24"
CORAL = "#FB7185"

# =========================
# GLOBAL CSS
# =========================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 15% 0%, #14213D 0%, {BG} 45%);
    color: {TEXT};
}}

section[data-testid="stSidebar"] {{
    background-color: {SURFACE_ALT};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{
    color: {TEXT};
}}

/* Eyebrow + hero */
.eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: {TEAL};
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    color: {TEXT};
    line-height: 1.15;
}}
.hero-sub {{
    color: {MUTED};
    font-size: 0.98rem;
    margin-top: 0.5rem;
    max-width: 540px;
}}

/* Panels */
.panel {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
}}
.panel-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 1rem;
}}

/* Number inputs */
div[data-testid="stNumberInput"] label p {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED} !important;
}}
div[data-testid="stNumberInput"] input {{
    background-color: {SURFACE_ALT} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace;
}}
div[data-testid="stNumberInput"] button {{
    background-color: {SURFACE_ALT} !important;
    border-color: {BORDER} !important;
    color: {MUTED} !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 0.7rem 0.9rem 0.5rem 0.9rem;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MUTED} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Space Grotesk', sans-serif;
    color: {TEXT} !important;
}}

/* Button */
div.stButton > button {{
    background: linear-gradient(135deg, {TEAL} 0%, #0EA5B5 100%);
    color: #06141F;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    width: 100%;
    letter-spacing: 0.02em;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(34, 211, 238, 0.25);
}}

/* Verdict badges */
.verdict-churn, .verdict-stay {{
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    margin-top: 1rem;
}}
.verdict-churn {{
    background-color: rgba(251, 113, 133, 0.1);
    border: 1px solid rgba(251, 113, 133, 0.4);
}}
.verdict-stay {{
    background-color: rgba(34, 211, 238, 0.1);
    border: 1px solid rgba(34, 211, 238, 0.4);
}}
.verdict-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
}}
.verdict-sub {{
    color: {MUTED};
    font-size: 0.88rem;
    margin-top: 0.3rem;
}}

hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO
# =========================
st.markdown(f"""
<div class="eyebrow">CUSTOMER RISK INSTRUMENT</div>
<div class="hero-title">📡 Churn Radar</div>
<div class="hero-sub">Masukkan profil customer di bawah, dan radar akan menunjukkan
seberapa dekat customer tersebut dengan potensi churn.</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# LAYOUT: INPUT (kiri) + GAUGE (kanan)
# =========================
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Profil Customer</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("🎂 Age", 18, 100, 30)
        score = st.number_input("⭐ Spending Score", 1, 100, 50)
    with c2:
        income = st.number_input("💰 Income", 1000, 100000, 5000)
        tenure = st.number_input("📅 Tenure (bulan)", 1, 60, 12)

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Age", f"{age}")
    m2.metric("Income", f"${income:,}")
    m3.metric("Score", score)
    m4.metric("Tenure", f"{tenure}b")

    with st.expander("Lihat data mentah (DataFrame)"):
        input_df = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'SpendingScore': [score],
            'Tenure': [tenure]
        })
        st.dataframe(input_df, use_container_width=True)

    st.write("")
    predict_clicked = st.button("🔍 Scan Customer")

input_df = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'SpendingScore': [score],
    'Tenure': [tenure]
})

# =========================
# GAUGE BUILDER
# =========================
def render_gauge(frac, label_top, color_hex):
    """frac: 0.0 (aman) - 1.0 (churn). Render gauge semicircle SVG dengan needle."""
    frac = max(0.0, min(1.0, frac))
    r = 90
    cx, cy = 100, 100
    arc_len = math.pi * r
    dash_offset = arc_len * (1 - frac)
    needle_deg = -90 + 180 * frac

    svg = f"""
    <svg viewBox="0 0 200 120" width="100%" height="auto" style="max-width:320px;">
        <defs>
            <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="{TEAL}" />
                <stop offset="50%" stop-color="{AMBER}" />
                <stop offset="100%" stop-color="{CORAL}" />
            </linearGradient>
        </defs>
        <path d="M10,100 A90,90 0 0,1 190,100" stroke="{BORDER}" stroke-width="14"
              fill="none" stroke-linecap="round" />
        <path d="M10,100 A90,90 0 0,1 190,100" stroke="url(#gaugeGrad)" stroke-width="14"
              fill="none" stroke-linecap="round"
              stroke-dasharray="{arc_len:.2f} {arc_len:.2f}"
              stroke-dashoffset="{dash_offset:.2f}" />
        <line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy-65}" stroke="{TEXT}" stroke-width="4"
              stroke-linecap="round" transform="rotate({needle_deg:.1f} {cx} {cy})" />
        <circle cx="{cx}" cy="{cy}" r="7" fill="{TEXT}" />
        <text x="100" y="98" text-anchor="middle" font-family="JetBrains Mono, monospace"
              font-size="22" font-weight="600" fill="{color_hex}">{label_top}</text>
    </svg>
    """
    return svg

with right:
    st.markdown('<div class="panel" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Risk Gauge</div>', unsafe_allow_html=True)

    if predict_clicked:
        try:
            result = model.predict(input_df)
            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(input_df)[0]
                except Exception:
                    proba = None

            if proba is not None and len(proba) > 1:
                churn_prob = float(proba[1])
            else:
                churn_prob = 1.0 if result[0] == 1 else 0.0

            gauge_color = CORAL if churn_prob >= 0.5 else TEAL
            st.markdown(
                render_gauge(churn_prob, f"{churn_prob*100:.0f}%", gauge_color),
                unsafe_allow_html=True
            )

            if result[0] == 1:
                st.markdown(f"""
                <div class="verdict-churn">
                    <p class="verdict-title">❌ Berpotensi CHURN</p>
                    <p class="verdict-sub">Pertimbangkan strategi retensi untuk customer ini.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-stay">
                    <p class="verdict-title">✅ Kemungkinan STAY</p>
                    <p class="verdict-sub">Customer ini diprediksi akan tetap loyal.</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error("Terjadi error saat prediksi")
            st.text(e)
    else:
        st.markdown(render_gauge(0.0, "—", MUTED), unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{MUTED}; font-size:0.85rem; margin-top:0.5rem;">'
            f'Klik "Scan Customer" untuk melihat hasil</p>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🛰️ Info Model")
    st.markdown(
        f'<p style="color:{MUTED}; font-size:0.9rem;">Model ini memprediksi potensi '
        f'<b style="color:{TEXT}">customer churn</b> berdasarkan beberapa fitur utama.</p>',
        unsafe_allow_html=True
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
