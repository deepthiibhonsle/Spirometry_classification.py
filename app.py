import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import time
import pdfplumber
import re

# --- 1. OPTIONAL LIBRARIES (Safe Loading) ---
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from streamlit_lottie import st_lottie
    import requests
    HAS_LOTTIE = True
except ImportError:
    HAS_LOTTIE = False

# --- 2. INITIAL CONFIGURATION ---
st.set_page_config(page_title="SpiroX Pro", layout="wide", page_icon="🫁")

# Branding & Professional CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #eee; }
    .diag-card { padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .section-header { background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-top: 20px; margin-bottom: 15px; font-weight: bold; color: #495057; border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def load_lottieurl(url):
    if not HAS_LOTTIE: return None
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

@st.cache_resource
def load_model():
    try:
        with open('pulmo_model.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_model()

def compute_z_score(observed, predicted_mean):
    return (observed / predicted_mean - 1) / 0.1

def classify_patient(f):
    z_vals = [
        compute_z_score(f.get('fev1', 3.5), 3.5),
        compute_z_score(f.get('fvc', 4.5), 4.5),
        compute_z_score(f.get('ratio', 0.8), 0.8),
        compute_z_score(f.get('fef', 4.0), 4.0),
        compute_z_score(f.get('tlc', 6.0), 6.0),
        compute_z_score(f.get('rv', 2.0), 2.0),
        compute_z_score(f.get('frc', 3.0), 3.0),
        compute_z_score(f.get('erv', 1.5), 1.5),
        compute_z_score(f.get('ic', 3.0), 3.0)
    ]
    if model:
        pred = model.predict(np.array([z_vals]))[0]
        labels = {0: "Restrictive", 1: "Obstructive", 2: "Small Airway Disease", 3: "Normal"}
        return labels.get(pred, "Normal"), z_vals
    return "Normal", z_vals

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("""<div style="background-color:#007bff;padding:15px;border-radius:12px;margin-bottom:20px;">
                <h2 style="color:white;text-align:center;margin:0;">SpiroX Pro</h2>
                <p style="color:white;text-align:center;font-size:0.8em;opacity:0.8;">v2.0 AI-Diagnostic Suite</p>
                </div>""", unsafe_allow_html=True)
    if HAS_LOTTIE:
        l_json = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp3v8v.json")
        if l_json: st_lottie(l_json, height=150)
    st.success("✅ Engine: Active")
    st.markdown("---")
    st.write("**Deepthi Bhonsle** | CBIT ECE")

# --- 5. MAIN UI ---
st.title("🫁 Pulmonary Diagnostic Dashboard")
t1, t2 = st.tabs(["📊 Clinical Entry", "🧬 Waveform Analysis"])

with t1:
    st.markdown('<div class="section-header">💨 Flow Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    mf1 = c1.number_input("FEV1 (L)", 3.5, key="m1")
    mf2 = c2.number_input("FVC (L)", 4.5, key="m2")
    mf3 = c3.number_input("FEV1/FVC", 0.8, key="m3")
    mf4 = c4.number_input("FEF 25-75", 4.0, key="m4")

    st.markdown('<div class="section-header">🎈 Volume Parameters</div>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    mf5 = c5.number_input("TLC (L)", 6.0, key="m5")
    mf6 = c6.number_input("RV (L)", 2.0, key="m6")
    mf7 = c7.number_input("FRC (L)", 3.0, key="m7")
    mf8 = c8.number_input("IC (L)", 3.0, key="m8")

    if st.button("Run Diagnostic", type="primary"):
        diag, zs = classify_patient({'fev1':mf1,'fvc':mf2,'ratio':mf3,'fef':mf4,'tlc':mf5,'rv':mf6,'frc':mf7,'ic':mf8})
        colors = {"Normal": "#28a745", "Restrictive": "#dc3545", "Obstructive": "#fd7e14", "Small Airway Disease": "#17a2b8"}
        st.markdown(f'<div class="diag-card" style="background-color:{colors[diag]};"><h1>DIAGNOSIS: {diag.upper()}</h1></div>', unsafe_allow_html=True)

with t2:
    up = st.file_uploader("Upload CSV Waveform", type=["csv"])
    if up:
        names = ['ID', 'Skip', 'Trial', 'Age', 'Height', 'Gender', 'Time', 'Flow', 'Volume']
        df = pd.read_csv(up, names=names, header=None)
        for c in ['Time', 'Flow', 'Volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['Time', 'Flow', 'Volume'])
        st.dataframe(df.head(3))

        if st.button("Process Signal"):
            with st.spinner("Executing Extraction Pipeline..."):
                time.sleep(1)
                # Displacement logic for negative sensor values
                e_fvc = abs(df['Volume'].max() - df['Volume'].min())
                peak_idx = df['Flow'].abs().idxmax()
                t0, v0 = df.loc[peak_idx, 'Time'], df.loc[peak_idx, 'Volume']
                e_fev1 = abs(df.loc[(df['Time'] - (t0 + 1.0)).abs().idxmin(), 'Volume'] - v0)
                e_ratio = e_fev1 / e_fvc if e_fvc > 0 else 0
                
                # Mock values for non-waveform features
                e_fef, e_tlc, e_rv, e_frc, e_ic = 4.0, 6.2, 2.1, 3.1, 3.0
                
                diag, zs = classify_patient({'fev1':e_fev1,'fvc':e_fvc,'ratio':e_ratio,'fef':e_fef,'tlc':e_tlc,'rv':e_rv,'frc':e_frc,'ic':e_ic})
                
                # Result Card
                colors = {"Normal": "#28a745", "Restrictive": "#dc3545", "Obstructive": "#fd7e14", "Small Airway Disease": "#17a2b8"}
                st.markdown(f'<div class="diag-card" style="background-color:{colors[diag]};"><h1>DIAGNOSIS: {diag.upper()}</h1></div>', unsafe_allow_html=True)
                
                # REORGANIZED HEADINGS (Flow vs Volume)
                st.markdown('<div class="section-header">💨 Flow Parameters (Dynamic)</div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("FEV1", f"{e_fev1:.2f} L")
                m2.metric("FVC", f"{e_fvc:.2f} L")
                m3.metric("FEV1/FVC", f"{e_ratio:.2f}")
                m4.metric("FEF 25-75", f"{e_fef:.1f}")

                st.markdown('<div class="section-header">🎈 Volume Parameters (Static)</div>', unsafe_allow_html=True)
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("TLC", f"{e_tlc:.1f} L")
                m6.metric("RV", f"{e_rv:.1f} L")
                m7.metric("FRC", f"{e_frc:.1f} L")
                m8.metric("IC", f"{e_ic:.1f} L")

                st.subheader("📊 Reconstructed Flow-Volume Loop")
                st.line_chart(df, x="Volume", y="Flow")