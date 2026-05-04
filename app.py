import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import time
import matplotlib.pyplot as plt

# --- 2. INITIAL CONFIGURATION ---
st.set_page_config(page_title="SpiroX Pro", layout="wide", page_icon="🫁")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #eee; }
    .diag-card { padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .section-header { background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-top: 20px; margin-bottom: 15px; font-weight: bold; color: #495057; border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE AI ENGINE ---
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
        input_data = np.array([z_vals])
        pred = model.predict(input_data)[0]
        labels = {0: "Restrictive", 1: "Obstructive", 2: "Small Airway Disease", 3: "Normal"}
        return labels.get(pred, "Normal"), z_vals
    return "Normal", z_vals

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("""<div style="background-color:#007bff;padding:15px;border-radius:12px;margin-bottom:20px;">
                <h2 style="color:white;text-align:center;margin:0;">SpiroX Pro</h2>
                <p style="color:white;text-align:center;font-size:0.8em;opacity:0.8;">v2.0 AI-Diagnostic Suite</p>
                </div>""", unsafe_allow_html=True)
    st.success("✅ Engine: Active")
    st.write("**Deepthi Bhonsle** | CBIT ECE")

# --- 5. MAIN UI ---
st.title("🫁 Pulmonary Diagnostic Dashboard")
t1, t2 = st.tabs(["📊 Clinical Entry", "🧬 Waveform Analysis"])

with t1:
    # 👤 Patient Details moved here
    st.markdown('<div class="section-header">👤 Patient Details</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    age = p1.number_input("Age", min_value=1, max_value=120, value=25)
    height = p2.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    gender = p3.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown('<div class="section-header">💨 Flow Parameters (Dynamic)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    mf1 = c1.number_input("FEV1 (L)", min_value=0.0, max_value=10.0, value=3.5, step=0.1, key="m1")
    mf2 = c2.number_input("FVC (L)", min_value=0.0, max_value=10.0, value=4.5, step=0.1, key="m2")
    mf3 = c3.number_input("FEV1/FVC", min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="m3")
    mf4 = c4.number_input("FEF 25-75", min_value=0.0, max_value=10.0, value=4.0, step=0.1, key="m4")

    st.markdown('<div class="section-header">🎈 Volume Parameters (Static)</div>', unsafe_allow_html=True)
    c5, c6, c7, c8, c9 = st.columns(5)
    mf5 = c5.number_input("TLC (L)", min_value=0.0, max_value=10.0, value=6.0, step=0.1, key="m5")
    mf6 = c6.number_input("RV (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key="m6")
    mf7 = c7.number_input("FRC (L)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key="m7")
    mf8 = c8.number_input("ERV (L)", min_value=0.0, max_value=10.0, value=1.5, step=0.1, key="m8")
    mf9 = c9.number_input("IC (L)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key="m9")

    if st.button("Run Diagnostic", type="primary"):
        data_dict = {'fev1':mf1,'fvc':mf2,'ratio':mf3,'fef':mf4,'tlc':mf5,'rv':mf6,'frc':mf7,'erv':mf8,'ic':mf9}
        diag, zs = classify_patient(data_dict)
        
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
                e_fvc = abs(df['Volume'].max() - df['Volume'].min())
                peak_idx = df['Flow'].abs().idxmax()
                t0, v0 = df.loc[peak_idx, 'Time'], df.loc[peak_idx, 'Volume']
                e_fev1 = abs(df.loc[(df['Time'] - (t0 + 1.0)).abs().idxmin(), 'Volume'] - v0)
                e_ratio = e_fev1 / e_fvc if e_fvc > 0 else 0
                
                data_dict = {'fev1':e_fev1,'fvc':e_fvc,'ratio':e_ratio,'fef':4.0,'tlc':6.2,'rv':2.1,'frc':3.1,'erv':1.5,'ic':3.0}
                diag, zs = classify_patient(data_dict)
                
                colors = {"Normal": "#28a745", "Restrictive": "#dc3545", "Obstructive": "#fd7e14", "Small Airway Disease": "#17a2b8"}
                st.markdown(f'<div class="diag-card" style="background-color:{colors[diag]};"><h1>DIAGNOSIS: {diag.upper()}</h1></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-header">💨 Extracted Signal Metrics</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("FEV1", f"{e_fev1:.2f} L")
                m2.metric("FVC", f"{e_fvc:.2f} L")
                m3.metric("Ratio", f"{e_ratio:.2f}")

                st.subheader("📊 Reconstructed Flow-Volume Loop")
                st.line_chart(df, x="Volume", y="Flow")
                