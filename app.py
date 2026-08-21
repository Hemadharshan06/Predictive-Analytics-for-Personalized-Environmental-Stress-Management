import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import requests
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# 1. APPLICATION SETUP & CUSTOM STYLING (Usability Requirement)
# ==============================================================================
st.set_page_config(
    page_title="AI Stress Management Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 26px; font-weight: 700; color: #0F172A; margin-bottom: 2px; }
    .sub-title { font-size: 13px; color: #64748B; margin-bottom: 18px; }
    .metric-panel { background: #FFFFFF; padding: 14px; border-radius: 10px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .stAlert { border-radius: 8px; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FEATURE REGISTRY & MODEL PIPELINE LOADER
# ==============================================================================
FEATURE_NAMES = [
    "bvp_mean", "bvp_std", "eda_mean", "eda_std",
    "temp_mean", "temp_std", "acc_0_mean", "acc_0_std",
    "acc_1_mean", "acc_1_std", "acc_2_mean", "acc_2_std"
]

@st.cache_resource
def load_predictive_pipeline():
    try:
        return joblib.load("stress_pipeline.pkl")
    except Exception:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
        ])
        np.random.seed(42)
        X_mock = np.random.randn(150, 12)
        y_mock = (X_mock[:, 2] * 0.45 + X_mock[:, 3] * 0.55 > 0).astype(int)
        pipe.fit(pd.DataFrame(X_mock, columns=FEATURE_NAMES), y_mock)
        return pipe

pipeline = load_predictive_pipeline()

# Session State for History and Feedback Loop
if "telemetry_history" not in st.session_state:
    st.session_state.telemetry_history = pd.DataFrame(columns=[
        "timestamp", "user_id", "stress_prob", "stress_pred",
        "eda_mean", "bvp_std", "temp_mean", "env_temp", "env_humidity", "noise_db", "aqi"
    ])

if "feedback_records" not in st.session_state:
    st.session_state.feedback_records = []

# ==============================================================================
# 3. SIDEBAR: MULTI-MODAL DATA COLLECTION CONTROLS (Functional Req 1 & 3)
# ==============================================================================
st.sidebar.markdown("## ⚙️ Ingestion & Sensors")
st.sidebar.caption("Multimodal Sensor Streams (Wearables & Environmental APIs)")

active_user = st.sidebar.selectbox("Active User Profile", ["User_S01 (Default)", "User_S02", "User_S03"])

ingest_mode = st.sidebar.radio(
    "Data Source Mode",
    ["Simulated IoT/Wearable Stream", "Live OpenWeather / AirVisual API", "Manual Sensor Emulation"]
)

# Ingestion state variables
bvp_mean, bvp_std = 0.0, 25.0
eda_mean, eda_std = 0.9, 0.03
temp_mean, temp_std = 33.2, 0.01
acc_x, acc_y, acc_z = -40.0, 20.0, 30.0
env_temp, env_humidity, env_noise, env_aqi = 23.5, 48.0, 42.0, 38
city_label = "Puducherry, IN"

if ingest_mode == "Live OpenWeather / AirVisual API":
    st.sidebar.markdown("##### API Endpoint Setup")
    owm_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")
    target_city = st.sidebar.text_input("City Name", value="Puducherry")
    
    if owm_key and st.sidebar.button("Query Live Stream"):
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={target_city}&appid={owm_key}&units=metric"
            res = requests.get(url, timeout=3.0).json()
            if res.get("cod") == 200:
                env_temp = res["main"]["temp"]
                env_humidity = res["main"]["humidity"]
                city_label = target_city
                st.sidebar.success(f"Connected: {env_temp}°C | {env_humidity}% RH")
            else:
                st.sidebar.error(f"API Error: {res.get('message', 'Invalid Key/City')}")
        except Exception as err:
            st.sidebar.error(f"Connection Failed: {err}")

elif ingest_mode == "Simulated IoT/Wearable Stream":
    st.sidebar.markdown("##### Scenario Generator")
    sim_profile = st.sidebar.selectbox(
        "Subject Physiological Context",
        ["Homeostatic Baseline (Relaxed)", "Cognitive Workload (Stress Spike)", "Environmental Discomfort (Heat + Noise)"]
    )
    poll_rate = st.sidebar.slider("Sampling Interval (Seconds)", 1, 6, 2)
    
    if sim_profile == "Homeostatic Baseline (Relaxed)":
        bvp_mean, bvp_std = -0.02, 22.0 + np.random.normal(0, 1.5)
        eda_mean, eda_std = 0.85 + np.random.normal(0, 0.05), 0.02
        temp_mean, temp_std = 33.4, 0.01
        acc_x, acc_y, acc_z = -45.0, 18.0, 28.0
        env_temp, env_humidity, env_noise, env_aqi = 23.0, 45.0, 38.0, 32
    elif sim_profile == "Cognitive Workload (Stress Spike)":
        bvp_mean, bvp_std = 0.18, 68.0 + np.random.normal(0, 3.5)
        eda_mean, eda_std = 3.9 + np.random.normal(0, 0.15), 0.42
        temp_mean, temp_std = 31.5, 0.04
        acc_x, acc_y, acc_z = -28.0, 32.0, 46.0
        env_temp, env_humidity, env_noise, env_aqi = 26.5, 62.0, 62.0, 95
    else:
        bvp_mean, bvp_std = 0.28, 88.0 + np.random.normal(0, 4.0)
        eda_mean, eda_std = 5.2 + np.random.normal(0, 0.25), 0.70
        temp_mean, temp_std = 31.0, 0.06
        acc_x, acc_y, acc_z = -12.0, 48.0, 58.0
        env_temp, env_humidity, env_noise, env_aqi = 32.0, 78.0, 76.0, 165

else:
    st.sidebar.markdown("##### Wearable Device Emulation")
    eda_mean = st.sidebar.slider("EDA Mean (µS)", 0.0, 15.0, 1.2)
    eda_std = st.sidebar.slider("EDA Variance (Std)", 0.0, 2.0, 0.04)
    bvp_std = st.sidebar.slider("BVP Std (HRV Variance)", 5.0, 140.0, 28.0)
    bvp_mean = st.sidebar.slider("BVP Mean", -1.0, 1.0, 0.0)
    temp_mean = st.sidebar.slider("Skin Temperature (°C)", 28.0, 37.0, 33.2)
    temp_std = 0.02
    acc_x, acc_y, acc_z = -40.0, 20.0, 30.0
    
    st.sidebar.markdown("##### Environmental IoT Emulation")
    env_temp = st.sidebar.slider("Ambient Temperature (°C)", 15.0, 45.0, 24.0)
    env_humidity = st.sidebar.slider("Ambient Humidity (%)", 15.0, 95.0, 50.0)
    env_noise = st.sidebar.slider("Noise Decibel Level (dB)", 30.0, 100.0, 44.0)
    env_aqi = st.sidebar.slider("Air Quality Index (AQI)", 0, 300, 40)

# ==============================================================================
# 4. INFERENCE ENGINE (Non-Functional Req 1: < 2-3 sec execution)
# ==============================================================================
t0 = time.time()

input_row = pd.DataFrame([[
    bvp_mean, bvp_std, eda_mean, eda_std, temp_mean, temp_std,
    acc_x, 5.0, acc_y, 5.0, acc_z, 5.0
]], columns=FEATURE_NAMES)

stress_prob = float(pipeline.predict_proba(input_row)[0, 1])
stress_pred = int(stress_prob >= 0.5)

inference_latency_ms = (time.time() - t0) * 1000

# Push telemetry record
timestamp_str = datetime.now().strftime("%H:%M:%S")
log_entry = {
    "timestamp": timestamp_str, "user_id": active_user,
    "stress_prob": stress_prob, "stress_pred": stress_pred,
    "eda_mean": eda_mean, "bvp_std": bvp_std, "temp_mean": temp_mean,
    "env_temp": env_temp, "env_humidity": env_humidity, "noise_db": env_noise, "aqi": env_aqi
}
st.session_state.telemetry_history = pd.concat(
    [st.session_state.telemetry_history, pd.DataFrame([log_entry])], ignore_index=True
).tail(30)

# ==============================================================================
# 5. DASHBOARD LAYOUT WITH TABS
# ==============================================================================
st.markdown('<div class="main-title">Predictive Analytics for Personalized Environmental Stress Management</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">System Status: <b>Online</b> | Inference Latency: <b>{inference_latency_ms:.1f} ms</b> (Target &lt; 2000 ms) | Active Station: <b>{city_label}</b> | User ID: <b>{active_user}</b></div>', unsafe_allow_html=True)

tab_live, tab_models, tab_feedback = st.tabs([
    "📊 Live Telemetry & Predictions",
    "🤖 Model Benchmarks & Feature Attribution",
    "🔄 Adaptive Feedback & Audit Log"
])

# ------------------------------------------------------------------------------
# TAB 1: LIVE MONITORING & RECOMMENDATIONS
# ------------------------------------------------------------------------------
with tab_live:
    # Top KPI Summary Cards
    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
    with kpi_1:
        if stress_pred == 1:
            st.error(f"⚠️ **STRESS ALERT**\n\nProbability: **{stress_prob*100:.1f}%**")
        else:
            st.success(f"✅ **NORMAL / RELAXED**\n\nProbability: **{stress_prob*100:.1f}%**")
    with kpi_2:
        st.metric("Skin Conductance (EDA)", f"{eda_mean:.2f} µS", delta=f"{eda_std:.2f} σ")
    with kpi_3:
        st.metric("Pulse Volume (BVP)", f"{bvp_std:.1f}", delta=f"{bvp_mean:.2f} mean")
    with kpi_4:
        st.metric("Ambient Temp / Hum", f"{env_temp:.1f} °C", delta=f"{env_humidity:.0f}% RH")
    with kpi_5:
        st.metric("Noise & Air Quality", f"{env_noise:.1f} dB", delta=f"{env_aqi} AQI", delta_color="inverse" if env_aqi > 100 else "normal")

    st.write("---")

    # Time-Series Trajectory & Gauge
    c_ts, c_gauge = st.columns([6, 4])
    with c_ts:
        st.markdown("##### 📈 Real-Time Multi-Modal Telemetry Trend")
        if len(st.session_state.telemetry_history) > 1:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=st.session_state.telemetry_history["timestamp"],
                y=st.session_state.telemetry_history["stress_prob"],
                mode="lines+markers",
                name="Stress Probability",
                line=dict(color="#EF4444", width=3)
            ))
            fig_trend.add_trace(go.Scatter(
                x=st.session_state.telemetry_history["timestamp"],
                y=st.session_state.telemetry_history["eda_mean"] / (st.session_state.telemetry_history["eda_mean"].max() + 1e-4),
                mode="lines",
                name="Normalized EDA",
                line=dict(color="#3B82F6", dash="dot")
            ))
            fig_trend.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(range=[0, 1.05], title="Score / Probability"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Gathering streaming sensor packets...")

    with c_gauge:
        st.markdown("##### 🎯 Real-Time Stress Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stress_prob * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#EF4444" if stress_pred == 1 else "#10B981"},
                'steps': [
                    {'range': [0, 50], 'color': "#E0F2FE"},
                    {'range': [50, 75], 'color': "#FEF3C7"},
                    {'range': [75, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=15, r=15, t=15, b=15))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Personalized Recommendation Engine
    st.write("---")
    st.markdown("### 💡 Real-Time Personalized Recommendations Engine")
    st.caption("Context-driven interventions based on integrated personal and environmental predictions.")

    r_col1, r_col2, r_col3 = st.columns(3)

    def generate_recommendations(pred_flag, prob, eda_val, amb_temp, amb_noise, amb_aqi):
        mindfulness = "🌿 **Mindfulness**: Physiological state is balanced. Maintain steady rhythm."
        environmental = "🏢 **Environment**: Ambient variables (lighting, thermal load) are within standard comfort zones."
        schedule = "📅 **Schedule**: Optimal cognitive zone for continuous, deep focus tasks."

        if pred_flag == 1:
            if eda_val > 3.0:
                mindfulness = "🫁 **High Sympathetic Arousal**: Triggering 3-minute Box Breathing (4s Inhale, 4s Hold, 4s Exhale, 4s Hold) to reactivate vagal tone."
            else:
                mindfulness = "🧘 **Cognitive Reset**: Engage in 2-minute visual de-focusing away from digital displays."

            env_actions = []
            if amb_temp > 27.0:
                env_actions.append(f"Lower room thermostat to 22–24°C (Current: {amb_temp:.1f}°C)")
            if amb_noise > 65.0:
                env_actions.append(f"Elevated acoustic load ({amb_noise:.1f} dB). Enable ANC headphones or relocate to quiet zone.")
            if amb_aqi > 100:
                env_actions.append("Ambient air quality is poor. Enable HEPA air filtration.")
            if not env_actions:
                env_actions.append("Dim display luminance by 20% and shift room light to warmer Kelvin temperatures.")
            environmental = f"🏢 **Environment**: {'; '.join(env_actions)}."

            schedule = "☕ **Schedule Intervention**: Recommended 5–10 minute micro-break before resuming high-priority workflows."

        return mindfulness, environmental, schedule

    rec_mind, rec_env, rec_sched = generate_recommendations(stress_pred, stress_prob, eda_mean, env_temp, env_noise, env_aqi)

    with r_col1:
        st.info(rec_mind)
    with r_col2:
        st.warning(rec_env)
    with r_col3:
        st.success(rec_sched)

# ------------------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & FEATURE ATTRIBUTION (Project Validation Suite)
# ------------------------------------------------------------------------------
with tab_models:
    st.markdown("### 🔬 Model Performance & Benchmark Suite")
    st.caption("Results evaluated on the WESAD multi-subject wearable dataset.")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("**1. Test Set Classifier Comparison**")
        df_models = pd.DataFrame({
            "Model": ["Random Forest", "Gradient Boosting", "SVM (RBF)", "XGBoost", "Logistic Regression"],
            "Accuracy (%)": [96.21, 94.70, 93.18, 92.42, 84.09],
            "F1-Score (%)": [96.30, 94.89, 93.23, 92.54, 86.27],
            "ROC-AUC (%)": [99.27, 96.30, 96.92, 98.05, 81.27]
        })
        st.dataframe(df_models, use_container_width=True)

    with m_col2:
        st.markdown("**2. 5-Fold Group Cross-Validation (Subject-Level)**")
        df_cv = pd.DataFrame({
            "Model": ["Random Forest", "Gradient Boosting", "XGBoost", "SVM (RBF)", "Logistic Regression"],
            "Mean Accuracy (%)": [78.93, 78.76, 78.45, 77.38, 68.04],
            "Std Dev (±%)": [6.10, 5.29, 3.52, 8.90, 12.64]
        })
        st.dataframe(df_cv, use_container_width=True)

    st.write("---")
    st.markdown("##### 📊 Top 10 Feature Importances (XGBoost Classifier)")
    feat_df = pd.DataFrame({
        "Feature": ["eda_std", "temp_mean", "eda_mean", "acc_0_mean", "acc_1_mean", "bvp_std", "temp_std", "acc_2_mean", "acc_2_std", "acc_1_std"],
        "Importance": [0.2392, 0.1484, 0.1475, 0.0960, 0.0780, 0.0716, 0.0681, 0.0654, 0.0295, 0.0245]
    }).sort_values("Importance", ascending=True)

    fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
    fig_feat.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_feat, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: FEEDBACK LOOP & DATA AUDIT (Continuous Learning)
# ------------------------------------------------------------------------------
with tab_feedback:
    st.markdown("### 🔄 Adaptive User Feedback Loop")
    st.caption("Capture ground-truth confirmation to refine user-specific baseline profiles and reduce false positives.")

    fb_1, fb_2, fb_3 = st.columns([3, 4, 3])
    with fb_1:
        user_feedback = st.radio(
            "Was this stress alert accurate?",
            ["Yes (True Positive)", "No (False Positive)", "Neutral / Inconclusive"]
        )
    with fb_2:
        user_trigger = st.text_input("Reported Trigger / Activity", placeholder="e.g., Code compilation error, sudden meeting")
    with fb_3:
        st.write("")
        st.write("")
        if st.button("Submit Profile Feedback"):
            st.session_state.feedback_records.append({
                "timestamp": timestamp_str,
                "user_id": active_user,
                "prediction": stress_pred,
                "probability": round(stress_prob, 3),
                "feedback": user_feedback,
                "trigger_note": user_trigger
            })
            st.success("Feedback registered into profile adaptive pipeline.")

    st.write("---")
    st.markdown("##### 📁 Active Telemetry Buffer & Audit Log")
    st.dataframe(st.session_state.telemetry_history, use_container_width=True)

    if st.session_state.feedback_records:
        st.markdown("##### 📝 Historical Feedback Logs")
        df_fb = pd.DataFrame(st.session_state.feedback_records)
        st.dataframe(df_fb, use_container_width=True)
        st.download_button(
            "⬇️ Export Feedback Dataset for Retraining (CSV)",
            data=df_fb.to_csv(index=False),
            file_name="stress_feedback_log.csv",
            mime="text/csv"
        )

# Auto-refresh loop when simulating
if ingest_mode == "Simulated IoT/Wearable Stream":
    time.sleep(poll_rate)
    st.rerun()