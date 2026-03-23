"""
app.py
------
Streamlit dashboard for the Network Failure Prediction System.

Run with:
    streamlit run app.py
"""

import os
import sys
import time
import random
import numpy as np
import streamlit as st
import joblib
import pandas as pd

# Allow project-root imports when launched from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import FEATURE_COLS, THRESHOLDS, scale_single, load_data

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Failure Prediction",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base theme ── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: rgba(16, 16, 32, 0.95) !important;
    border-right: 1px solid #2a2a4a;
}
/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(108,99,255,.25);
}
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.75rem; color: #8888aa; text-transform: uppercase; letter-spacing: .05em; }
/* ── Prediction banner ── */
.pred-normal {
    background: linear-gradient(135deg,#00c48c22,#00c48c11);
    border: 2px solid #00c48c;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.pred-failure {
    background: linear-gradient(135deg,#ff4d6d22,#ff4d6d11);
    border: 2px solid #ff4d6d;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.pred-title { font-size: 2rem; font-weight: 800; margin: 0; }
.pred-sub   { font-size: 0.9rem; color: #aaaacc; margin-top: 0.4rem; }
/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: .04em;
    color: #c8c8ff;
    border-bottom: 1px solid #2a2a4a;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
/* ── Abnormal badge ── */
.badge-warn {
    background: #ff4d6d22;
    color: #ff4d6d;
    border: 1px solid #ff4d6d55;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-ok {
    background: #00c48c22;
    color: #00c48c;
    border: 1px solid #00c48c55;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
}
/* ── Streamlit overrides ── */
div[data-testid="stProgress"] > div { background: #6c63ff !important; }
.stButton>button {
    background: linear-gradient(135deg,#6c63ff,#9c55ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: .6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: opacity .2s;
}
.stButton>button:hover { opacity: .85; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load trained RF model and scaler from disk (cached across reruns)."""
    model_path  = "models/rf_model.pkl"
    scaler_path = "models/scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)


model, scaler = load_model()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 Network Failure\n### Prediction System")
    st.markdown("---")

    st.markdown("### 🎛️ Input Parameters")
    st.caption("Drag sliders to enter current network metrics.")

    latency         = st.slider("Latency (ms)",         5,    500,  35,   step=1)
    packet_loss     = st.slider("Packet Loss (%)",      0.0,  25.0, 0.5,  step=0.1)
    bandwidth_usage = st.slider("Bandwidth Usage (%)",  10,   100,  50,   step=1)
    jitter          = st.slider("Jitter (ms)",          0,    100,  5,    step=1)
    error_rate      = st.slider("Error Rate",           0.0,  1.0,  0.1,  step=0.01)
    cpu_usage       = st.slider("CPU Usage (%)",        5,    100,  40,   step=1)
    memory_usage    = st.slider("Memory Usage (%)",     10,   100,  50,   step=1)

    st.markdown("---")

    predict_btn = st.button("🔮 Predict Now", use_container_width=True)

    st.markdown("---")
    st.caption("Model: **Random Forest** · 100 estimators\n\n"
               "Dataset: 10 000 synthetic samples with\nrealistic failure patterns.")


# ─── Build input dict ─────────────────────────────────────────────────────────
user_inputs = {
    "latency":         latency,
    "packet_loss":     packet_loss,
    "bandwidth_usage": bandwidth_usage,
    "jitter":          jitter,
    "error_rate":      error_rate,
    "cpu_usage":       cpu_usage,
    "memory_usage":    memory_usage,
}


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
  <h1 style='font-size:2.4rem; font-weight:800; background:linear-gradient(90deg,#6c63ff,#00c48c);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>
    🌐 Network Failure Prediction System
  </h1>
  <p style='color:#8888aa; margin-top:0.4rem; font-size:0.95rem;'>
    AI-powered real-time monitoring &amp; anomaly detection
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_predict, tab_simulation, tab_charts, tab_about = st.tabs(
    ["🔮 Prediction", "⚡ Live Simulation", "📊 Analytics", "ℹ️ About"]
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    if model is None:
        st.error("⚠️ Model not found. Please run **`python train.py`** first to train the model.")
    else:
        # ── Auto-predict on slider change (show last result) ──────────────
        col_pred, col_gauge = st.columns([1, 1], gap="large")

        with col_pred:
            st.markdown('<p class="section-header">🎯 Prediction Result</p>', unsafe_allow_html=True)

            X_scaled  = scale_single(user_inputs, scaler)
            pred      = model.predict(X_scaled)[0]
            proba     = model.predict_proba(X_scaled)[0]
            prob_fail = proba[1]
            prob_norm = proba[0]

            if pred == 1:
                st.markdown(f"""
                <div class='pred-failure'>
                  <p class='pred-title'>🔴 Failure Likely</p>
                  <p class='pred-sub'>Confidence: {prob_fail*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-normal'>
                  <p class='pred-title'>🟢 Normal</p>
                  <p class='pred-sub'>Confidence: {prob_norm*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Failure Probability**")
            st.progress(float(prob_fail))
            st.caption(f"{prob_fail*100:.1f}% — {'⚠️ High Risk!' if prob_fail > 0.5 else '✅ Low Risk'}")

        with col_gauge:
            st.markdown('<p class="section-header">📋 Parameter Status</p>', unsafe_allow_html=True)

            # Build a clean native Streamlit table using columns per row
            hdr1, hdr2, hdr3, hdr4 = st.columns([2, 1.2, 1.2, 1.5])
            hdr1.caption("Metric")
            hdr2.caption("Value")
            hdr3.caption("Threshold")
            hdr4.caption("Status")
            st.divider()

            for feat, val in user_inputs.items():
                threshold = THRESHOLDS[feat]
                abnormal  = val > threshold
                c1, c2, c3, c4 = st.columns([2, 1.2, 1.2, 1.5])

                c1.markdown(f"**{feat.replace('_', ' ').title()}**")
                # Value — red if abnormal, green if normal
                val_color = "#ff4d6d" if abnormal else "#00c48c"
                c2.markdown(
                    f"<span style='font-weight:700; color:{val_color};'>{val}</span>",
                    unsafe_allow_html=True,
                )
                c3.markdown(f"<span style='color:#666688;'>&gt; {threshold}</span>",
                            unsafe_allow_html=True)
                if abnormal:
                    c4.markdown("🔴 **ABNORMAL**")
                else:
                    c4.markdown("🟢 Normal")

        st.markdown("---")

        # ── Summary metrics row ────────────────────────────────────────────
        st.markdown('<p class="section-header">📈 Quick Stats</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)

        abnormal_count = sum(user_inputs[f] > THRESHOLDS[f] for f in FEATURE_COLS)
        with c1:
            st.markdown(f"""<div class='metric-card'><div class='metric-value' style='color:#6c63ff;'>
            {abnormal_count}</div><div class='metric-label'>Anomalous Metrics</div></div>""",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'><div class='metric-value' style='color:#ff4d6d;'>
            {prob_fail*100:.0f}%</div><div class='metric-label'>Failure Probability</div></div>""",
                        unsafe_allow_html=True)
        with c3:
            risk_level = "Critical" if prob_fail > 0.75 else "High" if prob_fail > 0.5 else "Medium" if prob_fail > 0.25 else "Low"
            risk_color = "#ff4d6d" if prob_fail > 0.5 else "#ffb347" if prob_fail > 0.25 else "#00c48c"
            st.markdown(f"""<div class='metric-card'><div class='metric-value' style='color:{risk_color};'>
            {risk_level}</div><div class='metric-label'>Risk Level</div></div>""",
                        unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='metric-card'><div class='metric-value' style='color:#4fc3f7;'>
            RF</div><div class='metric-label'>Algorithm</div></div>""",
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_simulation:
    st.markdown('<p class="section-header">⚡ Real-Time Network Simulation</p>', unsafe_allow_html=True)
    st.caption("Simulates live network metrics arriving every second. Uses the trained model to predict failures on the fly.")

    if model is None:
        st.warning("⚠️ Run `python train.py` first to enable simulation.")
    else:
        col_ctrl, col_info = st.columns([1, 2])
        with col_ctrl:
            run_sim = st.button("▶ Start Simulation", use_container_width=True, key="sim_btn")
            stress  = st.checkbox("🔥 Inject Stress (force failure)", value=False)
            n_steps = st.slider("Simulation steps", 5, 30, 10)

        if run_sim:
            metric_log: list[dict] = []
            pred_log:   list[int]  = []

            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            chart_placeholder  = st.empty()

            for step in range(n_steps):
                # Generate random metrics (stressed = failure-zone)
                if stress:
                    sim = {
                        "latency":         random.uniform(150, 400),
                        "packet_loss":     random.uniform(5,   20),
                        "bandwidth_usage": random.uniform(80,  100),
                        "jitter":          random.uniform(25,  80),
                        "error_rate":      random.uniform(0.4, 0.9),
                        "cpu_usage":       random.uniform(75,  100),
                        "memory_usage":    random.uniform(80,  100),
                    }
                else:
                    sim = {
                        "latency":         random.gauss(40,  30),
                        "packet_loss":     random.gauss(1,   2),
                        "bandwidth_usage": random.gauss(55,  20),
                        "jitter":          random.gauss(8,   10),
                        "error_rate":      random.gauss(0.15, 0.15),
                        "cpu_usage":       random.gauss(45,  20),
                        "memory_usage":    random.gauss(55,  20),
                    }
                    # Clip to valid ranges
                    sim = {k: max(0.0, v) for k, v in sim.items()}

                X_s    = scale_single(sim, scaler)
                p      = model.predict(X_s)[0]
                pf     = model.predict_proba(X_s)[0][1]

                metric_log.append({**sim, "fail_prob": pf})
                pred_log.append(p)

                # Update progress and status
                progress_bar.progress((step + 1) / n_steps)
                icon = "🔴 FAILURE" if p == 1 else "🟢 NORMAL"
                status_placeholder.markdown(
                    f"**Step {step+1}/{n_steps}** — Predicted: **{icon}** | "
                    f"Failure probability: **{pf*100:.1f}%** | "
                    f"Latency: `{sim['latency']:.1f} ms` · "
                    f"Packet Loss: `{sim['packet_loss']:.2f}%`"
                )

                # Live chart (failure probability over time)
                log_df = pd.DataFrame(metric_log)
                chart_placeholder.line_chart(log_df["fail_prob"], height=200)

                time.sleep(0.6)

            failures = sum(pred_log)
            st.success(
                f"✅ Simulation complete. "
                f"**{failures}/{n_steps}** steps predicted as network failures "
                f"({failures/n_steps*100:.0f}%)."
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ANALYTICS / CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.markdown('<p class="section-header">📊 Model Analytics & Visualisations</p>', unsafe_allow_html=True)

    img_paths = {
        "Network Metrics Over Time":   "models/metrics_over_time.png",
        "Confusion Matrix":            "models/confusion_matrix.png",
        "Feature Importance":          "models/feature_importance.png",
    }

    missing = [name for name, path in img_paths.items() if not os.path.exists(path)]
    if missing:
        st.warning(f"⚠️ The following charts are missing (run `python train.py`): {', '.join(missing)}")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📈 Network Metrics Over Time")
            st.image("models/metrics_over_time.png", use_container_width=True)
        with c2:
            st.markdown("##### 🔲 Confusion Matrix")
            st.image("models/confusion_matrix.png", use_container_width=True)

        st.markdown("##### 🎯 Feature Importance")
        st.image("models/feature_importance.png", use_container_width=True)

    # Dataset stats (if CSV exists)
    csv_path = "data/network_data.csv"
    if os.path.exists(csv_path):
        st.markdown("---")
        st.markdown('<p class="section-header">🗃️ Dataset Overview</p>', unsafe_allow_html=True)
        df = load_data(csv_path)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Rows",   f"{len(df):,}")
        col_b.metric("Features",     len(FEATURE_COLS))
        col_c.metric("Failures",     f"{df['network_status'].sum():,}")
        col_d.metric("Failure Rate", f"{df['network_status'].mean()*100:.1f}%")

        with st.expander("📋 Sample Data (first 50 rows)"):
            st.dataframe(df.head(50), use_container_width=True)

        with st.expander("📐 Statistical Summary"):
            st.dataframe(df[FEATURE_COLS].describe().round(3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## 🌐 Network Failure Prediction System

A complete, end-to-end machine learning system for predicting network failures
based on real-time and historical performance data.

---

### 🏗️ Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Dataset** | NumPy + Pandas | 10 000-row synthetic CSV |
| **Model** | Scikit-learn Random Forest | Binary classification (Normal / Failure) |
| **Visualisation** | Matplotlib + Seaborn | Charts saved during training |
| **Dashboard** | Streamlit | Interactive UI with live simulation |

---

### 📂 Project Structure

```
NextGen_pro/
├── data/
│   ├── generate_dataset.py   ← Synthetic data generator
│   └── network_data.csv      ← Generated dataset
├── models/
│   ├── rf_model.pkl          ← Trained Random Forest
│   ├── scaler.pkl            ← StandardScaler
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── metrics_over_time.png
├── utils.py                  ← Shared helpers
├── train.py                  ← Model training pipeline
├── app.py                    ← This Streamlit app
├── requirements.txt
└── README.md
```

---

### 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python data/generate_dataset.py

# 3. Train model
python train.py

# 4. Run dashboard
streamlit run app.py
```

---

### 📊 Features

- **Real-time prediction** — Instant classification as you drag sliders  
- **Live simulation** — Automated metric generation with failure injection  
- **Abnormal value highlighting** — Red indicators for out-of-range metrics  
- **Visual analytics** — Three publication-quality charts embedded in the dashboard  
- **Risk levels** — Low / Medium / High / Critical risk assessment  

---
*Built with ❤️ using Python, Scikit-learn & Streamlit*
    """)
