# 🌐 Network Failure Prediction System

An end-to-end machine learning system that predicts network failures in real time using historical performance metrics. Built with Python, Scikit-learn, and Streamlit.

---

## 📸 Dashboard Preview

The dashboard features:
- **🔮 Prediction tab** — Input sliders with instant failure probability
- **⚡ Live Simulation** — Auto-generates metrics and shows predictions step-by-step
- **📊 Analytics** — Confusion matrix, feature importance, and metrics-over-time charts
- **ℹ️ About** — Project overview and quick-start guide

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### 1. Clone / Download the project

```bash
cd NextGen_pro
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python data/generate_dataset.py
```

This creates `data/network_data.csv` (10 000 rows of synthetic network metrics with realistic failure patterns).

### 4. Train the model

```bash
python train.py
```

This will:
- Train a **Random Forest** classifier on the dataset
- Print accuracy, precision, recall, and F1-score to the console
- Save `models/rf_model.pkl` and `models/scaler.pkl`
- Generate three PNG charts in the `models/` folder

### 5. Launch the dashboard

```bash
streamlit run app.py
```

The browser will open at `http://localhost:8501` automatically.

---

## 📂 Project Structure

```
NextGen_pro/
├── data/
│   ├── generate_dataset.py   ← Synthetic dataset generator
│   └── network_data.csv      ← Generated CSV (created in step 3)
├── models/
│   ├── rf_model.pkl          ← Trained Random Forest model
│   ├── scaler.pkl            ← Feature scaler
│   ├── confusion_matrix.png  ← Model evaluation chart
│   ├── feature_importance.png
│   └── metrics_over_time.png
├── utils.py                  ← Shared data + plotting helpers
├── train.py                  ← Training pipeline
├── app.py                    ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Features

| Column | Description | Typical Range |
|--------|-------------|---------------|
| `timestamp` | Measurement timestamp | 2024-01 to 2024-07 |
| `latency` | Network round-trip time (ms) | 5 – 500 |
| `packet_loss` | % of lost packets | 0 – 25 |
| `bandwidth_usage` | % of available bandwidth | 10 – 100 |
| `jitter` | Variance in latency (ms) | 0 – 100 |
| `error_rate` | Fraction of errored packets | 0 – 1 |
| `cpu_usage` | Device CPU usage (%) | 5 – 100 |
| `memory_usage` | Device memory usage (%) | 10 – 100 |
| `network_status` | **Target**: 0 = Normal, 1 = Failure | 0 or 1 |

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Estimators | 100 trees |
| Train / Test split | 80% / 20% (stratified) |
| Feature scaling | StandardScaler |
| Class weighting | `balanced` (handles imbalance) |
| Expected F1-score | ≥ 0.90 |

### Failure threshold rules used during labelling
A sample is marked `failure` when multiple indicators exceed safe thresholds simultaneously:
- Latency > 120 ms  
- Packet Loss > 5 %  
- Error Rate > 0.4  
- CPU > 80 % or Memory > 85 %

---

## 📈 Visualisations

Three charts are generated automatically by `train.py`:

1. **Network Metrics Over Time** — Line plots for all metrics, failure windows shaded in red
2. **Confusion Matrix** — Green/red heatmap showing true/false positives and negatives
3. **Feature Importance** — Ranked horizontal bar chart of the most predictive features

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data generation | NumPy, Pandas |
| Machine learning | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model persistence | Joblib |

---

## 💡 Tips

- **Stress test**: In the **Live Simulation** tab, enable "Inject Stress" to see the model predict failures under extreme conditions.
- **Feature investigation**: Drag the **Latency** slider above 120 ms and the **Error Rate** above 0.4 at the same time — the model should immediately classify as "Failure Likely".
- **Re-train**: You can re-run `python train.py` at any time. The Streamlit app always loads the latest saved model.

---

*Built with Python · Scikit-learn · Streamlit · Matplotlib*
