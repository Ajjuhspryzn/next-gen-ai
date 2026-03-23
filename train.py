"""
train.py
--------
Trains a Random Forest classifier on the network metrics dataset,
evaluates it comprehensively, saves the model + scaler, and produces
all visualisation plots required by the Streamlit dashboard.

Usage:
    python data/generate_dataset.py   # once, to create the CSV
    python train.py
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Allow imports from project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_data,
    preprocess_data,
    compute_metrics,
    plot_metrics_over_time,
    plot_confusion_matrix,
    plot_feature_importance,
    FEATURE_COLS,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/network_data.csv"
MODEL_PATH   = "models/rf_model.pkl"
SCALER_PATH  = "models/scaler.pkl"

# ─── Main ─────────────────────────────────────────────────────────────────────

def train():
    print("=" * 55)
    print("  Network Failure Prediction — Model Training")
    print("=" * 55)

    # 1. Load data ─────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
    df = load_data(DATA_PATH)
    print(f"      Rows: {len(df):,}  |  Failures: {df['network_status'].sum():,}"
          f"  ({df['network_status'].mean()*100:.1f}%)")

    # 2. Preprocess ────────────────────────────────────────────
    print("\n[2/5] Preprocessing (scaling + train/test split) …")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 3. Train ─────────────────────────────────────────────────
    print("\n[3/5] Training Random Forest (100 trees) …")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,          # grow full trees
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,               # use all CPU cores
        class_weight="balanced", # handle any class imbalance
    )
    model.fit(X_train, y_train)
    print("      Training complete ✅")

    # 4. Evaluate ──────────────────────────────────────────────
    print("\n[4/5] Evaluating on test set …")
    y_pred   = model.predict(X_test)
    metrics  = compute_metrics(y_test, y_pred)

    print(f"\n  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%")
    print(f"  F1 Score  : {metrics['f1']*100:.2f}%")
    print("\n  Full Classification Report:")
    print("  " + metrics["report"].replace("\n", "\n  "))

    # 5. Save artefacts ────────────────────────────────────────
    print("\n[5/5] Saving model, scaler and plots …")
    os.makedirs("models", exist_ok=True)

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  💾  Saved model  → {MODEL_PATH}")
    print(f"  💾  Saved scaler → {SCALER_PATH}")

    # Plots
    plot_metrics_over_time(df)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model.feature_importances_)

    print("\n" + "=" * 55)
    print("  All done! Run:  streamlit run app.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    train()
