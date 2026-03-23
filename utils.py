"""
utils.py
--------
Shared helper functions used by both train.py and app.py.
Handles data loading, preprocessing, and all visualisation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

# ─── Constants ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "latency", "packet_loss", "bandwidth_usage",
    "jitter", "error_rate", "cpu_usage", "memory_usage",
]
TARGET_COL   = "network_status"

# Thresholds used by the Streamlit app to highlight abnormal values
THRESHOLDS = {
    "latency":         120,
    "packet_loss":     5,
    "bandwidth_usage": 85,
    "jitter":          25,
    "error_rate":      0.4,
    "cpu_usage":       80,
    "memory_usage":    85,
}

# Colour palette (consistent dark theme)
PALETTE = {
    "normal":  "#00C48C",
    "failure": "#FF4D6D",
    "accent":  "#6C63FF",
    "bg":      "#1A1A2E",
    "text":    "#E0E0E0",
}

# ─── Data Helpers ─────────────────────────────────────────────────────────────

def load_data(csv_path: str = "data/network_data.csv") -> pd.DataFrame:
    """Load the network metrics CSV and parse timestamps."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Run  python data/generate_dataset.py  first."
        )
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    assert all(c in df.columns for c in FEATURE_COLS + [TARGET_COL]), \
        "CSV is missing required columns!"
    return df


def preprocess_data(df: pd.DataFrame, scaler: StandardScaler = None):
    """
    Scale features and split into train / test sets.

    Returns:
        X_train, X_test, y_train, y_test, fitted_scaler
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    else:
        X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def scale_single(values: dict, scaler: StandardScaler) -> np.ndarray:
    """Scale a single observation dict into a 2-D array ready for prediction."""
    row = np.array([[values[c] for c in FEATURE_COLS]])
    return scaler.transform(row)


def compute_metrics(y_true, y_pred) -> dict:
    """Return a dict with all key evaluation metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "report":    classification_report(
                         y_true, y_pred,
                         target_names=["Normal", "Failure"],
                         zero_division=0
                     ),
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _apply_dark_style(ax, fig):
    """Apply a unified dark background style to an axis."""
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.tick_params(colors=PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444455")


def plot_metrics_over_time(df: pd.DataFrame, save_path: str = "models/metrics_over_time.png"):
    """
    Plot six network metrics over time in a 3×2 grid.
    Highlights failure windows in red.
    """
    metrics = ["latency", "packet_loss", "bandwidth_usage", "jitter", "error_rate", "cpu_usage"]
    colors  = ["#6C63FF", "#FF4D6D", "#00C48C", "#FFB347", "#FF69B4", "#4FC3F7"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.patch.set_facecolor(PALETTE["bg"])

    # Sample 2000 rows for a readable plot
    sample = df.sample(n=min(2000, len(df)), random_state=42).sort_values("timestamp")

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i]
        ax.set_facecolor(PALETTE["bg"])

        # Shade failure windows
        failure_mask = sample["network_status"] == 1
        ax.fill_between(
            sample["timestamp"], 0, sample[metric].max() * 1.1,
            where=failure_mask, alpha=0.15, color=PALETTE["failure"], label="Failure"
        )
        ax.plot(sample["timestamp"], sample[metric], color=color, linewidth=0.8, alpha=0.9)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylabel(metric, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        _apply_dark_style(ax, fig)

    fig.suptitle("Network Metrics Over Time", fontsize=16, fontweight="bold",
                 color=PALETTE["text"], y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  📊  Saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path: str = "models/confusion_matrix.png"):
    """Save a styled seaborn confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn",
        xticklabels=["Normal", "Failure"],
        yticklabels=["Normal", "Failure"],
        linewidths=0.5, linecolor="#333344",
        ax=ax, annot_kws={"size": 14, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", color=PALETTE["text"], fontsize=11)
    ax.set_ylabel("True Label",      color=PALETTE["text"], fontsize=11)
    ax.set_title("Confusion Matrix", color=PALETTE["text"], fontsize=14, fontweight="bold")
    ax.tick_params(colors=PALETTE["text"])
    plt.setp(ax.get_xticklabels(), color=PALETTE["text"])
    plt.setp(ax.get_yticklabels(), color=PALETTE["text"])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  📊  Saved → {save_path}")


def plot_feature_importance(importances: np.ndarray, save_path: str = "models/feature_importance.png"):
    """Horizontal bar chart of Random Forest feature importances."""
    fi = pd.Series(importances, index=FEATURE_COLS).sort_values()

    gradient_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))  # type: ignore

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.barh(fi.index, fi.values, color=gradient_colors, edgecolor="none", height=0.6)

    # Annotate bars with percentage
    for bar, val in zip(bars, fi.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", color=PALETTE["text"], fontsize=9)

    ax.set_xlabel("Importance Score", color=PALETTE["text"], fontsize=11)
    ax.set_title("Feature Importance (Random Forest)",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")
    _apply_dark_style(ax, fig)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  📊  Saved → {save_path}")
