"""
generate_dataset.py
-------------------
Generates a synthetic network performance dataset with realistic failure patterns.
Run this script once to create data/network_data.csv before training the model.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ─── Configuration ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
NUM_ROWS    = 10_000
START_DATE  = datetime(2024, 1, 1)

np.random.seed(RANDOM_SEED)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _simulate_normal_period(n):
    """Return a dict of feature arrays for NORMAL network conditions."""
    return {
        "latency":         np.random.normal(30,  8,  n).clip(5, 80),
        "packet_loss":     np.random.normal(0.5, 0.3, n).clip(0, 3),
        "bandwidth_usage": np.random.normal(50,  15, n).clip(10, 90),
        "jitter":          np.random.normal(5,   2,  n).clip(0, 20),
        "error_rate":      np.random.normal(0.1, 0.05, n).clip(0, 0.5),
        "cpu_usage":       np.random.normal(40,  12, n).clip(5, 85),
        "memory_usage":    np.random.normal(50,  15, n).clip(10, 90),
    }


def _simulate_failure_period(n):
    """Return a dict of feature arrays for STRESSED / near-failure conditions."""
    return {
        "latency":         np.random.normal(180, 50,  n).clip(80, 500),
        "packet_loss":     np.random.normal(8,   3,   n).clip(2, 25),
        "bandwidth_usage": np.random.normal(88,  8,   n).clip(60, 100),
        "jitter":          np.random.normal(40,  15,  n).clip(10, 100),
        "error_rate":      np.random.normal(0.6, 0.2, n).clip(0.3, 1.0),
        "cpu_usage":       np.random.normal(85,  8,   n).clip(60, 100),
        "memory_usage":    np.random.normal(88,  6,   n).clip(65, 100),
    }


def _label_from_features(df):
    """
    Derive network_status (0 = normal, 1 = failure) from the generated
    feature values using a realistic threshold-based rule.
    """
    score = (
        (df["latency"]         > 120) * 2 +
        (df["packet_loss"]     > 5)   * 2 +
        (df["bandwidth_usage"] > 85)  * 1 +
        (df["jitter"]          > 25)  * 1 +
        (df["error_rate"]      > 0.4) * 2 +
        (df["cpu_usage"]       > 80)  * 1 +
        (df["memory_usage"]    > 85)  * 1
    )
    # Failure if score >= 4 (multiple indicators firing simultaneously)
    return (score >= 4).astype(int)


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_dataset(output_path: str = "data/network_data.csv") -> pd.DataFrame:
    """Generate the synthetic dataset and save to CSV."""

    # Split rows into alternating normal / stressed windows
    rows_per_window = 200
    records = []
    current_ts = START_DATE

    for block_start in range(0, NUM_ROWS, rows_per_window):
        n = min(rows_per_window, NUM_ROWS - block_start)
        stressed = (block_start // rows_per_window) % 3 == 2  # every 3rd window is stressed

        feats = _simulate_failure_period(n) if stressed else _simulate_normal_period(n)

        # Add small Gaussian noise to every metric for realism
        for key in feats:
            noise = np.random.normal(0, feats[key].std() * 0.05, n)
            feats[key] = (feats[key] + noise).round(3)

        # Build timestamp array (1-minute intervals)
        timestamps = [current_ts + timedelta(minutes=i) for i in range(n)]
        current_ts += timedelta(minutes=n)

        block_df = pd.DataFrame(feats)
        block_df.insert(0, "timestamp", timestamps)
        records.append(block_df)

    df = pd.concat(records, ignore_index=True)

    # Label based on feature thresholds (realistic, not random)
    df["network_status"] = _label_from_features(df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    failure_rate = df["network_status"].mean() * 100
    print(f"✅  Dataset saved → {output_path}")
    print(f"    Rows            : {len(df):,}")
    print(f"    Failure rate    : {failure_rate:.1f}%")
    print(f"    Columns         : {list(df.columns)}")
    return df


if __name__ == "__main__":
    generate_dataset()
