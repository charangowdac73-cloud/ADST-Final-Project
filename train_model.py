"""
train_model.py
----------------
Generates a labeled network-traffic dataset and trains a machine-learning
model that classifies each connection record as NORMAL traffic or one of
three attack categories: DOS, PROBE, or R2L.

The feature set is a simplified version of the well-known NSL-KDD /
KDD Cup 99 intrusion-detection schema, which is the standard reference
dataset used in cyber-threat-detection research and coursework.

Run this once before starting the Flask server:
    python train_model.py

It writes:
    dataset/network_traffic.csv   -> the generated training data
    model/threat_model.pkl        -> the trained scikit-learn pipeline
    model/stats.json              -> accuracy + confusion matrix for the dashboard
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ----------------------------------------------------------------------
# Fixed vocabularies. The frontend dropdowns must match these exactly,
# since the OneHotEncoder is fit on these categories.
# ----------------------------------------------------------------------
PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "dns", "telnet", "other"]
FLAGS = ["SF", "S0", "REJ", "RSTR"]

NUMERIC_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "serror_rate", "same_srv_rate", "diff_srv_rate", "dst_host_count",
]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
LABELS = ["normal", "dos", "probe", "r2l"]


def _clip(value, low, high):
    return float(np.clip(value, low, high))


def generate_record(label):
    """Generate one synthetic connection record for the given class label.

    The value ranges are chosen to mimic well-documented behavioral
    signatures of each traffic type (e.g. DoS floods show very high
    connection counts and SYN-error rates; probes touch many different
    services with short-lived connections; R2L attempts look like
    slow, deliberate logins to a single service).
    """
    if label == "normal":
        duration = rng.exponential(50)
        protocol = rng.choice(PROTOCOLS, p=[0.6, 0.3, 0.1])
        service = rng.choice(["http", "ftp", "smtp", "dns", "other"], p=[0.45, 0.1, 0.1, 0.25, 0.1])
        flag = "SF"
        src_bytes = abs(rng.normal(500, 200))
        dst_bytes = abs(rng.normal(1000, 400))
        count = rng.integers(1, 20)
        srv_count = rng.integers(1, 20)
        serror_rate = rng.uniform(0, 0.05)
        same_srv_rate = rng.uniform(0.8, 1.0)
        diff_srv_rate = rng.uniform(0, 0.1)
        dst_host_count = rng.integers(1, 50)

    elif label == "dos":
        duration = rng.uniform(0, 2)
        protocol = rng.choice(["tcp", "icmp"], p=[0.7, 0.3])
        service = rng.choice(["http", "other"], p=[0.6, 0.4])
        flag = rng.choice(["S0", "REJ"], p=[0.8, 0.2])
        src_bytes = rng.uniform(0, 50)
        dst_bytes = rng.uniform(0, 5)
        count = rng.integers(150, 511)
        srv_count = rng.integers(150, 511)
        serror_rate = rng.uniform(0.7, 1.0)
        same_srv_rate = rng.uniform(0.9, 1.0)
        diff_srv_rate = rng.uniform(0, 0.05)
        dst_host_count = rng.integers(150, 256)

    elif label == "probe":
        duration = rng.uniform(0, 5)
        protocol = rng.choice(["tcp", "udp"], p=[0.6, 0.4])
        service = rng.choice(SERVICES)
        flag = rng.choice(["S0", "REJ", "SF"], p=[0.5, 0.3, 0.2])
        src_bytes = rng.uniform(0, 20)
        dst_bytes = rng.uniform(0, 20)
        count = rng.integers(1, 10)
        srv_count = rng.integers(1, 10)
        serror_rate = rng.uniform(0.3, 0.7)
        same_srv_rate = rng.uniform(0, 0.3)
        diff_srv_rate = rng.uniform(0.6, 1.0)
        dst_host_count = rng.integers(1, 256)

    elif label == "r2l":
        duration = rng.uniform(5, 100)
        protocol = "tcp"
        service = rng.choice(["ftp", "telnet", "smtp"])
        flag = "SF"
        src_bytes = rng.uniform(200, 2000)
        dst_bytes = rng.uniform(200, 2000)
        count = rng.integers(1, 5)
        srv_count = rng.integers(1, 5)
        serror_rate = rng.uniform(0, 0.1)
        same_srv_rate = rng.uniform(0.5, 1.0)
        diff_srv_rate = rng.uniform(0, 0.2)
        dst_host_count = rng.integers(1, 10)

    else:
        raise ValueError(f"Unknown label: {label}")

    return {
        "duration": round(_clip(duration, 0, 10000), 3),
        "protocol_type": protocol,
        "service": service,
        "flag": flag,
        "src_bytes": round(_clip(src_bytes, 0, 1_000_000), 2),
        "dst_bytes": round(_clip(dst_bytes, 0, 1_000_000), 2),
        "count": int(count),
        "srv_count": int(srv_count),
        "serror_rate": round(_clip(serror_rate, 0, 1), 3),
        "same_srv_rate": round(_clip(same_srv_rate, 0, 1), 3),
        "diff_srv_rate": round(_clip(diff_srv_rate, 0, 1), 3),
        "dst_host_count": int(dst_host_count),
        "label": label,
    }


def build_dataset(n_per_class=1500):
    records = []
    for label in LABELS:
        for _ in range(n_per_class):
            records.append(generate_record(label))
    df = pd.DataFrame.from_records(records)
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(categories=[PROTOCOLS, SERVICES, FLAGS], handle_unknown="ignore"),
             CATEGORICAL_COLUMNS),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main():
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    print("Generating synthetic network-traffic dataset...")
    df = build_dataset(n_per_class=1500)
    df.to_csv("dataset/network_traffic.csv", index=False)
    print(f"  saved {len(df)} rows -> dataset/network_traffic.csv")

    X = df[FEATURE_COLUMNS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print("Training RandomForestClassifier...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels_sorted = LABELS
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist()
    report = classification_report(y_test, y_pred, labels=labels_sorted, output_dict=True)

    print(f"  test accuracy: {accuracy:.4f}")

    joblib.dump(pipeline, "model/threat_model.pkl")
    print("  saved -> model/threat_model.pkl")

    stats = {
        "accuracy": round(accuracy, 4),
        "labels": labels_sorted,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "protocols": PROTOCOLS,
        "services": SERVICES,
        "flags": FLAGS,
    }
    with open("model/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("  saved -> model/stats.json")
    print("Done.")


if __name__ == "__main__":
    main()
