"""
app.py
------
Flask backend for the Cyber Threat Detection mini project.

Serves:
  GET  /                      -> the frontend dashboard (templates/index.html)
  POST /api/predict           -> classify a single connection record
  GET  /api/sample?type=...   -> generate a synthetic record to auto-fill the form
  GET  /api/stats             -> model accuracy + confusion matrix for the dashboard
  GET  /api/health            -> simple liveness check

Run:
    python train_model.py   # once, to create model/threat_model.pkl
    python app.py
    then open http://127.0.0.1:5000 in a browser
"""

import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from train_model import (
    FEATURE_COLUMNS,
    FLAGS,
    LABELS,
    NUMERIC_COLUMNS,
    PROTOCOLS,
    SERVICES,
    generate_record,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "threat_model.pkl")
STATS_PATH = os.path.join(BASE_DIR, "model", "stats.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

_model = None
_stats = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model file not found. Run 'python train_model.py' first to train and save it."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_stats():
    global _stats
    if _stats is None:
        with open(STATS_PATH) as f:
            _stats = json.load(f)
    return _stats


def validate_payload(data):
    """Check the incoming JSON has every required field with an acceptable value."""
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        return f"Missing field(s): {', '.join(missing)}"

    for col in NUMERIC_COLUMNS:
        try:
            float(data[col])
        except (TypeError, ValueError):
            return f"Field '{col}' must be a number."

    if data["protocol_type"] not in PROTOCOLS:
        return f"protocol_type must be one of {PROTOCOLS}"
    if data["service"] not in SERVICES:
        return f"service must be one of {SERVICES}"
    if data["flag"] not in FLAGS:
        return f"flag must be one of {FLAGS}"

    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    error = validate_payload(data)
    if error:
        return jsonify({"error": error}), 400

    model = get_model()

    row = {col: data[col] for col in FEATURE_COLUMNS}
    for col in NUMERIC_COLUMNS:
        row[col] = float(row[col])

    import pandas as pd
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    prob_map = {cls: round(float(p), 4) for cls, p in zip(classes, probabilities)}
    confidence = prob_map[prediction]

    return jsonify({
        "prediction": prediction,
        "is_attack": prediction != "normal",
        "confidence": confidence,
        "probabilities": prob_map,
    })


@app.route("/api/sample")
def sample():
    kind = request.args.get("type", "random")
    if kind == "random":
        kind = np.random.choice(LABELS)
    if kind not in LABELS:
        return jsonify({"error": f"type must be one of {LABELS + ['random']}"}), 400

    record = generate_record(kind)
    record.pop("label", None)
    return jsonify(record)


@app.route("/api/stats")
def stats():
    return jsonify(get_stats())


if __name__ == "__main__":
    # Load the model once at startup so the first request isn't slow,
    # and so a missing model file fails fast with a clear message.
    get_model()
    get_stats()
    app.run(debug=True, host="0.0.0.0", port=5000)
