# Cyber Threat Detection Using Machine Learning

A full-stack mini project that classifies network connection records as
**Normal** traffic or one of three attack types — **DoS**, **Probe**, or
**R2L** — using a Random Forest model trained with scikit-learn, served
through a Flask API, with a browser dashboard as the frontend.

## How it works

1. **`train_model.py`** generates a labeled dataset of synthetic network
   connection records (feature schema based on the well-known NSL-KDD /
   KDD Cup 99 intrusion-detection format: protocol, service, connection
   flag, byte counts, connection-rate statistics, etc.), then trains a
   `RandomForestClassifier` inside a scikit-learn `Pipeline` (one-hot
   encoding + scaling included) and saves it to `model/threat_model.pkl`,
   along with accuracy/confusion-matrix stats in `model/stats.json`.
2. **`app.py`** is a Flask server that loads the trained model and exposes
   a small JSON API, and also serves the dashboard itself — so frontend
   and backend run from a single process on `http://127.0.0.1:5000`.
3. **`templates/index.html` + `static/`** is the dashboard: a form for a
   connection record, a "Generate sample" button that asks the backend
   for a realistic Normal/DoS/Probe/R2L example, a result panel with the
   predicted class and per-class probabilities, a model-performance
   panel (accuracy + confusion matrix), and a running detection log.

## Project structure

```
cyber_threat_detection/
├── app.py                  # Flask backend (API + serves the frontend)
├── train_model.py          # Dataset generation + model training
├── requirements.txt
├── model/
│   ├── threat_model.pkl    # trained pipeline (pre-built, included)
│   └── stats.json          # accuracy + confusion matrix (pre-built)
├── dataset/
│   └── network_traffic.csv # the generated training data (pre-built)
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
└── README.md
```

## Setup and run

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) retrain the model — a trained model is already included,
#    so you only need this if you want to regenerate it
python train_model.py

# 4. Start the server
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | Body: JSON with all 12 feature fields. Returns `prediction`, `confidence`, and per-class `probabilities`. |
| `/api/sample?type=normal\|dos\|probe\|r2l\|random` | GET | Returns a synthetic example record of the requested class, for auto-filling the form. |
| `/api/stats` | GET | Returns model accuracy, confusion matrix, and the valid dropdown vocabularies. |
| `/api/health` | GET | Basic liveness check. |

Example request:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "S0",
        "src_bytes": 0, "dst_bytes": 0, "count": 300, "srv_count": 300,
        "serror_rate": 0.95, "same_srv_rate": 0.98, "diff_srv_rate": 0.01,
        "dst_host_count": 255
      }'
```

## Notes for extending this project

- The dataset is synthetically generated so the project runs fully offline
  with no external downloads. To use a real dataset instead (e.g. NSL-KDD
  or CICIDS2017), replace the CSV produced by `train_model.py` with the
  real one, keep the same column names, and retrain.
- The classifier is a Random Forest for speed and interpretability; the
  pipeline structure (`ColumnTransformer` + estimator) makes it easy to
  swap in another scikit-learn estimator (e.g. `GradientBoostingClassifier`
  or an `SVC`) without changing `app.py`.
- All API responses are JSON, so the existing HTML/CSS/JS frontend could
  be swapped for a different framework (React, Vue) without touching the
  backend.
