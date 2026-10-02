"""
app.py
------
Flask backend. Serves the web page and exposes:
  POST /predict  -> returns predicted crop + explanation
  GET  /metrics  -> returns real model evaluation metrics
Run:  python app.py   then open http://127.0.0.1:5000
"""
import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from data_loader import FEATURES
from explainer import explain

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model", "crop_model.pkl")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")

# Train automatically on first run if the model file is missing
if not os.path.exists(MODEL_PATH):
    import train_model
    train_model.main()

model = joblib.load(MODEL_PATH)
with open(METRICS_PATH) as f:
    METRICS = json.load(f)

# Allowed input ranges (simple validation rules)
RANGES = {
    "N": (0, 200), "P": (0, 200), "K": (0, 250),
    "temperature": (0, 55), "humidity": (0, 100),
    "ph": (0, 14), "rainfall": (0, 500),
}

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    keys = ["accuracy", "train_size", "test_size", "total_records", "num_classes",
            "labels", "confusion_matrix", "macro_avg", "weighted_avg",
            "per_class", "feature_importance"]
    return jsonify({k: METRICS[k] for k in keys})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    inputs, errors = {}, []

    # Validate every field: present, numeric, and inside a sensible range
    for field in FEATURES:
        value = data.get(field)
        if value is None or str(value).strip() == "":
            errors.append(f"{field} is required.")
            continue
        try:
            value = float(value)
        except ValueError:
            errors.append(f"{field} must be a number.")
            continue
        lo, hi = RANGES[field]
        if not lo <= value <= hi:
            errors.append(f"{field} must be between {lo} and {hi}.")
            continue
        inputs[field] = value

    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    # Build a one-row DataFrame in the same column order used for training
    X = pd.DataFrame([inputs], columns=FEATURES)
    crop = model.predict(X)[0]
    confidence = float(model.predict_proba(X).max())

    return jsonify({
        "crop": crop,
        "confidence": round(confidence, 4),
        "explanation": explain(crop, inputs, METRICS["crop_means"]),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
