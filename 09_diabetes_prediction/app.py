"""
Flask Web Application for Diabetes Progression Prediction.
Uses a trained Decision Tree Regressor model with StandardScaler.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Verify artifacts exist
for path, name in [
    (MODEL_PATH, "model.pkl"),
    (SCALER_PATH, "scaler.pkl"),
    (FEATURES_PATH, "features.pkl"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required artifact: {name} in {MODELS_DIR}. "
            "Run 'python train.py' first to generate pickle files."
        )

# Load model artifacts
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    feature_names = pickle.load(f)

# Load metadata (optional, for UI hints)
metadata = {}
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

feature_stats = metadata.get("feature_stats", {})


def predict_diabetes(raw_values: dict) -> float:
    """
    Accept raw clinical feature values, scale them, and predict diabetes progression.

    Args:
        raw_values: dict mapping feature name -> raw clinical value

    Returns:
        Predicted diabetes progression score (float).
    """
    # Build input array in correct feature order
    input_array = np.array([[float(raw_values[f]) for f in feature_names]])

    # Scale raw clinical values to the model's expected scaled space
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(scaled_input)[0]
    return round(float(prediction), 2)


@app.route("/", methods=["GET"])
def index():
    """Render the prediction form."""
    return render_template(
        "index.html",
        feature_names=feature_names,
        feature_stats=feature_stats,
        metadata=metadata,
        prediction=None,
        input_data=None,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and return prediction."""
    try:
        input_data = {}
        for f in feature_names:
            value = request.form.get(f, "").strip()
            if not value:
                raise ValueError(
                    f"Please provide a value for '{feature_stats.get(f, {}).get('description', f)}'."
                )
            try:
                input_data[f] = float(value)
            except ValueError:
                raise ValueError(
                    f"Invalid number for '{feature_stats.get(f, {}).get('description', f)}': '{value}'"
                )

        prediction = predict_diabetes(input_data)

        return render_template(
            "index.html",
            feature_names=feature_names,
            feature_stats=feature_stats,
            metadata=metadata,
            prediction=prediction,
            input_data=input_data,
            error=None,
        )
    except Exception as e:
        return render_template(
            "index.html",
            feature_names=feature_names,
            feature_stats=feature_stats,
            metadata=metadata,
            prediction=None,
            input_data=request.form.to_dict() if request.form else None,
            error=str(e),
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic predictions."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON body"}), 400

        for f in feature_names:
            if f not in data:
                return jsonify({"status": "error", "message": f"Missing field '{f}'"}), 400

        prediction = predict_diabetes(data)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "description": "Predicted diabetes disease progression (quantitative measure)"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": "Decision Tree Regressor",
        "features": feature_names,
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
