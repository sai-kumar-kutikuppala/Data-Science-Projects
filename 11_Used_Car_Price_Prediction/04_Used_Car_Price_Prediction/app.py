import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Expected model artifacts
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
OPTIONS_PATH = os.path.join(MODELS_DIR, "form_options.pkl")
DEFAULTS_PATH = os.path.join(MODELS_DIR, "model_defaults.json")

# Verify and load models
for path, name in [
    (LE_PATH, "label_encoder.pkl"),
    (PREPROCESSOR_PATH, "preprocessor.pkl"),
    (MODEL_PATH, "xgboost_model.pkl"),
    (OPTIONS_PATH, "form_options.pkl")
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required model artifact: {name} in {MODELS_DIR}. Run the training step first.")

with open(LE_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(OPTIONS_PATH, "rb") as f:
    form_options = pickle.load(f)

model_defaults = {}
if os.path.exists(DEFAULTS_PATH):
    with open(DEFAULTS_PATH, "r") as f:
        model_defaults = json.load(f)


def format_inr(val):
    """Format currency in Indian Lakhs / Crores and comma-separated rupees."""
    if val is None or np.isnan(val):
        return "N/A", "N/A"
    val = max(0.0, float(val))
    exact_str = f"₹ {int(round(val)):,}"
    if val >= 10000000:
        short_str = f"₹ {val / 10000000.0:.2f} Crore"
    elif val >= 100000:
        short_str = f"₹ {val / 100000.0:.2f} Lakhs"
    else:
        short_str = exact_str
    return short_str, exact_str


def predict_price(car_data):
    """
    Accepts a dictionary with car features, transforms them, and returns predicted price.
    """
    selected_model = car_data.get("model", "").strip()
    if selected_model not in label_encoder.classes_:
        raise ValueError(f"Unknown car model '{selected_model}'. Please select a valid model.")

    encoded_model = label_encoder.transform([selected_model])[0]

    input_df = pd.DataFrame([{
        "model": encoded_model,
        "vehicle_age": float(car_data["vehicle_age"]),
        "km_driven": float(car_data["km_driven"]),
        "seller_type": str(car_data["seller_type"]),
        "fuel_type": str(car_data["fuel_type"]),
        "transmission_type": str(car_data["transmission_type"]),
        "mileage": float(car_data["mileage"]),
        "engine": float(car_data["engine"]),
        "max_power": float(car_data["max_power"]),
        "seats": int(car_data["seats"])
    }])

    transformed = preprocessor.transform(input_df)
    predicted_val = model.predict(transformed)[0]
    return max(0.0, float(predicted_val))


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        form_options=form_options,
        model_defaults=model_defaults,
        prediction=None,
        prediction_exact=None,
        input_data=None,
        error=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "model": request.form.get("model", ""),
            "vehicle_age": request.form.get("vehicle_age", ""),
            "km_driven": request.form.get("km_driven", ""),
            "seller_type": request.form.get("seller_type", ""),
            "fuel_type": request.form.get("fuel_type", ""),
            "transmission_type": request.form.get("transmission_type", ""),
            "mileage": request.form.get("mileage", ""),
            "engine": request.form.get("engine", ""),
            "max_power": request.form.get("max_power", ""),
            "seats": request.form.get("seats", "")
        }

        # Check for missing values
        for key, value in input_data.items():
            if value is None or str(value).strip() == "":
                raise ValueError(f"Please provide a valid value for '{key.replace('_', ' ').title()}'.")

        predicted_val = predict_price(input_data)
        short_price, exact_price = format_inr(predicted_val)

        return render_template(
            "index.html",
            form_options=form_options,
            model_defaults=model_defaults,
            prediction=short_price,
            prediction_exact=exact_price,
            raw_prediction=round(predicted_val, 2),
            input_data=input_data,
            error=None
        )
    except Exception as e:
        return render_template(
            "index.html",
            form_options=form_options,
            model_defaults=model_defaults,
            prediction=None,
            prediction_exact=None,
            input_data=request.form.to_dict() if request.form else None,
            error=str(e)
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic car price predictions."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON request body"}), 400

        required_fields = [
            "model", "vehicle_age", "km_driven", "seller_type",
            "fuel_type", "transmission_type", "mileage", "engine",
            "max_power", "seats"
        ]
        for f in required_fields:
            if f not in data:
                return jsonify({"status": "error", "message": f"Missing field '{f}'"}), 400

        pred_val = predict_price(data)
        short_price, exact_price = format_inr(pred_val)

        return jsonify({
            "status": "success",
            "predicted_price": round(pred_val, 2),
            "formatted_short": short_price,
            "formatted_exact": exact_price
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/options", methods=["GET"])
def api_options():
    return jsonify({
        "status": "success",
        "options": form_options,
        "defaults": model_defaults
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "XGBoost Regressor (Tuned)"})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
