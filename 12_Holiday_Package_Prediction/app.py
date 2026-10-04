from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

from src.preprocess import clean_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "holiday_package_model.pkl"

app = Flask(__name__)
artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
metrics = artifact["metrics"]

FORM_DEFAULTS = {
    "Age": "36",
    "TypeofContact": "Self Enquiry",
    "CityTier": "1",
    "DurationOfPitch": "14",
    "Occupation": "Salaried",
    "Gender": "Male",
    "NumberOfPersonVisiting": "3",
    "NumberOfFollowups": "3",
    "ProductPitched": "Basic",
    "PreferredPropertyStar": "3",
    "MaritalStatus": "Married",
    "NumberOfTrips": "3",
    "Passport": "1",
    "PitchSatisfactionScore": "3",
    "OwnCar": "1",
    "NumberOfChildrenVisiting": "1",
    "Designation": "Executive",
    "MonthlyIncome": "22000",
}


def form_to_dataframe(form) -> pd.DataFrame:
    row = {
        "Age": float(form["Age"]),
        "TypeofContact": form["TypeofContact"],
        "CityTier": int(form["CityTier"]),
        "DurationOfPitch": float(form["DurationOfPitch"]),
        "Occupation": form["Occupation"],
        "Gender": form["Gender"],
        "NumberOfPersonVisiting": int(form["NumberOfPersonVisiting"]),
        "NumberOfFollowups": float(form["NumberOfFollowups"]),
        "ProductPitched": form["ProductPitched"],
        "PreferredPropertyStar": float(form["PreferredPropertyStar"]),
        "MaritalStatus": form["MaritalStatus"],
        "NumberOfTrips": float(form["NumberOfTrips"]),
        "Passport": int(form["Passport"]),
        "PitchSatisfactionScore": int(form["PitchSatisfactionScore"]),
        "OwnCar": int(form["OwnCar"]),
        "NumberOfChildrenVisiting": float(form["NumberOfChildrenVisiting"]),
        "Designation": form["Designation"],
        "MonthlyIncome": float(form["MonthlyIncome"]),
    }
    df = pd.DataFrame([row])
    return clean_features(df)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = FORM_DEFAULTS.copy()

    if request.method == "POST":
        values.update({key: request.form.get(key, "") for key in FORM_DEFAULTS})
        try:
            features = form_to_dataframe(request.form)
            probability = float(pipeline.predict_proba(features)[0][1])
            will_buy = probability >= 0.5
            prediction = {
                "will_buy": will_buy,
                "probability": round(probability * 100, 1),
                "label": "Likely to purchase" if will_buy else "Unlikely to purchase",
                "advice": (
                    "Prioritize this customer for the Wellness Tourism Package."
                    if will_buy
                    else "Keep this lead in a lower-priority nurture list."
                ),
            }
        except (ValueError, KeyError) as exc:
            error = f"Please check the form values. ({exc})"

    return render_template(
        "index.html",
        metrics=metrics,
        prediction=prediction,
        error=error,
        values=values,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
