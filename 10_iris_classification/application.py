import pickle

import pandas as pd
from flask import Flask, render_template, request

application = Flask(__name__)
app = application

# ── Load model & metadata ────────────────────────────────────────────────────
model         = pickle.load(open("models/decision_tree_model.pkl", "rb"))
feature_names = pickle.load(open("models/feature_names.pkl", "rb"))
target_names  = pickle.load(open("models/target_names.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error  = None

    if request.method == "POST":
        try:
            sepal_length = float(request.form.get("sepal_length"))
            sepal_width  = float(request.form.get("sepal_width"))
            petal_length = float(request.form.get("petal_length"))
            petal_width  = float(request.form.get("petal_width"))

            features = pd.DataFrame(
                [[sepal_length, sepal_width, petal_length, petal_width]],
                columns=feature_names,
            )
            pred_idx = model.predict(features)[0]
            proba    = model.predict_proba(features)[0]
            result   = {
                "species":     target_names[pred_idx],
                "confidence":  f"{proba[pred_idx] * 100:.1f}",
                "all_probas":  list(zip(target_names, [f"{p*100:.1f}" for p in proba])),
            }
        except Exception as exc:
            error = str(exc)

    return render_template("predict.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
