"""Train the holiday-package XGBoost model and save a pickle artifact."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.preprocess import clean_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Travel.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "holiday_package_model.pkl"


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["TypeofContact"] = df["TypeofContact"].fillna(df["TypeofContact"].mode()[0])
    df["DurationOfPitch"] = df["DurationOfPitch"].fillna(df["DurationOfPitch"].median())
    df["NumberOfFollowups"] = df["NumberOfFollowups"].fillna(df["NumberOfFollowups"].mode()[0])
    df["PreferredPropertyStar"] = df["PreferredPropertyStar"].fillna(
        df["PreferredPropertyStar"].mode()[0]
    )
    df["NumberOfTrips"] = df["NumberOfTrips"].fillna(df["NumberOfTrips"].median())
    df["NumberOfChildrenVisiting"] = df["NumberOfChildrenVisiting"].fillna(
        df["NumberOfChildrenVisiting"].mode()[0]
    )
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    return df


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = impute_missing(df)
    df = clean_features(df)

    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_features = X.select_dtypes(include=["object", "string"]).columns
    num_features = X.select_dtypes(exclude=["object", "string"]).columns

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
            ("StandardScaler", StandardScaler(), num_features),
        ]
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=12,
        learning_rate=0.1,
        colsample_bytree=1,
        eval_metric="logloss",
        random_state=42,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "metrics": metrics,
            "input_columns": list(X.columns),
        },
        MODEL_PATH,
    )

    print(f"Saved model to {MODEL_PATH}")
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
