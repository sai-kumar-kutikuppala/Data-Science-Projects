"""
Training Script for Diabetes Progression Prediction using Decision Tree Regressor.
Replicates the notebook logic and saves pickle files to the models/ directory.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main():
    print("=" * 60)
    print("Training Decision Tree Regressor for Diabetes Prediction")
    print("=" * 60)

    # Resolve directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 1. Load the sklearn diabetes dataset (scaled version, same as notebook)
    dataset = load_diabetes()
    feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

    df = pd.DataFrame(dataset.data, columns=feature_names)
    y = dataset.target

    # Save scaled dataset CSV
    df_with_target = df.copy()
    df_with_target['target'] = y
    df_with_target.to_csv(os.path.join(data_dir, "diabetes.csv"), index=False)
    print("[OK] Scaled dataset saved to data/diabetes.csv")

    # 2. Save raw (unscaled) dataset for reference
    dataset_raw = load_diabetes(scaled=False)
    df_raw = pd.DataFrame(dataset_raw.data, columns=feature_names)
    df_raw['target'] = y
    df_raw.to_csv(os.path.join(data_dir, "diabetes_unscaled.csv"), index=False)
    print("[OK] Raw clinical dataset saved to data/diabetes_unscaled.csv")

    # 3. Train-test split (matching notebook: test_size=0.3, random_state=10)
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    print(f"[OK] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Train the selected model (best params from notebook's GridSearchCV)
    #    Notebook found: criterion='friedman_mse', max_depth=4, max_features='log2', splitter='random'
    #    Using 'squared_error' instead of deprecated 'friedman_mse' (they are equivalent)
    model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=4,
        max_features='log2',
        splitter='random',
        random_state=54
    )
    model.fit(X_train, y_train)

    # 5. Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n--- Model Evaluation on Test Set ---")
    print(f"R² Score              : {r2:.4f}")
    print(f"Mean Absolute Error   : {mae:.2f}")
    print(f"Mean Squared Error    : {mse:.2f}")
    print(f"Root Mean Squared Err : {rmse:.2f}")

    # 6. Build a StandardScaler to convert raw clinical inputs -> scaled space
    #    The sklearn diabetes dataset is scaled by: (x - mean) / (std * sqrt(n_samples))
    scaler = StandardScaler()
    scaler.fit(dataset_raw.data)
    n_samples = dataset_raw.data.shape[0]
    scaler.scale_ = scaler.scale_ * np.sqrt(n_samples)

    # Verify scaler precision
    transformed = scaler.transform(dataset_raw.data)
    max_err = np.max(np.abs(transformed - dataset.data))
    print(f"\n[OK] Scaler verified (max transform deviation: {max_err:.2e})")

    # 7. Save pickle files
    # a) Model
    pickle.dump(model, open(os.path.join(models_dir, "model.pkl"), "wb"))
    pickle.dump(model, open(os.path.join(models_dir, "decision_tree_model.pkl"), "wb"))
    print("[OK] Model saved to models/model.pkl and models/decision_tree_model.pkl")

    # b) Scaler
    pickle.dump(scaler, open(os.path.join(models_dir, "scaler.pkl"), "wb"))
    print("[OK] Scaler saved to models/scaler.pkl")

    # c) Feature names
    pickle.dump(feature_names, open(os.path.join(models_dir, "features.pkl"), "wb"))
    print("[OK] Features saved to models/features.pkl")

    # 8. Save metadata JSON with feature statistics for the web app
    feature_stats = {}
    for col in feature_names:
        raw_vals = df_raw[col]
        feature_stats[col] = {
            "raw_min": round(float(raw_vals.min()), 2),
            "raw_max": round(float(raw_vals.max()), 2),
            "raw_mean": round(float(raw_vals.mean()), 2),
            "description": ""
        }

    # Add human-readable descriptions
    descriptions = {
        "age": "Age (years)",
        "sex": "Sex (1 = Male, 2 = Female)",
        "bmi": "Body Mass Index (kg/m²)",
        "bp": "Average Blood Pressure (mm Hg)",
        "s1": "Total Serum Cholesterol (tc)",
        "s2": "Low-Density Lipoproteins (ldl)",
        "s3": "High-Density Lipoproteins (hdl)",
        "s4": "Total Cholesterol / HDL (tch)",
        "s5": "Log of Serum Triglycerides (ltg)",
        "s6": "Blood Sugar Level (glu)"
    }
    for col in feature_names:
        feature_stats[col]["description"] = descriptions.get(col, col)

    metadata = {
        "model_name": "DecisionTreeRegressor",
        "task": "Diabetes Progression Prediction",
        "parameters": {
            "criterion": "squared_error",
            "max_depth": 4,
            "max_features": "log2",
            "splitter": "random",
            "random_state": 54
        },
        "metrics": {
            "r2_score": round(r2, 4),
            "mean_absolute_error": round(mae, 2),
            "mean_squared_error": round(mse, 2),
            "root_mean_squared_error": round(rmse, 2)
        },
        "target_stats": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": round(float(y.mean()), 2)
        },
        "feature_names": feature_names,
        "feature_stats": feature_stats
    }

    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print("[OK] Metadata saved to models/model_metadata.json")

    # 9. Quick verification
    loaded_model = pickle.load(open(os.path.join(models_dir, "model.pkl"), "rb"))
    loaded_scaler = pickle.load(open(os.path.join(models_dir, "scaler.pkl"), "rb"))
    sample_pred = loaded_model.predict(X_test.iloc[0:1])
    actual = y_test.iloc[0] if hasattr(y_test, "iloc") else y_test[0]
    print(f"\n[VERIFY] Sample prediction: {sample_pred[0]:.2f}, Actual: {actual:.2f}")

    print("\n[DONE] Training and artifact generation completed successfully!")


if __name__ == "__main__":
    main()
