"""
Script to train the Decision Tree Classifier on the Iris dataset
and save the model + metadata as pickle files into the models/ directory.
Run from the project root:
    python notebooks/save_model.py
"""

import os
import pickle
import warnings

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
iris = load_iris()

FEATURE_NAMES = ["sepal length in cm", "sepal width", "petal length", "petal width"]
TARGET_NAMES  = list(iris.target_names)          # ['setosa', 'versicolor', 'virginica']

X = pd.DataFrame(iris["data"], columns=FEATURE_NAMES)
y = iris["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

# ── 2. Hyper-parameter search (mirrors the notebook) ─────────────────────────
param_grid = {
    "criterion":  ["gini", "entropy", "log_loss"],
    "splitter":   ["best", "random"],
    "max_depth":  [1, 2, 3, 4, 5],
    "max_features": ["sqrt", "log2", None],      # 'auto' removed in sklearn ≥1.2
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ── 3. Evaluate ───────────────────────────────────────────────────────────────
y_pred      = best_model.predict(X_test)
test_acc    = accuracy_score(y_test, y_pred)
cv_best     = grid.best_score_

print("Best params  :", grid.best_params_)
print(f"CV accuracy  : {cv_best:.4f}")
print(f"Test accuracy: {test_acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

# ── 4. Save pickle files ──────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

with open("models/decision_tree_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(FEATURE_NAMES, f)

with open("models/target_names.pkl", "wb") as f:
    pickle.dump([str(n) for n in TARGET_NAMES], f)

print("Pickle files saved in models/")
print("  - models/decision_tree_model.pkl")
print("  - models/feature_names.pkl")
print("  - models/target_names.pkl")
