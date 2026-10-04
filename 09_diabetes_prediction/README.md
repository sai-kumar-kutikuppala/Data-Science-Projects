# Diabetes Prediction Using Decision Tree Regressor

## 📌 Project Overview

This project uses **Machine Learning regression** to predict a quantitative measure of diabetes disease progression using patient-related medical features.

A **Decision Tree Regressor** is trained on the Scikit-learn Diabetes dataset. The project also performs **hyperparameter tuning using GridSearchCV** to find better model parameters and evaluates the final model using standard regression metrics.

---

## 📊 Dataset

The project uses the built-in **Diabetes dataset from Scikit-learn**.

* **Number of samples:** 442
* **Number of input features:** 10
* **Target:** Quantitative measure of disease progression one year after baseline

### Features

| Feature | Description                      |
| ------- | -------------------------------- |
| `age`   | Age in years                     |
| `sex`   | Sex                              |
| `bmi`   | Body Mass Index                  |
| `bp`    | Average blood pressure           |
| `s1`    | Total cholesterol                |
| `s2`    | Low-density lipoproteins         |
| `s3`    | High-density lipoproteins        |
| `s4`    | Thyroid-related measurement      |
| `s5`    | Triglyceride-related measurement |
| `s6`    | Blood sugar level                |

The dataset features provided by Scikit-learn are already mean-centered and scaled.

---

## ⚙️ Project Workflow

The project follows these main steps:

1. Load the Diabetes dataset using Scikit-learn.
2. Convert the dataset into a Pandas DataFrame.
3. Separate the input features (`X`) and target variable (`y`).
4. Split the data into training and testing sets.
5. Perform correlation analysis on the input features.
6. Train a Decision Tree Regressor.
7. Perform hyperparameter tuning using `GridSearchCV`.
8. Select the best hyperparameters.
9. Make predictions on the test data.
10. Evaluate the model using regression metrics.
11. Visualize the trained decision tree.
12. Save the trained model and supporting files using Pickle.
13. Load the saved files again and verify prediction.

---

## 🤖 Machine Learning Model

### Decision Tree Regressor

A Decision Tree Regressor is used because the target variable is a **continuous numerical value** representing disease progression.

The model recursively divides the data into smaller groups based on feature values and predicts a numerical value at the leaf nodes.

---

## 🔧 Hyperparameter Tuning

`GridSearchCV` with **5-fold cross-validation** is used to search for suitable Decision Tree parameters.

The parameters explored include:

* `criterion`
* `splitter`
* `max_depth`
* `max_features`

The GridSearchCV scoring metric used is **negative mean squared error**.

### Best Parameters

The notebook obtained:

```text
criterion = friedman_mse
max_depth = 4
max_features = log2
splitter = random
```

These parameters were then used to create the selected Decision Tree Regressor.

---

## 📈 Model Evaluation

The model was evaluated using:

* **R² Score**
* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**

### Results

| Metric   |     Score |
| -------- | --------: |
| R² Score |    0.2037 |
| MAE      |   59.4483 |
| MSE      | 5038.8774 |

These results represent the performance obtained in the notebook's test split.

---

## 🌳 Decision Tree Visualization

The project also visualizes the trained Decision Tree using Scikit-learn's `plot_tree()` function.

This makes it possible to see how the model makes regression decisions based on the input features.

---

## 💾 Saved Model Files

The notebook saves the trained components inside the `models` directory:

```text
models/
├── decision_tree_model.pkl
├── model.pkl
├── scaler.pkl
└── features.pkl
```

### Files

* `decision_tree_model.pkl` — trained Decision Tree Regressor
* `model.pkl` — saved copy of the trained model
* `scaler.pkl` — saved feature scaler
* `features.pkl` — list of feature names

The notebook also verifies that the saved model and supporting files can be loaded successfully and used to generate a prediction.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Pickle
* Jupyter Notebook

---

## 📁 Project Structure

```text
Diabetes-Prediction/
│
├── Diabetes Prediction Using Decision Tree Regressor.ipynb
│
├── models/
│   ├── decision_tree_model.pkl
│   ├── model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install the required libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Open the notebook

```bash
jupyter notebook
```

Then open:

```text
Diabetes Prediction Using Decision Tree Regressor.ipynb
```

Run the cells sequentially to reproduce the analysis, model training, evaluation, visualization, and model saving steps.

---

## 🎯 Key Takeaways

* The project demonstrates a complete **machine learning regression workflow**.
* A **Decision Tree Regressor** is used for predicting diabetes disease progression.
* `GridSearchCV` is used for hyperparameter tuning.
* The model is evaluated using multiple regression metrics.
* The trained model and supporting preprocessing information are saved using Pickle.
* The saved model is loaded again to verify that it can be used for prediction.

---

## 📌 Note

This project demonstrates machine learning on the **Scikit-learn Diabetes dataset** and is intended for educational and machine learning practice purposes. The model should not be considered a medical diagnostic system.

