# Used Car Price Prediction Using XGBoost Regression

## 📌 Project Overview

This project uses **Machine Learning regression techniques to predict the selling price of used cars** based on their specifications and other vehicle-related information.

The project compares multiple regression algorithms and identifies **XGBoost Regressor** as the best-performing model after hyperparameter tuning.

The trained XGBoost model, preprocessing pipeline, label encoder, and form options are also saved using Pickle for further use in a web application.

---

## 📊 Dataset

The project uses a used-car dataset containing **15,411 records**.

The target variable is:

```text
selling_price
```

### Input Features

| Feature             | Description          |
| ------------------- | -------------------- |
| `model`             | Car model            |
| `vehicle_age`       | Age of the vehicle   |
| `km_driven`         | Kilometers driven    |
| `seller_type`       | Type of seller       |
| `fuel_type`         | Fuel type            |
| `transmission_type` | Transmission type    |
| `mileage`           | Vehicle mileage      |
| `engine`            | Engine capacity      |
| `max_power`         | Maximum engine power |
| `seats`             | Number of seats      |

The dataset contains **120 different car models**.

---

## ⚙️ Project Workflow

The project follows these major steps:

1. Load the used-car dataset.
2. Explore the dataset and its features.
3. Identify numerical and categorical features.
4. Separate independent features (`X`) and target variable (`y`).
5. Encode the categorical `model` feature using `LabelEncoder`.
6. Apply **One-Hot Encoding** to categorical features:

   * `seller_type`
   * `fuel_type`
   * `transmission_type`
7. Apply **StandardScaler** to numerical features.
8. Split the dataset into training and testing sets.
9. Train and compare multiple regression models.
10. Evaluate models using MAE, RMSE, and R² Score.
11. Perform hyperparameter tuning for Random Forest and XGBoost.
12. Retrain the models using the best parameters.
13. Select XGBoost as the final model.
14. Save the preprocessing objects and trained model for deployment.

---

## 🔄 Data Preprocessing

### Label Encoding

The `model` column contains **120 unique car models**, so it is converted into numerical values using `LabelEncoder`.

### One-Hot Encoding

The following categorical features are one-hot encoded:

```text
seller_type
fuel_type
transmission_type
```

`drop='first'` is used to avoid redundant dummy variables.

### Feature Scaling

Numerical features are standardized using:

```python
StandardScaler()
```

The preprocessing is implemented using a `ColumnTransformer`.

After preprocessing, the dataset contains **14 input features**.

---

## 🤖 Models Compared

The project compares the following regression algorithms:

* Linear Regression
* Lasso Regression
* Ridge Regression
* K-Neighbors Regressor
* Decision Tree Regressor
* Random Forest Regressor
* AdaBoost Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

The models are evaluated using:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

---

## 📈 Initial Model Comparison

The initial test-set results obtained in the notebook are:

| Model                 |           RMSE |        MAE | R² Score |
| --------------------- | -------------: | ---------: | -------: |
| Linear Regression     |     502,543.59 | 279,618.58 |   0.6645 |
| Lasso                 |     502,542.67 | 279,614.75 |   0.6645 |
| Ridge                 |     502,533.82 | 279,557.22 |   0.6645 |
| K-Neighbors Regressor |     253,118.42 | 112,704.35 |   0.9149 |
| Decision Tree         |     309,397.78 | 126,510.29 |   0.8728 |
| Random Forest         |     228,899.95 | 102,234.13 |   0.9304 |
| AdaBoost              |     566,346.28 | 412,179.93 |   0.5739 |
| Gradient Boosting     |     256,543.91 | 126,580.90 |   0.9126 |
| **XGBoost**           | **216,809.08** |      **—** |    **—** |

XGBoost showed strong performance among the initial models and was selected for further tuning.

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning is performed using **RandomizedSearchCV** with:

* **100 parameter combinations**
* **3-fold cross-validation**

### XGBoost Parameters Tuned

The search includes:

```text
learning_rate
max_depth
n_estimators
colsample_bytree
```

### Best XGBoost Parameters

The best parameters obtained were:

```text
n_estimators = 300
max_depth = 5
learning_rate = 0.1
colsample_bytree = 0.5
```

---

## 🏆 Final XGBoost Model

After retraining XGBoost using the best hyperparameters, the final model achieved:

| Dataset  |           RMSE |           MAE |   R² Score |
| -------- | -------------: | ------------: | ---------: |
| Training |     115,252.44 |     75,276.86 |     0.9836 |
| Testing  | **191,852.08** | **96,118.82** | **0.9511** |

The final model achieved an **R² score of 0.9511 on the test set**, meaning it explains a large proportion of the variation in used-car selling prices in this dataset.

---

## 💾 Model Deployment Preparation

The project saves the required objects using Pickle so that the trained model can later be integrated into a web application.

The following files are saved:

```text
models/
├── label_encoder.pkl
├── preprocessor.pkl
├── xgboost_model.pkl
└── form_options.pkl
```

### Saved Files

* `label_encoder.pkl` — fitted LabelEncoder for car models
* `preprocessor.pkl` — fitted ColumnTransformer containing encoding and scaling
* `xgboost_model.pkl` — trained XGBoost regression model
* `form_options.pkl` — valid dropdown options for the web form

The form options contain:

* 120 car models
* Seller types
* Fuel types
* Transmission types

This allows a future Flask application to use the same preprocessing and trained model for predictions.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**
* **Pickle**
* **Jupyter Notebook**

---

## 📁 Project Structure

```text
Used-Car-Price-Prediction/
│
├── notebooks/
│   └── Xgboost Regression Implementation(1).ipynb
│
├── models/
│   ├── label_encoder.pkl
│   ├── preprocessor.pkl
│   ├── xgboost_model.pkl
│   └── form_options.pkl
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
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open:

```text
Xgboost Regression Implementation(1).ipynb
```

Run the cells sequentially to reproduce the preprocessing, model comparison, hyperparameter tuning, evaluation, and model-saving steps.

---

## 🎯 Key Takeaways

* Built a complete **used-car price prediction regression pipeline**.
* Performed categorical encoding and numerical feature scaling.
* Compared **9 different regression algorithms**.
* Used **RandomizedSearchCV** for XGBoost hyperparameter tuning.
* Achieved a **0.9511 R² score** on the test set with the tuned XGBoost model.
* Saved the trained model and preprocessing objects for future deployment.
* Prepared valid categorical options that can be used by a Flask-based prediction form.

---

## 📌 Note

This project is intended for **machine learning learning and practical implementation**. The predicted prices are based on patterns learned from the dataset and should not be treated as guaranteed market prices.

