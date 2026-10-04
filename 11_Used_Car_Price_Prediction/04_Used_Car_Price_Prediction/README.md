# 🚗 Used Car Price Prediction using XGBoost & Flask

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x%20%7C%203.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-blue.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application that estimates the fair market valuation of used cars in India. Built with a tuned **XGBoost Regressor**, serialized preprocessing pipelines, and a modern, responsive **Flask** web interface.

---

## 📌 Project Overview

Determining the fair selling price of a pre-owned vehicle depends on multiple non-linear factors such as age, mileage, engine displacement, fuel efficiency, transmission, and market demand for specific models. 

This project trains and compares multiple regression algorithms on **15,411** used car listings scraped from CarDekho India, hyperparameter-tunes an **XGBoost Regressor** via `RandomizedSearchCV`, serializes the pipelines into reusable pickle artifacts, and serves predictions through an interactive web app and REST API.

---

## ✨ Features

- **Tuned XGBoost Regressor:** Optimized for non-linear feature interactions and high regression accuracy ($R^2 \approx 0.95$ on test set).
- **Automated Preprocessing Pipeline:** Integrates `LabelEncoder` (120 vehicle models), `OneHotEncoder` (seller, fuel, and transmission types), and `StandardScaler` inside a scikit-learn `ColumnTransformer`.
- **Interactive Web Interface:** Modern, responsive UI with a sleek dark automotive theme.
- **Quick-Fill Presets:** 1-click test buttons for popular cars:
  - 🚗 Maruti Alto (Budget Hatchback)
  - 🚘 Hyundai i20 (Premium Hatchback)
  - 🏎️ Honda City (Sedan)
  - 🚙 Hyundai Creta (Compact SUV)
  - 🚐 Toyota Innova (MPV)
  - ✨ BMW X5 (Luxury SUV)
- **Model Auto-Fill Specs:** Select any car model and click *"Auto-fill typical specs"* to pre-populate realistic median values.
- **Dual Currency Formatting:** Outputs estimates in Indian numbering units (e.g., **₹ 8.30 Lakhs** / **₹ 1.25 Crore**) alongside exact rupee figures.
- **Developer REST API:** Programmatic endpoint (`POST /api/predict`) accepting JSON payloads for seamless integration into other apps.

---

## 📁 Repository Structure

```
├── app.py                     # Flask application & prediction endpoints
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── data/
│   ├── cardekho_imputated.csv # Cleaned & imputed CarDekho dataset (15,411 rows)
│   └── README.md
├── models/                    # Serialized machine learning artifacts
│   ├── form_options.pkl       # Categorical options for form dropdowns
│   ├── label_encoder.pkl      # Fitted LabelEncoder for car models
│   ├── model_defaults.json    # Median specs per car model for UI autofill
│   ├── preprocessor.pkl       # Fitted ColumnTransformer (OHE + Scaler)
│   ├── xgboost_model.pkl      # Tuned XGBoost Regressor model
│   └── README.md
├── notebooks/
│   └── Xgboost_Regression_Implementation.ipynb # End-to-end EDA, training & tuning
├── static/
│   └── style.css              # Custom styling (glassmorphism & responsive grid)
└── templates/
    └── index.html             # Web app template with dynamic JS helpers
```

---

## 📊 Dataset & Features

The dataset comprises **15,411 rows** and **13 attributes** from used cars listed across India:

| Feature | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `model` | Categorical | Car model name (120 classes) | `Alto`, `City`, `Creta`, `Innova` |
| `vehicle_age` | Numeric | Age of the vehicle in years | `1` to `29` |
| `km_driven` | Numeric | Total distance driven in km | `100` to `3,800,000` |
| `seller_type` | Categorical | Type of seller | `Individual`, `Dealer`, `Trustmark Dealer` |
| `fuel_type` | Categorical | Type of engine fuel | `Petrol`, `Diesel`, `CNG`, `LPG`, `Electric` |
| `transmission_type` | Categorical | Gearbox type | `Manual`, `Automatic` |
| `mileage` | Numeric | Fuel efficiency in kmpl | `4.0` to `33.54` |
| `engine` | Numeric | Displacement in CC | `793` to `6592` |
| `max_power` | Numeric | Power output in bhp | `38.4` to `626.0` |
| `seats` | Numeric | Seating capacity | `2`, `4`, `5`, `6`, `7`, `8` |
| **`selling_price`** | **Target** | **Final selling price (INR ₹)** | **₹ 40,000 to ₹ 39,500,000** |

---

## 🔬 Model Performance & Comparison

During model exploration, 9 regression models were trained and benchmarked on an 80/20 train-test split:

| Model | Test RMSE (₹) | Test MAE (₹) | Test $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **XGBoost Regressor (Tuned)** | **191,852** | **96,118** | **0.9511** |
| Random Forest Regressor (Tuned) | 211,535 | 98,281 | 0.9406 |
| Gradient Boosting Regressor | 256,543 | 126,580 | 0.9126 |
| K-Neighbors Regressor | 253,118 | 112,704 | 0.9149 |
| Decision Tree Regressor | 309,397 | 126,510 | 0.8728 |
| Ridge Regression | 502,533 | 279,557 | 0.6645 |
| Lasso Regression | 502,542 | 279,614 | 0.6645 |
| Linear Regression | 502,543 | 279,618 | 0.6645 |
| AdaBoost Regressor | 566,346 | 412,179 | 0.5739 |

### Best XGBoost Hyperparameters:
```python
{
    'n_estimators': 300,
    'learning_rate': 0.1,
    'max_depth': 5,
    'colsample_bytree': 0.5
}
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ (Python 3.10 – 3.13 supported)
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/used-car-price-prediction.git
cd used-car-price-prediction
```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

The trained model artifacts are already included in the `models/` directory. Start the Flask server directly:

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

*(Optional)* To re-train or reproduce the model serialization from scratch, open and execute `notebooks/Xgboost_Regression_Implementation.ipynb`.

---

## 🔌 REST API Documentation

### Endpoint: `POST /api/predict`

Calculates the predicted price for a vehicle passed as JSON.

#### Request Headers:
```http
Content-Type: application/json
```

#### Request Body:
```json
{
  "model": "City",
  "vehicle_age": 4,
  "km_driven": 35000,
  "seller_type": "Dealer",
  "fuel_type": "Petrol",
  "transmission_type": "Manual",
  "mileage": 17.8,
  "engine": 1498,
  "max_power": 117.3,
  "seats": 5
}
```

#### Example cURL Request:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "City",
    "vehicle_age": 4,
    "km_driven": 35000,
    "seller_type": "Dealer",
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "mileage": 17.8,
    "engine": 1498,
    "max_power": 117.3,
    "seats": 5
  }'
```

#### Sample JSON Response:
```json
{
  "status": "success",
  "predicted_price": 830185.19,
  "formatted_short": "₹ 8.30 Lakhs",
  "formatted_exact": "₹ 830,185"
}
```

### Additional Endpoints:
- `GET /health`: Health-check endpoint returning API and model status.
- `GET /api/options`: Returns all available model names and dropdown values.

---

## 🛠️ Tech Stack

- **Machine Learning:** XGBoost, Scikit-Learn, NumPy, Pandas
- **Web Framework:** Flask, Jinja2
- **Frontend:** HTML5, CSS3 (Modern Glassmorphism), Vanilla JavaScript
- **Serialization:** Pickle

---

## 📜 License

This project is licensed under the MIT License - feel free to use and modify it for your learning and applications.

---

## 🤝 Acknowledgements

- **Krish Naik** for educational guidance in Data Science & Machine Learning workflows.
- **CarDekho.com** for vehicle marketplace data.
