# 🔥 Forest Fire Weather Index (FWI) Prediction

A machine learning web application that predicts the **Fire Weather Index (FWI)** — a measure of fire intensity — based on meteorological data from the Algerian Forest Fires Dataset.

## 📌 About the Project

This project uses regression-based machine learning to estimate the FWI using environmental parameters such as temperature, humidity, wind speed, and rainfall, along with fire behavior indices like FFMC, DMC, and ISI. The trained model is deployed as a Flask web application, allowing users to input real-time weather data and receive an instant FWI prediction.

## 🧠 Features

- Predicts Forest Fire Weather Index using a trained regression model
- Simple, responsive Bootstrap-based web interface
- Built with Flask
- 
## 📊 Dataset

The model is trained on the **Algerian Forest Fires Dataset**, which contains weather observations and fire indices collected from two regions in Algeria (Bejaia and Sidi Bel-abbes).

**Input features:**
| Feature | Description |
|---|---|
| Temperature | Temperature in °C |
| RH | Relative Humidity (%) |
| Ws | Wind speed (km/h) |
| Rain | Rainfall (mm) |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| ISI | Initial Spread Index |
| Classes | Fire occurrence class |
| Region | Region indicator (0 or 1) |

**Output:** FWI (Fire Weather Index) — a continuous value representing fire intensity risk.

## 🛠️ Tech Stack

- **Language:** Python
- **ML Libraries:** Scikit-learn, NumPy, Pandas
- **Web Framework:** Flask
- **Frontend:** HTML, Bootstrap
