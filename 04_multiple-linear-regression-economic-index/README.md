A Multiple Linear Regression project predicting stock index price using interest rate and unemployment rate, featuring EDA, correlation analysis, cross-validation, residual analysis, and OLS regression diagnostics with statsmodels.

## Multiple Linear Regression — Economic Index Prediction

* This project builds a Multiple Linear Regression model to predict an economic index price based on the interest rate and unemployment rate. It covers:
* Data cleaning (dropping irrelevant columns like year, month, index)
* Exploratory Data Analysis (pairplots, correlation heatmap, regression scatter plots)
* Feature standardization using StandardScaler
* Train-test split
* Model training using scikit-learn's LinearRegression
* Cross-validation using cross_val_score
* Performance evaluation: MAE, MSE, RMSE, R², and Adjusted R²
* Regression assumption checks: residual distribution (KDE plot) and residual vs. prediction scatter plot
* Statistical validation using statsmodels OLS regression summary
* 
Tech Stack: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels
