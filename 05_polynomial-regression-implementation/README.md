A hands-on implementation of Polynomial Regression using synthetic quadratic data, comparing linear vs. polynomial fits, exploring degree selection with PolynomialFeatures, and automating experimentation using sklearn.Pipeline.

## Polynomial Regression Implementation
This project demonstrates how Polynomial Regression improves upon Simple Linear Regression when data has a non-linear relationship. It covers:

Generating synthetic non-linear data from a quadratic equation (y = 0.5x² + 1.5x + 2 + noise)

Fitting a baseline Simple Linear Regression model and visualizing its poor fit

Applying PolynomialFeatures (degree 2 and 3) to transform features

Comparing R² scores across linear vs. polynomial models

Visualizing model predictions on new/unseen data

Building a reusable Pipeline to experiment with different polynomial degrees (e.g., degree 6) in one function

Tech Stack: Python, NumPy, Pandas, Matplotlib, Scikit-learn
