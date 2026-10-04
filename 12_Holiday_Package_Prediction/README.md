# Holiday Package Purchase Prediction Using XGBoost Classification

## 📌 Project Overview

This project uses **Machine Learning classification techniques to predict whether a customer will purchase a holiday package**.

The project is based on customer information collected by **Trips & Travel.Com**. The company wants to introduce a new **Wellness Tourism Package** and use existing customer data to make its marketing campaigns more efficient.

Instead of contacting customers randomly, the goal is to identify customers who are more likely to purchase the package.

Several classification algorithms are compared, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, and XGBoost.

The final **XGBoost Classifier** is hyperparameter-tuned using `RandomizedSearchCV` and achieves a **95.09% test accuracy** with an **ROC-AUC score of 0.8882**.

---

## 🎯 Problem Statement

Trips & Travel.Com currently offers five types of holiday packages:

* Basic
* Standard
* Deluxe
* Super Deluxe
* King

Based on previous customer data, approximately 18% of customers purchased a package. However, the company spent a significant amount on marketing because customers were contacted randomly.

The objective of this project is to use customer information to predict whether a customer is likely to purchase the proposed **Wellness Tourism Package**.

The target variable is:

```text
ProdTaken
```

where:

* `0` → Customer did not purchase the package
* `1` → Customer purchased the package

---

## 📊 Dataset

The dataset is obtained from the **Holiday Package Purchase Prediction** dataset on Kaggle.

* **Rows:** 4,888
* **Columns:** 20
* **Target variable:** `ProdTaken`

The dataset contains demographic, contact, travel, and income-related information about customers.

---

## 📝 Feature Description

The following features are used to predict whether a customer will purchase the package.

| Feature                    | Description                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `CustomerID`               | Unique identification number assigned to each customer. It is removed before model training because it does not provide useful predictive information. |
| `ProdTaken`                | Target variable indicating whether the customer purchased the holiday package. `0` = No, `1` = Yes.                                                    |
| `Age`                      | Age of the customer.                                                                                                                                   |
| `TypeofContact`            | Describes how the customer was contacted, such as through a company-generated contact or an existing customer contact.                                 |
| `CityTier`                 | Classification of the customer's city based on its development/business level.                                                                         |
| `DurationOfPitch`          | Duration of the sales pitch or presentation given to the customer.                                                                                     |
| `Occupation`               | Occupation category of the customer.                                                                                                                   |
| `Gender`                   | Gender of the customer.                                                                                                                                |
| `NumberOfPersonVisiting`   | Number of people visiting as part of the trip, excluding children.                                                                                     |
| `NumberOfFollowups`        | Number of follow-up contacts made with the customer after the initial interaction.                                                                     |
| `ProductPitched`           | Type of holiday product/package presented to the customer.                                                                                             |
| `PreferredPropertyStar`    | Customer's preferred hotel/property star rating.                                                                                                       |
| `MaritalStatus`            | Marital status of the customer.                                                                                                                        |
| `NumberOfTrips`            | Number of trips taken by the customer.                                                                                                                 |
| `Passport`                 | Indicates whether the customer has a passport.                                                                                                         |
| `PitchSatisfactionScore`   | Customer's satisfaction score for the sales pitch.                                                                                                     |
| `OwnCar`                   | Indicates whether the customer owns a car.                                                                                                             |
| `NumberOfChildrenVisiting` | Number of children accompanying the customer on the trip.                                                                                              |
| `Designation`              | Job/designation level of the customer.                                                                                                                 |
| `MonthlyIncome`            | Monthly income of the customer.                                                                                                                        |
| `TotalVisiting`            | New feature created by adding `NumberOfPersonVisiting` and `NumberOfChildrenVisiting`.                                                                 |

> **Note:** `CustomerID` is removed during preprocessing, so it is not used as a model feature.

---

## 🧹 Data Cleaning

The notebook performs several data-cleaning operations before training the models.

### 1. Handling Missing Values

Missing values are identified and handled using either the **median** or **mode**, depending on the feature.

| Feature                    | Missing Value Treatment |
| -------------------------- | ----------------------- |
| `Age`                      | Median                  |
| `TypeofContact`            | Mode                    |
| `DurationOfPitch`          | Median                  |
| `NumberOfFollowups`        | Mode                    |
| `PreferredPropertyStar`    | Mode                    |
| `NumberOfTrips`            | Median                  |
| `NumberOfChildrenVisiting` | Mode                    |
| `MonthlyIncome`            | Median                  |

### 2. Correcting Categories

Some inconsistent category values are corrected.

For example:

```text
Fe Male → Female
Single → Unmarried
```

### 3. Removing Customer ID

`CustomerID` is removed because it is only an identifier and does not represent a meaningful customer characteristic.

---

## 🔧 Feature Engineering

A new feature called `TotalVisiting` is created:

```text
TotalVisiting =
NumberOfPersonVisiting + NumberOfChildrenVisiting
```

After creating this feature, the original:

```text
NumberOfPersonVisiting
NumberOfChildrenVisiting
```

columns are removed.

This combines the total number of people visiting into a single feature.

---

## 🔄 Data Preprocessing

The dataset contains both numerical and categorical features.

### Numerical Features

Numerical features are standardized using:

```python
StandardScaler()
```

Standardization puts numerical features on a comparable scale.

### Categorical Features

Categorical features are converted into numerical form using:

```python
OneHotEncoder(drop='first')
```

The preprocessing is implemented using Scikit-learn's `ColumnTransformer`.

The preprocessing is fitted only on the training data and then applied to the test data.

---

## ✂️ Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The training data is used to train the models, while the test data is used to evaluate their performance on unseen customers.

---

## 🤖 Classification Models

The following classification algorithms are compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. AdaBoost Classifier
6. XGBoost Classifier

The models are evaluated using:

* Accuracy
* F1 Score
* Precision
* Recall
* ROC-AUC Score

---

## 📈 Initial Model Comparison

The initial test-set results are:

| Model               |   Accuracy |   F1 Score |  Precision |     Recall |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     83.54% |     0.8078 |     0.6829 |     0.2932 |     0.6301 |
| Decision Tree       |     91.92% |     0.9185 |     0.8077 |     0.7696 |     0.8626 |
| Random Forest       |     92.74% |     0.9221 |     0.9545 |     0.6597 |     0.8260 |
| Gradient Boosting   |     85.89% |     0.8398 |     0.7732 |     0.3927 |     0.6824 |
| AdaBoost            |     83.54% |     0.8115 |     0.6630 |     0.3194 |     0.6400 |
| **XGBoost**         | **93.56%** | **0.9318** | **0.9507** | **0.7068** | **0.8490** |

Among the initial models, **XGBoost achieved the highest test accuracy of 93.56%**.

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning is performed for:

* Random Forest
* XGBoost

using **RandomizedSearchCV**.

The search uses:

```text
100 parameter combinations
3-fold cross-validation
```

### XGBoost Parameters Considered

The following parameters are searched:

* `learning_rate`
* `max_depth`
* `n_estimators`
* `colsample_bytree`

### Best XGBoost Parameters

The notebook obtained:

```text
n_estimators = 200
max_depth = 12
learning_rate = 0.1
colsample_bytree = 1
```

---

## 🏆 Final XGBoost Performance

After hyperparameter tuning, the final XGBoost model achieved the following results on the test set:

| Metric    | Test Score |
| --------- | ---------: |
| Accuracy  | **95.09%** |
| F1 Score  | **0.9490** |
| Precision | **0.9554** |
| Recall    | **0.7853** |
| ROC-AUC   | **0.8882** |

The tuned XGBoost model improved the test accuracy from **93.56% to 95.09%**.

---

## 📈 ROC-AUC Curve

The project also generates a **Receiver Operating Characteristic (ROC) curve** for the final XGBoost model.

The final model achieved:

```text
ROC-AUC = 0.8882
```

A higher ROC-AUC indicates better ability of the classifier to distinguish between customers who purchase the package and those who do not.

The generated ROC curve is saved as:

```text
auc.png
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Plotly
* Jupyter Notebook

---

## 📁 Project Structure

```text
Holiday-Package-Purchase-Prediction/
│
├── XgboostBoost Classification Implementation.ipynb
│
├── auc.png
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
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost jupyter
```

### 3. Place the dataset

Place the `Travel.csv` dataset in the same directory as the notebook.

### 4. Open the notebook

```bash
jupyter notebook
```

Then open:

```text
XgboostBoost Classification Implementation.ipynb
```

Run the cells sequentially to reproduce the complete analysis and model training process.

---

## 🎯 Key Takeaways

* Built a complete **customer purchase prediction classification pipeline**.
* Performed data cleaning and missing-value treatment.
* Corrected inconsistent categorical values.
* Created a new `TotalVisiting` feature through feature engineering.
* Applied **One-Hot Encoding** to categorical variables.
* Applied **StandardScaler** to numerical variables.
* Compared six classification algorithms.
* Used **RandomizedSearchCV** for XGBoost hyperparameter tuning.
* Achieved **95.09% test accuracy** with the tuned XGBoost model.
* Achieved an **ROC-AUC score of 0.8882**.
* Generated an ROC curve to evaluate the final classifier.

---

## 📌 Conclusion

This project demonstrates how customer data can be used to predict the likelihood of purchasing a holiday package. Among the models tested, the tuned **XGBoost Classifier** provided the best test performance, achieving **95.09% accuracy**.

Such a prediction model can help a travel company identify customers who are more likely to purchase a package, allowing marketing efforts to be more targeted instead of contacting customers randomly.

---

## ⚠️ Note

This project is intended for **educational and machine learning practice purposes**. The predictions depend on the patterns present in the dataset and should not be treated as guaranteed customer behavior.

