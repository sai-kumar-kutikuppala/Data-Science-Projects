# Decision Tree Classifier – Iris Flower Classification

## 📌 Project Overview

This project demonstrates the implementation of a **Decision Tree Classifier** for classifying Iris flowers into different species based on their sepal and petal measurements.

The project uses the **Iris dataset** provided by Scikit-learn. A Decision Tree Classifier is first trained and evaluated, followed by **hyperparameter tuning using GridSearchCV** to explore different model configurations.

---

## 📊 Dataset

The project uses the built-in **Iris dataset from Scikit-learn**.

The dataset contains **150 samples** belonging to 3 different Iris species.

### Input Features

The model uses four independent features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Target Classes

The target variable contains three classes:

| Class | Species    |
| ----- | ---------- |
| 0     | Setosa     |
| 1     | Versicolor |
| 2     | Virginica  |

---

## ⚙️ Project Workflow

The project follows these steps:

1. Import the required Python libraries.
2. Load the Iris dataset using Scikit-learn.
3. Explore the dataset description and target values.
4. Create a DataFrame containing the input features.
5. Separate independent features (`X`) and dependent variable (`y`).
6. Split the dataset into training and testing sets.
7. Train a **Decision Tree Classifier**.
8. Visualize the trained Decision Tree.
9. Make predictions on the test data.
10. Generate a confusion matrix and classification report.
11. Perform hyperparameter tuning using **GridSearchCV**.
12. Evaluate the tuned model using accuracy, confusion matrix, precision, recall, and F1-score.

---

## 🤖 Machine Learning Model

### Decision Tree Classifier

A Decision Tree Classifier is a supervised machine learning algorithm used for classification problems.

The model makes decisions by splitting the dataset based on feature values and creates a tree-like structure consisting of:

* Root node
* Decision nodes
* Branches
* Leaf nodes

In this project, the Decision Tree is used to classify an Iris flower into one of three species.

---

## ✂️ Train-Test Split

The dataset is divided into training and testing data using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10
)
```

Therefore:

* **80%** of the data is used for training.
* **20%** of the data is used for testing.
* The test set contains **30 samples**.

---

## 🌳 Decision Tree Visualization

The trained Decision Tree is visualized using Scikit-learn's `plot_tree()` function.

This helps understand how the model makes classification decisions based on the Iris flower features.

---

## 📈 Initial Model Performance

The initial Decision Tree Classifier achieved an accuracy of approximately **96.7%** on the test set.

### Classification Report

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| Setosa               |      1.00 |   1.00 |     1.00 |
| Versicolor           |      1.00 |   0.92 |     0.96 |
| Virginica            |      0.88 |   1.00 |     0.93 |
| **Overall Accuracy** |           |        | **0.97** |

### Confusion Matrix

```text
[[10  0  0]
 [ 0 12  1]
 [ 0  0  7]]
```

---

## 🔧 Hyperparameter Tuning

The project uses **GridSearchCV** with **5-fold cross-validation** to search for better Decision Tree configurations.

The following parameters are explored:

```python
criterion:
    gini
    entropy
    log_loss

splitter:
    best
    random

max_depth:
    1, 2, 3, 4, 5

max_features:
    auto
    sqrt
    log2
```

### Best Parameters Found

```text
criterion = gini
max_depth = 5
max_features = sqrt
splitter = random
```

The best cross-validation accuracy obtained during GridSearchCV was approximately:

```text
95.83%
```

---

## 📊 Tuned Model Performance

After applying the best parameters found by GridSearchCV, the model achieved approximately **86.7% accuracy** on the test set.

### Classification Report

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| Setosa               |      1.00 |   1.00 |     1.00 |
| Versicolor           |      1.00 |   0.69 |     0.82 |
| Virginica            |      0.64 |   1.00 |     0.78 |
| **Overall Accuracy** |           |        | **0.87** |

### Confusion Matrix

```text
[[10  0  0]
 [ 0  9  4]
 [ 0  0  7]]
```

The notebook demonstrates that **hyperparameter tuning does not always guarantee better performance on a particular test split**. In this case, the tuned model's test accuracy was lower than the initial model's accuracy.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## 📁 Project Structure

```text
Decision-Tree-Classifier/
│
├── Decision Tree Classifier Practical Implementation.ipynb
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
pip install pandas numpy matplotlib scikit-learn jupyter
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open:

```text
Decision Tree Classifier Practical Implementation.ipynb
```

Run the cells sequentially to reproduce the analysis, model training, visualization, hyperparameter tuning, and evaluation.

---

## 🎯 Key Takeaways

* Implemented a **Decision Tree Classifier** for Iris flower classification.
* Used four flower measurements as input features.
* Achieved approximately **96.7% accuracy** with the initial Decision Tree.
* Visualized the trained Decision Tree.
* Evaluated the model using a **confusion matrix and classification report**.
* Used **GridSearchCV with 5-fold cross-validation** for hyperparameter tuning.
* Compared the performance of the initial and tuned models.
* Demonstrated that model tuning should be evaluated on unseen test data rather than assuming that the tuned model will always perform better.

---

## 📌 Note

This project is created for **machine learning learning and practical implementation**. The Iris dataset is a standard educational dataset used to demonstrate classification algorithms.

