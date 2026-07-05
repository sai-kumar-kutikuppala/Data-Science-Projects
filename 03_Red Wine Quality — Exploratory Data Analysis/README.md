# Red Wine Quality — Exploratory Data Analysis

Exploratory data analysis on the Red Wine Quality dataset, covering data cleaning, 
correlation analysis, and univariate/bivariate/multivariate visualizations to 
understand what drives wine quality.

##  What's Covered

- **Data Inspection** — shape, info, summary statistics, and null value checks
- **Data Cleaning** — identifying and removing duplicate records
- **Correlation Analysis** — heatmaps to identify features most linked to quality
- **Class Imbalance Check** — distribution analysis of the `quality` target variable
- **Univariate Analysis** — distribution plots (histograms + KDE) across all features
- **Bivariate & Multivariate Analysis** — pair plots, scatter plots, and categorical plots
- **Key Relationships Explored** — alcohol vs. pH vs. quality, pH distribution across quality levels

##  Tools Used

- Python
- Pandas
- Seaborn
- Matplotlib

##  Dataset

[Red Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) — UCI Machine Learning Repository

##  Key Insight

The dataset is **imbalanced**, with most wine samples clustered around quality 
scores of 5 and 6, which is an important consideration for any downstream 
modeling task.
