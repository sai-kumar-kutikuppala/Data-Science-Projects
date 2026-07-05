# Flight Price Prediction — EDA & Feature Engineering

Exploratory data analysis and feature engineering on a flight price dataset, 
transforming raw categorical and datetime fields into a clean, model-ready 
dataset for price prediction.

##  What's Covered

- **Data Inspection** — shape, structure, and summary statistics
- **Datetime Feature Extraction** — splitting `Date_of_Journey` into Date, Month, and Year
- **Time Feature Extraction** — parsing `Arrival_Time` and `Dep_Time` into hour/minute components
- **Duration Parsing** — converting flight `Duration` into separate hour and minute fields
- **Categorical Cleanup** — handling `Total_Stops` (missing values mapped to numeric stop counts)
- **Feature Selection** — dropping redundant columns (`Route`, `Additional_Info`) 
  after extracting useful information
- **One-Hot Encoding** — encoding `Airline`, `Source`, and `Destination` using 
  `sklearn`'s `OneHotEncoder`
- **Final Dataset Assembly** — merging encoded features into a model-ready dataframe

##  Tools Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (`OneHotEncoder`)

##  Dataset

Flight Price Prediction dataset (`flight_price.xlsx`)

##  Key Outcome

Raw flight data with mixed date/time strings and categorical text fields was 
transformed into a fully numeric, encoded dataset ready to be fed into a 
machine learning model.
