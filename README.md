# Titanic_Data-cleaning-preprocessing-For-Ml-
# 🚢 Titanic Data Cleaning & Preprocessing

## Project Overview

This project focuses on cleaning and preprocessing the Kaggle Titanic dataset for machine learning.

The objective is to transform raw passenger data into a clean, numerical, and standardized dataset suitable for machine-learning algorithms.

## Objectives

* Explore the raw Titanic dataset
* Identify missing values
* Handle missing data
* Remove duplicate records
* Encode categorical variables
* Create useful features
* Detect and handle outliers
* Standardize numerical features
* Export the processed dataset

## Dataset

The project uses the Kaggle Titanic dataset.

The main dataset contains passenger information such as:

* Passenger class
* Passenger sex
* Passenger age
* Number of siblings/spouses
* Number of parents/children
* Passenger fare
* Embarkation port
* Survival status

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Git
* GitHub

## Project Structure

```text
titanic-data-preprocessing/
│
├── data/
│   ├── raw/
│   └── 
│
├── outputs
│   └── data_cleaning_preprocessing.ipynb
│
├── src/
│   ├── data_loading.py
│   ├── data_cleaning.py
│   ├── 
│   ├── scaling encoding.py
│   ├── outlier_detection.py
│   └──prepr
│
├── visualizations/
├── reports/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Preprocessing Steps

### 1. Data Exploration

The dataset was inspected using:

* `head()`
* `shape`
* `info()`
* `dtypes`
* `describe()`
* missing-value analysis
* duplicate-value analysis

### 2. Missing Value Treatment

Numerical missing values such as Age were handled using median imputation.

Categorical missing values such as Embarked were handled using mode imputation.

The Cabin feature was transformed into a `CabinKnown` feature to preserve information about whether cabin information was available.

### 3. Feature Engineering

Two additional features were created:

* `FamilySize`
* `IsAlone`

These features provide additional information about the passenger's family situation.

### 4. Categorical Encoding

Categorical features such as Sex and Embarked were converted into numerical representations using one-hot encoding.

### 5. Outlier Detection

Boxplots were used to visualize potential outliers.

The IQR method was used to identify extreme numerical observations.

### 6. Feature Scaling

Numerical variables were standardized using `StandardScaler`.

Standardization transforms numerical variables to a common scale with approximately zero mean and unit variance.

## Output

The cleaned dataset is stored in:

```text
data/processed/titanic_cleaned.csv
```

## Learning Outcomes

Through this project, I learned how to:

* inspect raw datasets
* identify data-quality issues
* handle missing values
* engineer features
* encode categorical variables
* detect potential outliers
* standardize numerical features
* organize a machine-learning preprocessing project
* use Git and GitHub for version control

## Version

Current release:

`v1.0.0`

## Future Scope

The cleaned dataset can be used for the next stage of the project:

* Train/test splitting
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Model evaluation
* Survival prediction


The processed dataset can be used as input for the next stage of the project, where machine-learning models can be trained to predict passenger survival.
