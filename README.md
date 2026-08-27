# Titanic_Data-cleaning-preprocessing-For-Ml-
Titanic Dataset — Data Cleaning & Preprocessing Report
1. Introduction

The Titanic dataset is a well-known dataset used for learning data analysis and machine learning. It contains information about passengers who travelled on the Titanic, including passenger class, age, gender, family information, fare, and survival status.

The objective of this project was to clean and preprocess the raw Titanic dataset and prepare it for machine-learning applications.

2. Objective

The main objectives of the project were:

Import and explore the dataset.
Identify missing values and data-quality problems.
Handle missing values using appropriate imputation techniques.
Convert categorical variables into numerical variables.
Create useful features from existing data.
Identify potential outliers using visualization.
Remove selected extreme observations using the IQR method.
Standardize numerical features.
Save the final processed dataset.
3. Tools and Technologies

The following technologies were used:

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
VS Code
Git
GitHub
4. Dataset Exploration

The dataset was first loaded using Pandas.

Basic dataset properties were examined using:

head()
shape
info()
dtypes
describe()

Missing values and duplicate rows were also identified.

This initial exploration helped understand the structure and quality of the raw data before preprocessing.

5. Missing Value Handling

Missing values were identified using:

df.isnull().sum()
Age

The Age column contained missing values.

Median imputation was used:

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

Median was selected because it is less sensitive to extreme observations.

Embarked

Missing values in the Embarked column were replaced using the most frequent category:

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)
Cabin

Instead of attempting to predict missing cabin numbers, a new binary feature called CabinKnown was created.

1 → Cabin information available
0 → Cabin information unavailable

This preserves useful information contained in the missingness pattern.

6. Duplicate Removal

Duplicate rows were checked using:

df.duplicated().sum()

Duplicate observations were removed using:

df.drop_duplicates()

This prevents repeated observations from unnecessarily affecting the analysis.

7. Feature Engineering

Two new features were created.

FamilySize
df["FamilySize"] = (
    df["SibSp"] +
    df["Parch"] +
    1
)

FamilySize represents the total number of people in the passenger's immediate travelling family.

IsAlone
df["IsAlone"] = (
    df["FamilySize"] == 1
).astype(int)

This identifies whether a passenger travelled alone.

8. Feature Selection

The following columns were removed from the basic ML feature set:

PassengerId
Name
Ticket
Cabin

PassengerId is an identifier rather than a meaningful predictive feature.

Name and Ticket were excluded because they require additional feature extraction to be used effectively.

Cabin was replaced with the more useful CabinKnown feature.

9. Categorical Encoding

Machine-learning algorithms generally require numerical inputs.

Categorical features such as Sex and Embarked were converted into numerical variables using one-hot encoding.

pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True,
    dtype=int
)

This converted categorical information into machine-readable numerical features.

10. Outlier Detection

Potential outliers were visualized using boxplots.

Boxplots were generated for numerical variables such as:

Age
Fare
SibSp
Parch
FamilySize

The IQR method was used for selected numerical features.

The interquartile range was calculated as:

IQR = Q3 - Q1

The lower and upper limits were calculated as:

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR

Observations outside these limits were considered potential outliers.

Outlier treatment was performed cautiously because extreme values can sometimes represent legitimate observations.

11. Feature Scaling

Numerical features were standardized using StandardScaler.

The standardization formula is:

z = (x - μ) / σ

where:

x = original value
μ = mean
σ = standard deviation

The main numerical variables considered for scaling were:

Age
Fare
SibSp
Parch
FamilySize

Scaling ensures that variables with different numerical ranges can be processed on a comparable scale.

12. Final Dataset

After preprocessing, the dataset contained:

handled missing values
encoded categorical variables
engineered features
selected numerical features
standardized numerical variables
reduced data-quality issues

The final dataset was exported as:

data/processed/titanic_cleaned.csv
13. Project Workflow

The complete workflow was:

Raw Titanic Dataset
        ↓
Data Loading
        ↓
Data Exploration
        ↓
Missing Value Analysis
        ↓
Missing Value Treatment
        ↓
Duplicate Removal
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Outlier Detection
        ↓
Outlier Treatment
        ↓
Feature Scaling
        ↓
Processed Dataset
14. Key Learning Outcomes

This project provided practical experience with:

Pandas data manipulation
NumPy numerical operations
Missing-value treatment
Categorical encoding
Feature engineering
Outlier detection
IQR-based outlier treatment
Feature scaling
Data visualization
Python project organization
Git version control
GitHub repository management
15. Conclusion

The raw Titanic dataset was successfully transformed into a cleaner and machine-learning-ready dataset.

The project demonstrated the complete basic data preprocessing workflow, beginning with raw data exploration and ending with a standardized processed dataset.

The processed dataset can be used as input for the next stage of the project, where machine-learning models can be trained to predict passenger survival.
