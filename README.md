# salary-prediction-Linear-Regression
# Linear Regression Salary Prediction

## Salary Prediction using Linear Regression

This project predicts an employee’s **Annual Salary** using a **Linear Regression model** based on demographic, professional, and organizational features.  
It demonstrates a complete **data analytics + machine learning workflow** with exploratory analysis, statistical correlation, model training, and prediction.


## Project Overview

- Cleans and preprocesses a real-world salary dataset
- Performs exploratory data analysis (EDA)
- Analyzes numerical and categorical feature relationships
- Trains a Linear Regression model
- Evaluates model performance using regression metrics
- Predicts salary based on user input

This project is suitable for **Data Analyst / Data Science / Machine Learning internships**.



## Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- SciPy  



## Dataset

- **File:** `salary_prediction_30000.csv`
- Contains employee-related features such as:
  - Age
  - Experience
  - Department
  - Job Level
  - Education
  - City
  - Company Size
  - Remote Work Type
  - Performance Rating
- Target variable: **AnnualSalary**


## Features Implemented

### Data Preprocessing
- Handling missing numerical values using mean
- Handling missing categorical values using mode
- Removing duplicate records
- Removing rows with missing salary values

### Exploratory Data Analysis (EDA)
- Correlation heatmap for numerical features
- Salary distribution analysis
- Average salary by:
  - Department
  - Job Level
  - Education

### Categorical Feature Analysis
- Salary grouped into **Low / Medium / High**
- Categorical relationship measured using **Cramér’s V**
- Heatmap visualization for categorical correlations



## Machine Learning Model

- **Model Used:** Linear Regression
- **Feature Encoding:** One-hot encoding for categorical variables
- **Train-Test Split:** 80% training, 20% testing



## Model Evaluation Metrics

- **Root Mean Squared Error (RMSE)**
- **R² Score**
- Coefficient-based feature importance
- Actual vs Predicted salary visualization



## Visualizations Included

- Correlation heatmap (numerical features)
- Categorical vs salary correlation heatmap
- Actual vs Predicted Salary scatter plot
- Feature coefficient importance ranking
