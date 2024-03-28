# ML-Project---Pune-Real-Estate-Price-Prediction 

# Real Estate Price Prediction for Pune Region in India

## Project Overview

### Objective (Problem Statement) 
In this project, we aim to predict house per sq.ft. prices for properties in Pune city. We will use Linear Regression to predict the prices for properties. The goal is to help predict house prices based on different property features.

---

### Data
We have a dataset with around 13158 rows and 9 columns

---

### Tech Stack
- Language: `Python`
- Libraries: `scikit-learn`, `pandas`, `NumPy`, `matplotlib`, `seaborn`, `xgboost`

---

## Project Phases

### 1. Data Cleaning
- Import required libraries and load the dataset.
- Perform preliminary data exploration.
- Identify and remove outliers.
- Remove redundant feature columns.
- Handle missing values.
- Regularize categorical columns.
- Save the cleaned data.

### 2. Data Analysis
- Import the cleaned dataset.
- Convert binary columns to dummy variables.
- Perform feature engineering.
- Conduct univariate and bivariate analysis.
- Check for correlations.
- Select relevant features.
- Scale the data.
- Save the final updated dataset.

### 3. Model Building
- Prepare the data.
- Split the dataset into training and testing sets.
- Build various regression models, including Linear Regression, Ridge Regression, Lasso Regressor, Elastic Net, Random Forest Regressor, XGBoost Regressor, K-Nearest Neighbours Regressor, and Support Vector Regressor.

### 4. Model Validation
- Assess model performance using metrics like Mean Squared Error (MSE) and R2 score.
- Create residual plots for both training and testing data.

### 5. Hyperparameter Tuning
- Perform grid search and cross-validation for the chosen regressor.

### 6. Making Predictions
- Fit the model and make predictions on the test data.

### 7. Feature Importance
- Check for feature importance to identify the most influential factors in predicting house prices.

### 8. Model Comparison
- Compare the performance of different models to choose the best one.

---

