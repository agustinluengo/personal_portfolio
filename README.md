This repository stores my personal projects using Python for Data Analysis and ML.
Projects:
1. Breast Cancer (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
  a. breast_cancer_summary-V2 contains the main analysis, including EDA and comparing multiple ML models and their performance.
  b. breast_cancer_pca contains a second analysis ploting a 3D representation of the data to be able to see the clusters for the two classes and testing the accuracy of the SVC model using reduced dimensions.

2. Fraud Credit Card Transactions (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  a. unbalanced-data-creditcard-fraud-transactions focuses on creating a model able to predict exceptions as fraud transactions are in a highly unbalanced dataset. For this I've used hyperparameter tuning methods (GridSearchCV) and Cross-Validation (Stratified KFold) to make a robust and more trustworthy process of analysis.

3. Energy Consumption (https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
  a. electricity_consumption_notebook1 contains a time series analysis of seasonal data for energy consumption, the objective is to identify patters and provide insights for the different levels of granularity of the data (annual, quarterly, seasonally, monthly, daily, hourly) and create a ML model able to predict the energy cosumption for an unknown period.

4. Bank Transaction Dataset for Fraud Detection (https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)
  a. This dataset provides a detailed look into transactional behavior and financial activity patterns, ideal for exploring fraud detection and anomaly identification. It contains 2,512 samples of transaction data, covering various transaction attributes, customer demographics, and usage patterns. Each entry offers comprehensive insights into transaction behavior, enabling analysis for financial security and fraud detection applications.
