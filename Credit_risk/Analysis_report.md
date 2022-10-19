# Credit Risk Analysis Report 

## Overview of the Analysis

The purpose of the analysis to compare two versions of the dataset  for predicting creditworthiness of borrowers, by using the imbalanced-learn library, logistic regression model. First, prediction is with original dataset and then by resampled data by using the RandomOverSampler module from the imbalanced-learn library.

To come up with prediction, model uses historical data of lending services such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks and total debt to identify creditworthiness of borrowers by determining healthy loan and high-risk loans.

The model is predicting two variables i.e., healthy loans and high-risk loans. The total loans data provided to train a model were 75000 for low-risk loans and 2500 for high-risk loans.

The machine learning model begins with splitting the data into training and testing data. The lending data was divided into two datasets the labels set (y) from the “loan_status” column, and the features (X) DataFrame from the remaining columns. Then by using by train_test_split the data was divided into training dataset and testing dataset. This model uses 70% of data for training and 30% for testing.

Next, step was to create a logistic regression model with given data and to fit that model by using the training data to get the predictions for test data. Once we get the predictions, we can evaluate the model performance by calculate the accuracy score of the model, generating a confusion matrix and printing the classification report.

Final step is to predict a Logistic Regression Model with resampled training data. For this purpose, random oversampling has been used.  Random oversampling involves randomly duplicating examples from the minority class and adding them to the training dataset. To apply random oversampling, RandomOverSampler module from the imbalanced-learn library has been used to resample the data. Once the model goes through the process of create, fit and predict it is ready to evaluate the performance and comparison.

## Results

Logistic Regression Model with original imbalanced data and with Logistic Regression Model with resampled data

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  1. Accuracy 95%
  2. Precision for low risk loans 100%
  3. Precision for high risk loans 85%
  4. Recall for low risk loans 99%
  5. Recall for high risk loans 91%



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  1. Accuracy 99%
  2. Precision for low risk loans 100%
  3. Precision for high risk loans 84%
  4. Recall for low risk loans 99%
  5. Recall for high risk loans 99%


## Summary

Both the models performed very well. Imbalanced data model was able to predict both the high risk and low risk loans accurately by 95% whereas accuracy improved even further by 99% with resampled model. Both the models were great at correctly finding healthy loans with a precision of 100% and recall being 99% the reason being enough data was provided to models to be trained on low risk loans (75000 records) as compare to high risk loans (2500 records).

Imbalance model was able to find high risk loans 91% of the time with 85% precision whereas resampled model was better at finding high risk loans but the precision went down by 1%. Seems like resampled model became too ambitious in finding high risk loans.

So to summaries, resampled model performed better (f1 being 100% for low risk loans and 91% for high risk loans) as compare to original imbalance model (f1 being 100% for low risk loans and 88% for high risk loans). Still both models performed well but they might go in overfitting scenario. That means the model might not perform well with new unseen data. So before choosing any model further analysis must be done to identify overfitting possibilities.


