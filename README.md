# Credit Risk Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. In just 2019, more than 19 million Americans had at least one unsecured personal loan. That's a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.

This analysis will use Python to build and evaluate several machine learning models to predict credit risk to help banks and financial institutions better navigate anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

## Overview of Analysis
Using the credit card dataset from LendingClub (a peer-to-peer lending services company), this analysis will apply machine learning using `imbalanced-learn` and `scikit-learn` libraries to build and evaluate modules using resampling by:
 1) ***oversampling*** the data using `RandomOverSampler` and `SMOTE` algorithms  
 2) ***undersamplying*** the data using the `ClusterCentroids` algorithm

---
Then, the dataset will be resampled with a combinatorial approach of ***over-*** and ***undersampling*** using the `SMOTEENN` algorithm 
- in order to **view the count of the target classes**, **train a logistic regression classifier**, **calculate the balanced accuracy score**, **generate a confusion matrix**, and **generate a classification report**

--- 
Finally, the performance of these two machine learning models can be evaluated to to compare which is better at predicting credit risk: 
           `BalancedRandomClassifier`    or     `EasyEnsembleClassifier`?

## Results

## Summary