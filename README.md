# Credit Risk Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. 

## Overview of Analysis
Using the credit card dataset from LendingClub (a peer-to-peer lending services company), this analysis will apply machine learning using `imbalanced-learn` and `scikit-learn` libraries to build and evaluate modules using resampling by:
 1) ***oversampling*** the data using `RandomOverSampler` and `SMOTE` algorithms  
 2) ***undersamplying*** the data using the `ClusterCentroids` algorithm

---
Then, the dataset will be resampled with a combinatorial approach of over- and undersampling using the `SMOTEENN` algorithm in order to view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

--- 
Finally, the performance of these two machine learning models can be evaluated to to compare which is better at predicting credit risk: 
1) `BalancedRandomClassifier` 
2) `EasyEnsembleClassifier`

## Results

## Summary