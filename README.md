# Credit Risk Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. In just 2019, more than 19 million Americans had at least one unsecured personal loan. That's a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.

From the credit card dataset in LendingClub <sub>(a peer-to-peer lending services company)</sub>, Python will be used to build and evaluate several machine learning models to predict credit risk with the goal of helping banks and financial institutions better navigate anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

Resources: [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)

## Overview of Analysis
 > This Credit Risk Analysis applies six machine learning to investigate how each compare at *Predicting Credit Risk* by using **Resampling** in order to:
 > - *view* the **count of the target classes**
 > - *train* a **logistic regression classifier**
 > - *calculate* the **balanced accuracy score**
 > - *generate* a **confusion matrix**
 > - *generate* a **classification report**

 ---
 Initially through the `imbalanced-learn` and `scikit-learn` libraries , **Resample** the dataset to *Predict Credit Risk*
 [credit_risk_resampling.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)
 - ***oversampling*** using the `RandomOverSampler` and `SMOTE` algorithms  
 - ***undersamplying*** using the `ClusterCentroids` algorithm

Then using the `SMOTEENN` algorithm, the dataset will be **resampled** with a combinatorial approach of ***over-*** and ***undersampling*** in order to:


--- 
Finally using `imblearn.ensemble` library, the performance of two different ensemble classifiers,`BalancedRandomClassifier` and `EasyEnsembleClassifier`, will be used to predict credit risk and evaluate each model. Using both algorithms, resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
[credit_risk_ensemble.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)
  

## Results

Six machine learning models were used to calculate the balance accuracy score, precision, and recall scores:
1) Naive Random Oversampling using `RandomOverSampler` algorithm
![Naive Random Oversampling Imbalanced Classification Report](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/naive_random_sampling_imbclass.png)
2) `SMOTE` Oversampling
3) Undersampling using `ClusterCentroids` algorithm
4) Combinatio (Over and Under) Sampling with `SMOTEENN` algorithm
5) Balanced Forest Classifier
6) Easy Ensemble AdaBoost Classifier

## Summary