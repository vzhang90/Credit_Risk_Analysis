# Credit Risk Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. In just 2019, more than 19 million Americans had at least one unsecured personal loan. That's a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.

From the credit card dataset in LendingClub <sub>(a peer-to-peer lending services company)</sub>, Python will be used to build and evaluate several machine learning models to predict credit risk with the goal of helping banks and financial institutions better navigate anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

> <sub>**Resources:** [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)</sub>   
> <sub>**Resampling Models code:** [credit_risk_resampling.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)</sub>   
> <sub>**Ensemble Classifers code:** [credit_risk_ensemble.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)</sub>

## Overview of Analysis
 This Credit Risk Analysis applies six machine learning to investigate how each compares at *Predicting Credit Risk*

 ---
### Resampling Models to Predict Credit Risk
Initially through the `imbalanced-learn` and `scikit-learn` libraries, three **Resampling Models** will each view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
1) ***Naive Random Oversampling*** using the `RandomOverSampler` algorithm 
2) ***Oversampling*** using the `SMOTE` algorithms  
3) ***Undersamplying*** using the `ClusterCentroids` algorithm

### Combination (Over and Under) Sampling with SMOTEENN algorithm to Predict Credit Risk
4) Then using the `SMOTEENN` algorithm, the dataset will be **resampled** with a combinatorial approach of ***over-*** and ***undersampling***
 
### Ensemble Classifiers to Predict Credit Risk
Finally using `imblearn.ensemble` library, the performance of two different ensemble classifiers will resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
<ol start="5">
  <li><code>BalancedRandomForestClassifier</code> algorithm</li>
  <li><code>EasyEnsembleClassifier</code> algorithm</li>
</ol>

## Results
Six machine learning models were used to calculate the balance accuracy score, precision, and recall scores:
1) **Naive Random Oversampling** using `RandomOverSampler`
![Naive Random Oversampling Imbalanced Classification Report](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/naive_random_sampling_imbclass.png)
2) `SMOTE` **Oversampling**
![SMOTE imblanace classification report](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/SMOTE_oversampling_imbclass.png)
3) **Undersampling** using `ClusterCentroids`
![ClusterCentroids classification report imbalanced](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/SMOTE_oversampling_imbclass.png)
4) **Combination (Over and Under) Sampling** with `SMOTEENN`
![SMOTEEN classification report imbalanced](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/SMOTEENN_combosampling_imbclass.png)
5) **`BalancedRandomForestClassifier`**
![balanced forest classifier](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/balanced_random_forest_classifier_imbclass.png)
6) **Easy Ensemble AdaBoost Classifier** with `EasyEnsembleClassifier`
![ECC classification report imbalanced](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/ECC_imbclass.png)

## Summary