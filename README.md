# Credit Risk Analysis
<sub>**Credit Card dataset:** [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)</sub>   
<sub>**Resampling Models code:** [credit_risk_resampling.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)</sub>   
<sub>**Ensemble Classifers code:** [credit_risk_ensemble.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)</sub>

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. In just 2019, more than 19 million Americans had at least one unsecured personal loan. That's a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.

From the credit card dataset in [LendingClub](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)<sub>(a peer-to-peer lending services company)</sub>, Python will be used to build and evaluate several machine learning models to predict credit risk with the goal of helping banks and financial institutions better navigate anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

## Overview of Analysis
 > ***This Credit Risk Analysis will apply machine learning through six different techniques to train and evaluate models with unbalanced classes with the purpose in trying to solve the challenge of credit card risk.***

Initially after reading the [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv) and performing Basic Data Cleaning:
- Create the training variables by converting the string values into numerical ones using the `get_dummies()` method
- Create the target variables
- Check the balance of the target variables

### Resampling Models to Predict Credit Risk
Through the `imbalanced-learn` and `scikit-learn` libraries, three *Resampling Models* will each view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
1) *Naive Random Oversampling* using `RandomOverSampler` algorithm 
2) *Oversampling* using `SMOTE` algorithm  
3) *Undersamplying* using `ClusterCentroids` algorithm

> For each of the above three resampling algorithms:
>    - Used `LogisticRegression` classifier to make predictions and evaluate the model’s performance
>    - Calculated the accuracy score of the model
>    - Generated a confusion matrix
>    - Printed out the imbalanced classification report


### SMOTEENN algorithm to Predict Credit Risk
Using the `imbalanced-learn` and `scikit-learn` libraries, the fourth machine learning model will resample the data using a combinatorial approach of *over-* and *undersampling*. Continue using [credit_risk_resampling.ipynb file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb) from the first three resampling models with the already created training and target variables to view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
<ol start="4">
    <li><em>Combination (Over & Under) Sampling</em> using <code>SMOTEENN</code> algorithm</li>
        <ol>
            <li> resample the training data using the <code>SMOTEENN</code> algorithm</li>
            <li> After the data was resampled, the <code>LogisticRegression</code> classifier is employed to make predictions & evaluate the model’s performance
            <li>Calculated the accuracy score of the model</li>
            <li>Generated a confusion matrix</li>
            <li>Printed out the imbalanced classification report</li>
        </ol>
</ol>

### Ensemble Classifiers to Predict Credit Risk
Finally using `imblearn.ensemble` library, the performance of two different *ensemble classifiers* will resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

Because the code for ensemble classifiers is separate in the [credit_risk_ensemble.ipynb file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb), a training and target variable will be created again after reading the [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv) and performing Basic Data Cleaning.

<ol start="5">
  <li><em>Balanced Random Forest Classifier</em> using <code>BalancedRandomForestClassifier</code> algorithm</li>
    <ol>
        <li>Resampled training data using <code>BalancedRandomForestClassifier</code> algorithm with 100 estimators
        <li>Then, calculated the accuracy score of the model, generated a confusion matrix, and printed out the imbalanced classification report</li>
        <li>Finally, printed the feature importance sorted in descending order (from most to least important feature), along with the feature score</li>
    </ol>
  <li><em>Easy Ensemble AdaBoost Classifier</em> using <code>EasyEnsembleClassifier</code> algorithm</li>
    <ol>
        <li>Resampled the training data using <code>EasyEnsembleClassifier</code> algorithm with 100 estimators</li>
        <li>After resampling data, calculated the accuracy score of the model, generated a confusion matrix, and printed out the imbalanced classification report</li>
    </ol>
</ol>

## Results
***Six machine learning models were used to calculate the balance accuracy score, precision, and recall scores:***

---
1) **Naive Random Oversampling** using `RandomOverSampler`
![Naive Random Oversampling Imbalanced Classification Report](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/naive_random_sampling_imbclass.png)
    - Balanced Accuracy Score: 0.39
    - Precision Score: 0.99
    - Recall Score: 0.65
---
2) `SMOTE` **Oversampling**
![SMOTE imblanace classification report](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/SMOTE_oversampling_imbclass.png)
    - Balanced Accuracy Score: 0.42
    - Precision Score: 0.99
    - Recall Score: 0.66
---
3) **Undersampling** using `ClusterCentroids`
![ClusterCentroids classification report imbalanced](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/SMOTE_oversampling_imbclass.png)
    - Balanced Accuracy Score: 0.42
    - Precision Score: 0.99
    - Recall Score: 0.66
---
4) **Combination (Over & Under) Sampling** with `SMOTEENN`
![SMOTEEN classification report imbalanced](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/SMOTEENN_combosampling_imbclass.png)
    - Balanced Accuracy Score: 0.41
    - Precision Score: 0.99
    - Recall Score: 0.58
---
5) **`BalancedRandomForestClassifier`**
![balanced forest classifier](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/balanced_random_forest_classifier_imbclass.png)
    - Balanced Accuracy Score: 0.62
    - Precision Score: 0.99
    - Recall Score: 0.58
---
6) **Easy Ensemble AdaBoost Classifier** with `EasyEnsembleClassifier`
![ECC classification report imbalanced](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/ECC_imbclass.png)
    - Balanced Accuracy Score: 0.86
    - Precision Score: 0.99
    - Recall Score: 0.94
---

## Summary