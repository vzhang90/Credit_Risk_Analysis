# Credit Risk Analysis
<sub>**Credit Card dataset:** [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)</sub>   
<sub>**Resampling Models code:** [credit_risk_resampling.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)</sub>   
<sub>**Ensemble Classifers code:** [credit_risk_ensemble.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)</sub>

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. In just 2019, more than 19 million Americans had at least one unsecured personal loan. That's a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.

From the [credit card dataset in LendingClub](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)<sub>(a peer-to-peer lending services company)</sub>, Python will be used to build and evaluate several machine learning models to predict credit risk with the goal of helping banks and financial institutions better navigate anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

## Overview of Analysis
***This Credit Risk Analysis will apply machine learning through six different techniques to train and evaluate models with unbalanced classes with the purpose of trying to solve the challenge of predicting credit card risk***

Initially after reading the [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv) and performing Basic Data Cleaning:
- Split the Data into training and testing sets with Scikit-learn's `train_test_split` module
    - Create training variables *(or feature)* by converting the string values into numerical ones using the `get_dummies()` method
    - Create the target *(or output)* variables
        - Check balance of target variables
            - *confirms imbalance in training set*
            - *counts the number of instances by class to verify both classes are same size* 

### Resampling Models to Predict Credit Risk
Through the `imbalanced-learn` and `scikit-learn` libraries, three *Resampling Models* will be employed [credit_risk_resampling.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb):
1) *Naive Random Oversampling* using `RandomOverSampler` algorithm 
    - randomly oversamples the minority class with `imblearn` library
    - to increase number of minority class
2) *Oversampling* using `SMOTE` algorithm  
    - to increase number of minority class
3) *Undersamplying* using `ClusterCentroids` algorithm
    - only uses actual data to decrease size of majority class
        -involves loss of data so ust be enough usable data

> For each of the above three resampling algorithms:
>    - Used `LogisticRegression` classifier to make predictions and evaluate the model’s performance
>    - Calculated the accuracy score of the model
>    - Generated a confusion matrix
>    - Printed out the imbalanced classification report


### SMOTEENN algorithm to Predict Credit Risk
Using the `imbalanced-learn` and `scikit-learn` libraries, the fourth machine learning model will resample the data using a combinatorial approach of *over-* and *undersampling*. Building off the [credit_risk_resampling.ipynb file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb) from the first three resampling models with the already created training and target variables:
<ol start="4">
    <li><em>Combination (Over & Under) Sampling</em> using <code>SMOTEENN</code> algorithm</li>
        <ol>
            <li> Resample the training data using the <code>SMOTEENN</code> algorithm</li>
            <li> After the data was resampled, the <code>LogisticRegression</code> classifier is employed to make predictions & evaluate the model’s performance
            <li>Calculated the accuracy score of the model</li>
            <li>Generated a confusion matrix</li>
            <li>Printed out the imbalanced classification report</li>
        </ol>
</ol>

### Ensemble Classifiers to Predict Credit Risk
Finally using `imblearn.ensemble` library, the performance of two different *ensemble classifiers* will resample the dataset by randomly undersampling boostrap samples to reduce bias.

Because the code for ensemble classifiers is separate in the [credit_risk_ensemble.ipynb file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb), a training and target variable will be created newly again after reading the [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv) and performing Basic Data Cleaning.

<ol start="5">
  <li><em>Balanced Random Forest Classifier</em></li>
    <ol>
        <li>Resampled training data using <code>BalancedRandomForestClassifier</code> algorithm with 100 estimators
        <li>Then, calculated the accuracy score of the model, generated a confusion matrix, and printed out the imbalanced classification report</li>
        <li>Finally, listed the features sorted in descending order <em>(from most to least important feature by feature importance)</em>, along with the feature score</li>
    </ol>
  <li><em>Easy Ensemble AdaBoost Classifier</em></li>
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
    - *Balanced Accuracy Score:* 0.39
    - *Precision Score:* 0.99
    - *Recall Score:* 0.65

---

2) `SMOTE` **Oversampling**
![SMOTE imblanace classification report](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/SMOTE_oversampling_imbclass.png)
    - *Balanced Accuracy Score:* 0.42
    - *Precision Score:* 0.99
    - *Recall Score:* 0.66
---

3) **Undersampling** using `ClusterCentroids`
![ClusterCentroids classification report imbalanced](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/SMOTE_oversampling_imbclass.png)
    - *Balanced Accuracy Score:* 0.42
    - *Precision Score:* 0.99
    - *Recall Score:* 0.66

---

4) **Combination (Over & Under) Sampling** with `SMOTEENN`
![SMOTEEN classification report imbalanced](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/SMOTEENN_combosampling_imbclass.png)
    - *Balanced Accuracy Score:* 0.41
    - *Precision Score:* 0.99
    - *Recall Score:* 0.58

---

5) **`BalancedRandomForestClassifier`**
![balanced forest classifier](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/balanced_random_forest_classifier_imbclass.png)
    - *Balanced Accuracy Score:* 0.62
    - *Precision Score:* 0.99
    - *Recall Score:* 0.58

---

6) **Easy Ensemble AdaBoost Classifier** with `EasyEnsembleClassifier`
![ECC classification report imbalanced](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/images/ECC_imbclass.png)
    - *Balanced Accuracy Score:* 0.86
    - *Precision Score:* 0.99
    - *Recall Score:* 0.94

---

## Summary
When dealing with imbalanced classification problems, it is often useful to use metrics beyond just overall accuracy, as accuracy can be misleading in such scenarios.
>> **Balanced accuracy score:** the average of the recall scores of all classes, which is calculated by summing up the recall scores of all classes and dividing by the number of classes. It provides a more reliable evaluation of a model's performance when the classes are imbalanced, as it takes into account the fact that the model might be performing well on the majority class but poorly on the minority class. 

>> **Precision score:** The precision score is the ratio of true positives to the total number of positive predictions, measuring the proportion of positive predictions that were actually true positives.  

>> **Recall score:** the ratio of true positives to the total number of actual positives, measuring the proportion of actual positives that were correctly identified by the model.


Comparatively between the imbalanced classification reports of the six machine learning models:
- the **Easy Ensemble AdaBoost Classifier** using `EasyEnsembleClassifier` is best in predicting both the majority and minority classes because of its highest ***Balanced Accuracy Score=0.86***
- As shown with the ***Recall Score=0.94***, the **Easy Ensemble AdaBoost Classifier** machine learning model using the `EasyEnsembleClassifier` algorithm is also best at making the fewest false negative predictions
- All models make equally few false negative predictions as shown in each of ***Precision Scores=0.99***

Between these six machine learning models, I would recommend the **Easy Ensemble AdaBoost Classifer** machine learning model using the `EasyEnsembleClassifier` algorithm to use in predicting credit risk.