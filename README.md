# Credit Risk Analysis
<sub>**Credit Card dataset:** [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)</sub>   
<sub>**Resampling Models code:** [credit_risk_resampling.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)</sub>   
<sub>**Ensemble Classifers code:** [credit_risk_ensemble.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)</sub>

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. In just 2019, more than 19 million Americans had at least one unsecured personal loan. That's a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.

From the [credit card dataset in LendingClub](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)<sub>(a peer-to-peer lending services company)</sub>, python will be used to build and evaluate several machine learning models to predict credit risk with the goal of helping banks and financial institutions better navigate anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

## Overview of Analysis
***This Credit Risk Analysis will apply machine learning through six different techniques to train and evaluate models with unbalanced classes with the purpose of trying to solve the challenge of predicting credit card risk***

Initially after reading the [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv) and performing Basic Data Cleaning:
- Split the Data into training and testing sets with Scikit-learn's `train_test_split` module
    - Create training variables *(or feature)* by converting the string values into numerical ones using the `get_dummies()` method
    - Create target *(or output)* variables
        - Check balance of target variables
            - *confirms imbalance in training set*
            - *counts the number of instances by class to verify both classes are same size* 

### Resampling Models to Predict Credit Risk
Through the `imbalanced-learn` and `scikit-learn` libraries, three *Resampling Models* will be employed in [credit_risk_resampling.ipynb](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb):
1) *Naive Random Oversampling*
    - resample training data using `RandomOverSampler' algorithm
2) *Synthetic Minority Oversampling Technique*
    - resample training data using `SMOTE` algorithm
3) *Cluster Centroid Undersamplying*
    - resample training data using `ClusterCentroids` algorithm

> For each of the above three resampling algorithms:
>    - use a random state of 1 to ensure consistency between tests
>    - use `LogisticRegression` classifier to make predictions and evaluate the model’s performance
>    - calculate the accuracy score of the model
>    - generate a confusion matrix
>    - print out the imbalanced classification report


### SMOTEENN algorithm to Predict Credit Risk
Using the `imbalanced-learn` and `scikit-learn` libraries, this fourth machine learning model will resample the data using a combinatorial approach of *over-* and *undersampling*. Building off the [credit_risk_resampling.ipynb file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb) from the first three resampling models with the already created training and target variables:
<ol start="4">
    <li><em>Combination Sampling</em> with <code>SMOTEENN</code> </li>
        <ol>
            <li> resample the training data using the <code>SMOTEENN</code> algorithm</li>
            <li><code>LogisticRegression</code> classifierto make predictions & evaluate the model’s performance
            <li>calculate the accuracy score of the model</li>
            <li>generate a confusion matrix</li>
            <li>print out the imbalanced classification report</li>
        </ol>
</ol>

### Ensemble Classifiers to Predict Credit Risk
Finally using `imblearn.ensemble` library, the performance of two different *ensemble classifiers* will resample the dataset by randomly undersampling boostrap samples to reduce bias.

Because the code for ensemble classifiers is separate in the [credit_risk_ensemble.ipynb file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb), a training and target variable will be created newly again after reading the [LoanStats_2019Q1.csv file](https://github.com/vzhang90/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv) and performing Basic Data Cleaning.

<ol start="5">
  <li><em>Balanced Random Forest Classifier</em></li>
    <ol>
        <li>resample training data using <code>BalancedRandomForestClassifier</code> algorithm with 100 estimators
        <li>calculate the accuracy score of the model, generate a confusion matrix, and print out the imbalanced classification report</li>
        <li>finally, list the features sorted in descending order <em>(from most to least important feature by feature importance)</em>, along with the feature score</li>
    </ol>
  <li><em>Easy Ensemble AdaBoost Classifier</em></li>
    <ol>
        <li>resample training data using <code>EasyEnsembleClassifier</code> algorithm with 100 estimators</li>
        <li>calculate the accuracy score of the model, generate a confusion matrix, and print out the imbalanced classification report</li>
    </ol>
</ol>

## Results
***When dealing with imbalanced classification problems, it is often useful to use metrics beyond just overall accuracy, as accuracy can be misleading in such scenarios. Six machine learning models were employed to calculate the balance accuracy score, precision, and recall scores (as shown in each model's imbalanced classification report below).***
>> **Balanced accuracy score:** the average of the recall scores of all classes, which is calculated by summing up the recall scores of all classes and dividing by the number of classes. It provides a more reliable evaluation of a model's performance when the classes are imbalanced, as it takes into account the fact that the model might be performing well on the majority class but poorly on the minority class. 

>> **Precision score:** The precision score is the ratio of true positives to the total number of positive predictions, measuring the proportion of positive predictions that were actually true positives.  

>> **Recall score:** the ratio of true positives to the total number of actual positives, measuring the proportion of actual positives that were correctly identified by the model.

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
**The six machine learning models employed in this Credit Risk Analysis:**

1) ***Naive Random Oversampling*** with `RandomOverSampler` randomly oversamples the minority class with `imblearn` library to increase number of minority class. This is most appropriate if one class has too few instances in the training set where we dneed to choose more instances from that class for training by oversampling until it's larger.

2) `SMOTE` reduces risk of ***oversampling*** by also increasing number of minority class, but it does not always outperform random sampling because of its vulnerability to outliers.

3) ***Undersamplying*** using `ClusterCentroids` identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.Because this algorithm only uses actual data to decrease size of majority class, it involves loss of data where there must be enough usable data. 

4) ***Combination sampling*** with `SMOTEENN` combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms to oversample the minority class with SMOTE and clean the resulting data with an undersamplying strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

5) ***Random Forest Classifier*** samples the data and build several smaller, simpler decision trees (combining many decision trees into a forest of trees). Because random forest algorithms can run efficiently on large datasets handling thousands of input variables without variable deletion, this machine learning module is very robust against overfitting, outliers, and nonlinear data as all of those weak learners are trained on different pieces of the data. Additionally, this algorithm can also be used to rank the importance of input variables in a natural way.
    
6) ***Easy Ensemble AdaBoost Classifier*** uses *Boosting technique* to combine weak learners sequentially into a combined result, as one model learns from the mistakes of the previous model. In *AdaBoost*, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated until the error rate is minimized.

---

Comparatively between the imbalanced classification reports of the six machine learning models:
- the **Easy Ensemble AdaBoost Classifier** using `EasyEnsembleClassifier` is best in predicting both the majority and minority classes because of its highest ***Balanced Accuracy Score=0.86***
- As shown with the ***Recall Score=0.94***, the **Easy Ensemble AdaBoost Classifier** machine learning model using the `EasyEnsembleClassifier` algorithm is also best at making the fewest false negative predictions
- All models make equally few false negative predictions as shown in each of ***Precision Scores=0.99***



Between these six machine learning models, I would recommend the **Easy Ensemble AdaBoost Classifer** machine learning model using the `EasyEnsembleClassifier` algorithm to predict credit risk.