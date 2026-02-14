# 2024dc04259_ML_A2_Diabetes_Prediction
Diabetes Prediction Using Machine Learning: To predict whether a patient is diabetic or non-diabetic based on medical and demographic features
-----
**1. Problem Statement**

The objective of this project is to build and evaluate multiple Machine Learning classification models to predict whether a patient is diabetic or non-diabetic based on medical and demographic features.

The project aims to compare six different ML algorithms using standard evaluation metrics and determine the best-performing model for the dataset.

-----
**2. Dataset Description**

The dataset consists of patient medical records containing various health-related attributes used to predict diabetes.

**2.1.	Features Include:**

- Age
- Gender
- Blood Pressure
- Cholesterol
- Glucose Level
- BMI
- Triglycerides (TG)
- Other clinical measurements

` `**2.2.	Target Variable:**

- **CLASS**
  - 1 - Diabetic and Pre-Diabetic (both considered as diabetic)
  - 0 - Non-Diabetic

`   `**2.3.	Preprocessing Steps:**

- Removed duplicate labels from CLASS column (Target Column)
- Encoded categorical variables (Gender)
- Train-test split
- Feature scaling (for models requiring normalization)
-----
**3.    Models Used:**

The following six ML models were implemented and evaluated:-

3\.1.	   Logistic Regression

3\.2.	   Decision Tree

3\.3	    k-Nearest Neighbors (kNN)

3\.4.     Naive Bayes

3\.5.     Random Forest (Ensemble)

3\.6.	    XGBoost (Ensemble)

-----
**4.  Model Comparison Table**

|**ML Model Name**|**Accuracy**|**AUC**|**Precision**|**Recall**|**F1 Score**|**MCC**|
| :- | :- | :- | :- | :- | :- | :- |
|Logistic Regression|0\.9700|0\.8992|0\.9779|0\.9888|0\.9833|0\.8347|
|Decision Tree|0\.9900|1\.0000|1\.0000|0\.9888|0\.9944|0\.9502|
|KNN|0\.9450|0\.8011|0\.9565|0\.9832|0\.9697|0\.6806|
|Naive Bayes|0\.8950|0\.9274|0\.8950|1\.0000|0\.9446|0\.0000|
|Random Forest (Ensemble)|1\.0000|1\.0000|1\.0000|1\.0000|1\.0000|1\.0000|
|XGBoost (Ensemble)|1\.0000|1\.0000|1\.0000|1\.0000|1\.0000|1\.0000|

**5.  Observations on Model Performance *(3 Marks)***

|**ML Model Name**|**Observation about Model Performance**|
| :-: | :-: |
|Logistic Regression|Acts as a strong baseline model for binary classification. Performs well when data is linearly separable.|
|Decision Tree|Achieved very high accuracy and F1 score. May overfit if depth is not controlled.|
|kNN|Provided good recall and balanced F1 score. Performance depends on scaling and optimal selection of k.|
|Naive Bayes|Achieved perfect recall but lower overall robustness. Assumes independence between features, which may reduce accuracy.|
|Random Forest (Ensemble)|Achieved perfect performance across all metrics. Ensemble learning improved stability and reduced variance.|
|XGBoost (Ensemble)|Delivered top performance with strong generalization ability using gradient boosting.|

