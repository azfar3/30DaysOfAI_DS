# Day 4: Classification - Predicting App Type (Free vs. Paid)

## Objective
To predict the category (`Type`) of an app based on its other features.

## Steps Taken
1.  **Data Preparation:** Encoded the target variable (`Type` -> 0/1). Used One-Hot Encoding for categorical features and StandardScaler for numerical features.
2.  **Train-Test Split:** Split the data into 70% training and 30% testing sets, using stratification.
3.  **Model Training:** Trained two models:
    - Logistic Regression
    - Random Forest Classifier
4.  **Evaluation:** Evaluated models using Accuracy, Confusion Matrix, and Precision/Recall metrics.

## Results
- **Best Model:** Random Forest Classifier
- **Accuracy:** 100%
- **Key Metric - Precision:** 1.00.
- **Key Metric - Recall:** 1.00.

## Key Insight
The most important feature for predicting if an app is paid was `Price`, which makes sense. The second most important was `installs`