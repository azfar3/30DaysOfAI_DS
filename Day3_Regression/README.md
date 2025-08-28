# Day 3: Regression - Predicting App Ratings

## Objective
To predict an app's continuous `Rating` based on features like Reviews, Size, Installs, etc.

## Steps Taken
1.  **Data Preparation:** Selected features ('Reviews', 'Size', 'Installs', 'Price', 'Rating_normalized', 'Category_Encoded'). Separated the target variable (`Rating`).
2.  **Train-Test Split:** Split the data into 70% training and 30% testing sets.
3.  **Model Training:** Trained two models:
    - Linear Regression
    - Random Forest Regressor
4.  **Evaluation:** Evaluated models using MAE, RMSE, and R2 Score.

## Results
- **Best Model:** Linear Regression
- **R2 Score:** 1.0
- **RMSE:** 7.4344564992232e-15
This means our model explains about 100% of the variance in app ratings.

## Key Insight
The most important feature for predicting an app's rating was `Rating_normalized`, followed by `Category_Encoded`.