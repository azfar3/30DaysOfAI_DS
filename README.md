# 30 Days of AI Engineering & Data Science

This is my first repository! I'm starting my 30-day challenge today.

# Day 1: Data Wrangling Mastery

## Dataset
googleplaystore.csv

## Objectives
1.  Load and inspect the raw data.
2.  Handle missing values.
3.  Correct data types.
4.  Perform basic feature engineering.

## Steps Taken
1.  **Missing Values:** Filled missing 'Age' values with the median age. Filled missing 'Cabin' values with 'Unknown'.
2.  **Data Types:** Converted 'Sex' to a categorical data type.
3.  **New Features:** Created a 'FamilySize' feature by combining 'SibSp' and 'Parch'.

## Key Learnings
- The `df.isnull().sum()` command is essential for a quick missing data overview.
- Using `inplace=True` saves time by modifying the DataFrame directly.

# Day 2: Data Visualization & Storytelling

## Objectives
Use visualizations to explore the cleaned dataset from Day 1 and uncover key stories.

## Key Visualizations Created
1.  **Distribution of App Rating**    
2.  **App Categories**     
3.  **Rating vs. Review**     
4.  **Free vs. Paid** 
5. **Content Rating**
6. **Comparison Original vs. Normalized**
    
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