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