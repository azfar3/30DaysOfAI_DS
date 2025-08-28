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

# Day 5: ML Engineering - Model API with FastAPI

## Objective
Create a web API and beautiful frontend interface to serve a machine learning model that predicts whether a Google Play Store app is Free or Paid.

## What Was Built
- FastAPI Backend: RESTful API with prediction endpoints
- Beautiful Frontend: Modern HTML/CSS/JS interface with Bootstrap
- Interactive Features: Real-time predictions with loading states and animations
- Production-Ready: Error handling, input validation, and proper API design

## How to Run
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Run the server: `uvicorn main:app --reload`
3. Web Interface `http://127.0.0.1:8000` 

## Make a Prediction
1. Open the web interface
2. Fill in the app features:
- Reviews (e.g., 1000000)
- Size in MB (e.g., 25.0)
- Installs (e.g., 10000000)
- Price in $ (e.g., 0.99)
- Days since update (e.g., 30)
- Rating (0-5 scale, e.g., 4.5)
- Normalized Rating (0-1 scale, e.g., 0.9)
3. Click "Predict App Type"
4. View the results with confidence indicators

## API Endpoints
- `GET /` - Web interface
- `POST /predict` - Prediction endpoint
- `GET /health` - Health check
- `GET /model-info` - Model information
- `POST /debug-predict` - Debug endpoint to validate input data
- `GET /docs` - Automatic API documentation

## Project Structure
```
Day5_Model_API/
├── main.py              # FastAPI application
├── requirements.txt     # Dependencies
├── templates/
│   └── index.html      # Web interface
└── static/             # Static files (CSS, JS, images)
```