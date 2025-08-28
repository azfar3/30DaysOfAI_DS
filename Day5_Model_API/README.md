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