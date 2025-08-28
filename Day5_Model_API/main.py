# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# Load your saved model, label encoder, and scaler
model = joblib.load("../Day4_Classification/models/random_forest_model.pkl")

# Create FastAPI app
app = FastAPI(
    title="App Type Prediction API",
    description="API to predict if an app is Free or Paid",
    version="1.0",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Define the input data structure using Pydantic
class AppFeatures(BaseModel):
    Reviews: float
    Size: float
    Installs: int
    Price: float
    Days_Since_Update: int
    Rating: float
    Rating_normalized: float


# Home page with HTML interface
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is working!"}


# Prediction endpoint
@app.post("/predict")
def predict(features: AppFeatures):
    """
    Predict if an app is Free (0) or Paid (1) based on its features
    """
    try:
        print("Received features:", features.dict())

        # Convert input features to dictionary then to DataFrame
        input_data = features.dict()
        if "Rating" in input_data and "Rating_normalized" not in input_data:
            input_data["Rating_normalized"] = input_data["Rating"]
            del input_data["Rating"]

        input_df = pd.DataFrame([input_data])

        print("Input DataFrame shape:", input_df.shape)
        print("Input DataFrame columns:", input_df.columns.tolist())
        print("Input data types:", input_df.dtypes)

        # Make prediction
        prediction = model.predict(input_df)

        # Get prediction probability
        probability = model.predict_proba(input_df)

        print("Prediction:", prediction[0])
        print("Probability:", probability[0])

        # Return the prediction
        return {
            "prediction": int(prediction[0]),
            "prediction_label": "Paid" if prediction[0] == 1 else "Free",
            "probability": float(probability[0][1]),  # Probability of being Paid
            "confidence": "High" if max(probability[0]) > 0.7 else "Medium",
        }

    except Exception as e:
        print("Error in prediction:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Additional endpoint to get model info
@app.get("/model-info")
def model_info():
    return {
        "model_type": "Random Forest Classifier",
        "features_used": [
            "Reviews",
            "Size",
            "Installs",
            "Price",
            "Days_Since_Update",
            "Rating",
            "Rating_normalized",
        ],
        "version": "1.0",
    }


# Endpoint for debugging
@app.post("/debug-predict")
async def debug_predict(request: Request):
    """
    Debug endpoint to see what data is actually received
    """
    try:
        # Get the raw JSON data
        data = await request.json()
        print("Received data:", data)

        # Try to convert to your model
        try:
            features = AppFeatures(**data)
            print("Data validated successfully:", features.dict())
            return {"status": "valid", "data": features.dict()}
        except Exception as e:
            print("Validation error:", str(e))
            return {"status": "invalid", "error": str(e)}

    except Exception as e:
        print("Request error:", str(e))
        return {"status": "error", "message": str(e)}
