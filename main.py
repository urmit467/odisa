# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Optional

app = FastAPI(title="Odisha Crop Yield Predictor")

# Load the training pipeline (preproc + model)
PIPE_PATH = "odisha_crop_pipeline.joblib"
try:
    pipeline = joblib.load(PIPE_PATH)
    # Check how many outputs the model has
    if hasattr(pipeline.named_steps['model'], 'estimators_'):
        n_outputs = len(pipeline.named_steps['model'].estimators_)
        print(f"Model loaded with {n_outputs} outputs")
    else:
        n_outputs = 1
        print(f"Model loaded (single output)")
except FileNotFoundError:
    print(f"Warning: Model file {PIPE_PATH} not found. Please train the model first.")
    pipeline = None
    n_outputs = 0

class PredictionRequest(BaseModel):
    district: str
    crop: str
    season: str
    sowing_date: str  # "YYYY-MM-DD"

class PredictionResponse(BaseModel):
    predicted_environmental_conditions: dict
    predicted_soil_conditions: dict
    predicted_fertilizer_recommendation: dict
    predicted_yield_kg_per_ha: float
    predicted_harvest_days: float

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # Convert sowing_date to day of year
    try:
        sow_dt = pd.to_datetime(req.sowing_date)
        sowing_doy = int(sow_dt.dayofyear)
    except:
        raise HTTPException(status_code=400, detail="Invalid sowing_date format. Use YYYY-MM-DD.")
    
    # Build input data with only the provided features
    data = {
        "district": req.district,
        "crop": req.crop,
        "season": req.season,
        "sowing_doy": sowing_doy
    }
    
    X = pd.DataFrame([data])
    
    try:
        # Get predictions for all target features
        preds = pipeline.predict(X)
        
        # Extract predictions for each category
        environmental_conditions = {
            "season_total_rainfall_mm": round(float(preds[0, 0]), 1),
            "season_avg_temp_c": round(float(preds[0, 1]), 1),
            "season_avg_humidity": round(float(preds[0, 2]), 1)
        }
        
        soil_conditions = {
            "soil_pH": round(float(preds[0, 3]), 1),
            "soil_N_kg_ha": round(float(preds[0, 4]), 1),
            "soil_P_kg_ha": round(float(preds[0, 5]), 1),
            "soil_K_kg_ha": round(float(preds[0, 6]), 1),
            "organic_carbon_pct": round(float(preds[0, 7]), 2),
            "soil_moisture_pct": round(float(preds[0, 8]), 1)
        }
        
        fertilizer_recommendation = {
            "N": round(float(preds[0, 9]), 1),
            "P": round(float(preds[0, 10]), 1),
            "K": round(float(preds[0, 11]), 1)
        }
        
        yield_kg_per_ha = round(float(preds[0, 12]), 2)
        harvest_days = round(float(preds[0, 13]), 1)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return PredictionResponse(
        predicted_environmental_conditions=environmental_conditions,
        predicted_soil_conditions=soil_conditions,
        predicted_fertilizer_recommendation=fertilizer_recommendation,
        predicted_yield_kg_per_ha=yield_kg_per_ha,
        predicted_harvest_days=harvest_days
    )

@app.get("/")
def read_root():
    return {"message": "Odisha Crop Yield Prediction API"}

@app.get("/model-info")
def model_info():
    """Check what the model is configured to predict"""
    if pipeline is None:
        return {"error": "Model not loaded"}
    
    info = {
        "outputs": n_outputs,
        "has_fertilizer_predictions": n_outputs >= 14  # 14 outputs total
    }
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)