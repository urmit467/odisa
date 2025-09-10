# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Optional
from simple_fertilizer_recommender import recommend_fertilizer

app = FastAPI(title="Odisha Crop Yield Predictor")

# Load the training pipeline (preproc + model)
PIPE_PATH = "odisha_crop_pipeline.joblib"
try:
    pipeline = joblib.load(PIPE_PATH)
except FileNotFoundError:
    print(f"Warning: Model file {PIPE_PATH} not found. Please train the model first.")
    pipeline = None

class PredictionRequest(BaseModel):
    district: str
    crop: str
    season: str
    sowing_date: Optional[str] = None  # "YYYY-MM-DD"
    season_total_rainfall_mm: Optional[float] = None
    season_avg_temp_c: Optional[float] = None
    season_avg_humidity: Optional[float] = None
    soil_pH: Optional[float] = None
    soil_N_kg_ha: Optional[float] = None
    soil_P_kg_ha: Optional[float] = None
    soil_K_kg_ha: Optional[float] = None
    organic_carbon_pct: Optional[float] = None
    soil_moisture_pct: Optional[float] = None

class PredictionResponse(BaseModel):
    predicted_yield_kg_per_ha: float
    predicted_harvest_days: float
    fertilizer_recommendation_kg_per_ha: dict
    estimated_soil_conditions: dict

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # build one-row DataFrame
    data = {
        "district": req.district,
        "crop": req.crop,
        "season": req.season,
        "sowing_doy": -1,
        "season_total_rainfall_mm": req.season_total_rainfall_mm,
        "season_avg_temp_c": req.season_avg_temp_c,
        "season_avg_humidity": req.season_avg_humidity,
        "soil_pH": req.soil_pH,
        "soil_N_kg_ha": req.soil_N_kg_ha,
        "soil_P_kg_ha": req.soil_P_kg_ha,
        "soil_K_kg_ha": req.soil_K_kg_ha,
        "organic_carbon_pct": req.organic_carbon_pct,
        "soil_moisture_pct": req.soil_moisture_pct
    }
    
    # if sowing_date provided, convert to doy
    if req.sowing_date:
        try:
            sow_dt = pd.to_datetime(req.sowing_date)
            data["sowing_doy"] = int(sow_dt.dayofyear)
        except:
            pass

    X = pd.DataFrame([data])
    
    try:
        preds = pipeline.predict(X)
        pred_yield = float(preds[0, 0])
        pred_harvest_days = float(preds[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # fertilizer recommendation (use soil test if provided)
    fert = recommend_fertilizer(req.crop, req.soil_N_kg_ha, req.soil_P_kg_ha, req.soil_K_kg_ha)

    estimated_soil = {
        "soil_pH": req.soil_pH if req.soil_pH is not None else "unknown",
        "soil_N_kg_ha": req.soil_N_kg_ha if req.soil_N_kg_ha is not None else "unknown",
        "soil_P_kg_ha": req.soil_P_kg_ha if req.soil_P_kg_ha is not None else "unknown",
        "soil_K_kg_ha": req.soil_K_kg_ha if req.soil_K_kg_ha is not None else "unknown",
        "organic_carbon_pct": req.organic_carbon_pct if req.organic_carbon_pct is not None else "unknown"
    }

    return PredictionResponse(
        predicted_yield_kg_per_ha=round(pred_yield, 2),
        predicted_harvest_days=round(pred_harvest_days, 1),
        fertilizer_recommendation_kg_per_ha=fert,
        estimated_soil_conditions=estimated_soil
    )

@app.get("/")
def read_root():
    return {"message": "Odisha Crop Yield Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)