"""
IoT Predictive Maintenance — REST API
Serves the trained Random Forest model for real-time sensor inference.
Author: Mannat Singh
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ── Load Model ─────────────────────────────────────────────────────────
try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    raise RuntimeError("Model not found. Run supervised_detection.py first.")

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IoT Predictive Maintenance API",
    description=(
        "Real-time machine failure prediction from industrial IoT sensor readings. "
        "Trained on the AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository). "
        "Model: Random Forest with class balancing | F1: 85.27% | ROC-AUC: 97.14%"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema ─────────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    air_temperature_K: float     = Field(..., example=298.1,  description="Air temperature in Kelvin")
    process_temperature_K: float = Field(..., example=308.6,  description="Process temperature in Kelvin")
    rotational_speed_rpm: float  = Field(..., example=1551.0, description="Rotational speed (rpm)")
    torque_Nm: float             = Field(..., example=42.8,   description="Torque in Newton-metres")
    tool_wear_min: float         = Field(..., example=0.0,    description="Tool wear in minutes")

class PredictionResponse(BaseModel):
    prediction: str
    failure_probability: float
    risk_level: str
    confidence: str
    features_used: dict
    model_info: dict

# ── Feature Engineering ────────────────────────────────────────────────
def engineer_features(data: SensorReading) -> np.ndarray:
    """Replicates the training pipeline's feature engineering."""
    air_temp  = data.air_temperature_K
    proc_temp = data.process_temperature_K
    rpm       = data.rotational_speed_rpm
    torque    = data.torque_Nm
    wear      = data.tool_wear_min

    temp_diff         = proc_temp - air_temp
    power             = rpm * torque
    wear_torque       = wear * torque
    speed_torque_ratio = rpm / (torque + 1)

    features = np.array([[
        air_temp, proc_temp, rpm, torque, wear,
        temp_diff, power, wear_torque, speed_torque_ratio
    ]])
    return features

# ── Routes ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "IoT Predictive Maintenance API",
        "status": "operational",
        "model": "Random Forest (class_weight='balanced')",
        "metrics": {
            "accuracy": "99.05%",
            "f1_score": "85.27%",
            "roc_auc": "97.14%",
            "cv_f1": "83.96% (±2.27%)"
        },
        "dataset": "AI4I 2020 Predictive Maintenance — UCI ML Repository",
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    """
    Predict machine failure probability from real-time IoT sensor readings.
    Returns failure probability, risk level, and engineered feature values.
    """
    try:
        raw_features = engineer_features(reading)
        scaled       = scaler.transform(raw_features)
        prediction   = model.predict(scaled)[0]
        probability  = model.predict_proba(scaled)[0][1]

        # Risk stratification
        if probability < 0.3:
            risk = "LOW"
            confidence = "High confidence — normal operation"
        elif probability < 0.6:
            risk = "MEDIUM"
            confidence = "Monitor closely — early warning signs detected"
        else:
            risk = "HIGH"
            confidence = "Immediate inspection recommended"

        air_temp  = reading.air_temperature_K
        proc_temp = reading.process_temperature_K
        rpm       = reading.rotational_speed_rpm
        torque    = reading.torque_Nm
        wear      = reading.tool_wear_min

        return PredictionResponse(
            prediction="FAILURE" if prediction == 1 else "NORMAL",
            failure_probability=round(float(probability), 4),
            risk_level=risk,
            confidence=confidence,
            features_used={
                "air_temperature_K":     air_temp,
                "process_temperature_K": proc_temp,
                "rotational_speed_rpm":  rpm,
                "torque_Nm":             torque,
                "tool_wear_min":         wear,
                "temp_diff":             round(proc_temp - air_temp, 2),
                "power":                 round(rpm * torque, 2),
                "wear_torque":           round(wear * torque, 2),
                "speed_torque_ratio":    round(rpm / (torque + 1), 4),
            },
            model_info={
                "model": "Random Forest",
                "n_estimators": 300,
                "class_weight": "balanced",
                "f1_score": "85.27%",
                "roc_auc": "97.14%",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
