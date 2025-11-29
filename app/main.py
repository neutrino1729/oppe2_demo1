"""
FastAPI Application for Fraud Detection
Serves predictions using trained model
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field
import logging
import json
import sys

# Setup structured JSON logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record)
        }
        return json.dumps(log_record)

logger = logging.getLogger("fraud-detection-api")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using ML",
    version="1.0.0"
)

# Global state
model = None
feature_names = None
app_state = {"is_ready": False, "is_alive": True}

# API Models
class TransactionFeatures(BaseModel):
    """Transaction features for prediction"""
    Time: float = Field(..., description="Time elapsed since first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
                "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
                "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
                "V13": -0.9914, "V14": -0.3112, "V15": 1.4681, "V16": -0.4704,
                "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
                "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
                "V25": 0.1286, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
                "Amount": 149.62
            }
        }

class PredictionResponse(BaseModel):
    is_fraud: int = Field(..., description="Fraud prediction: 0=Normal, 1=Fraud")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
                "is_fraud": 0,
                "fraud_probability": 0.023
            }
        }

# Startup event
@app.on_event("startup")
async def startup_event():
    global model, feature_names
    logger.info({"event": "startup_begin"})
    
    try:
        # Load model
        model_path = "artifacts/model.pkl"
        logger.info({"event": "loading_model", "path": model_path})
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = joblib.load(model_path)
        feature_names = model.feature_names_in_.tolist()
        
        app_state["is_ready"] = True
        logger.info({"event": "startup_success", "features": len(feature_names)})
        
    except Exception as e:
        app_state["is_ready"] = False
        logger.error({"event": "startup_failure", "error": str(e)})

# Health probes
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check"""
    return {"status": "healthy"}

@app.get("/live", tags=["Health"])
async def liveness_probe():
    """Kubernetes liveness probe"""
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.get("/ready", tags=["Health"])
async def readiness_probe():
    """Kubernetes readiness probe"""
    if app_state["is_ready"]:
        return {"status": "ready", "model_loaded": model is not None}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: TransactionFeatures):
    """
    Predict fraud for a given transaction
    
    Returns fraud prediction and probability
    """
    if not app_state["is_ready"]:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Model not loaded."
        )
    
    try:
        # Convert to DataFrame
        features_dict = features.dict()
        features_df = pd.DataFrame([features_dict])
        
        # Ensure correct feature order (excluding Time if not in model)
        features_df = features_df[feature_names]
        
        # Make prediction
        prediction_proba = model.predict_proba(features_df)
        fraud_probability = float(prediction_proba[0][1])
        is_fraud = 1 if fraud_probability > 0.5 else 0
        
        logger.info({
            "event": "prediction_success",
            "is_fraud": is_fraud,
            "probability": fraud_probability
        })
        
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=fraud_probability
        )
        
    except Exception as e:
        logger.error({"event": "prediction_error", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "predict": "/predict",
            "docs": "/docs"
        }
    }
