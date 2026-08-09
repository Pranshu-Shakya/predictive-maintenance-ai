import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
try:
    # When running from the `backend` directory (recommended)
    from ml.fault_predictor import predict_fault
except Exception:
    # When running from the repo root as a package
    from backend.ml.fault_predictor import predict_fault
from schemas import SensorData

from rag.generate_response import (
    generate_troubleshooting_response
)


# ==========================================
# FastAPI Application
# ==========================================

app = FastAPI(
    title="AI-Powered Predictive Maintenance API",
    description="ML + RAG + Gemini based predictive maintenance system",
    version="1.0.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Load ML Model
# ==========================================

from pathlib import Path

# Load saved model from expected locations with a clear error message if missing
possible_model_paths = [
    Path(__file__).resolve().parent / "ml" / "model.pkl",
    Path(__file__).resolve().parent / "model.pkl",
]

for model_path in possible_model_paths:
    if model_path.exists():
        saved_model = joblib.load(str(model_path))
        break
else:
    raise FileNotFoundError(
        "Model file not found. Checked: "
        + ", ".join(str(p) for p in possible_model_paths)
        + ". Run the training script or place model.pkl in one of those locations."
    )

model = saved_model["model"]

FEATURES = saved_model["features"]


# ==========================================
# Root Endpoint
# ==========================================

@app.get("/")
def root():

    return {
        "message": "Predictive Maintenance API is running"
    }


# ==========================================
# Health Endpoint
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "ml_model": "loaded" if model else "not_loaded",
        "rag": "available",
        "llm": "Gemini API"
    }


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
def predict(sensor_data: SensorData):

    input_data = pd.DataFrame(
        [[
            sensor_data.Temperature,
            sensor_data.Vibration,
            sensor_data.Pressure,
            sensor_data.RPM,
            sensor_data.Operating_Hours,
            sensor_data.Flow_Rate
        ]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(
        input_data
    )[0]

    failure_probability = float(
        probabilities[1]
    )

    normal_probability = float(
        probabilities[0]
    )

    return {
        "prediction": int(prediction),
        "status": (
            "FAILURE"
            if prediction == 1
            else "NORMAL"
        ),
        "failure_probability": round(
            failure_probability,
            4
        ),
        "normal_probability": round(
            normal_probability,
            4
        )
    }


# ==========================================
# Complete Diagnosis Endpoint
# ==========================================

@app.post("/diagnose")
def diagnose(sensor_data: SensorData):

    input_data = pd.DataFrame(
        [[
            sensor_data.Temperature,
            sensor_data.Vibration,
            sensor_data.Pressure,
            sensor_data.RPM,
            sensor_data.Operating_Hours,
            sensor_data.Flow_Rate
        ]],
        columns=FEATURES
    )

    # --------------------------------------
    # ML Prediction
    # --------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(
        input_data
    )[0]

    failure_probability = float(
        probabilities[1]
    )


    # --------------------------------------
    # Determine predicted fault
    # --------------------------------------

    predicted_fault, fault_probabilities = predict_fault(
        sensor_data.model_dump()
    )

    # --------------------------------------
    # Risk Classification
    # --------------------------------------

    if failure_probability >= 0.70:

        machine_status = "HIGH_RISK"

    elif failure_probability >= 0.40:

        machine_status = "MEDIUM_RISK"

    else:

        machine_status = "LOW_RISK"

    # --------------------------------------
    # RAG + Gemini
    # --------------------------------------

    try:

        ai_diagnosis, documents = (
            generate_troubleshooting_response(
                sensor_data.model_dump(),
                predicted_fault,
                failure_probability,
                fault_probabilities
            )
        )

    except Exception as error:

        print("RAG/Gemini error:", error)

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate AI "
                "troubleshooting report."
            )
        )


    # --------------------------------------
    # Retrieved sources
    # --------------------------------------

    sources = list(
        dict.fromkeys(
            [
                doc.metadata.get("source")
                for doc in documents
            ]
        )
    )


    # --------------------------------------
    # Final response
    # --------------------------------------

    return {

        "machine_status": machine_status,

        "failure_probability": round(
            failure_probability,
            4
        ),

        "predicted_fault": predicted_fault,

        "fault_probabilities": {
            fault: round(probability, 4)
            for fault, probability
            in fault_probabilities.items()
        },

        "ai_diagnosis": ai_diagnosis,

        "sources": sources
    }