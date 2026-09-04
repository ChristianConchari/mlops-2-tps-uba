"""ML API — Mini-TP 1 (Sesión 1).

Sirve el modelo juguete (Iris / RandomForest) por REST con FastAPI.

Endpoints:
  GET  /health         -> estado del servicio y si el modelo está cargado
  POST /v1/predict     -> predicción online, devuelve clase + versión del modelo

Contrato de entrada validado con Pydantic: las 4 features reales del modelo,
todas > 0 (medidas en cm). Entradas mal formadas -> HTTP 422.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"

app = FastAPI(title="Mini-TP 1 — ML API", version="1.0.0")

_artifact = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
_pipeline = _artifact["pipeline"] if _artifact else None
_metadata = _artifact["metadata"] if _artifact else {}


class PredictRequest(BaseModel):
    """Features reales del modelo Iris (todas en cm, deben ser > 0)."""

    sepal_length_cm: float = Field(..., gt=0, le=20, examples=[5.1])
    sepal_width_cm: float = Field(..., gt=0, le=20, examples=[3.5])
    petal_length_cm: float = Field(..., gt=0, le=20, examples=[1.4])
    petal_width_cm: float = Field(..., gt=0, le=20, examples=[0.2])

    model_config = {"extra": "forbid"}


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _pipeline is not None else "degraded",
        model_loaded=_pipeline is not None,
        model_version=_metadata.get("model_version"),
    )


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    features = np.array(
        [[
            req.sepal_length_cm,
            req.sepal_width_cm,
            req.petal_length_cm,
            req.petal_width_cm,
        ]]
    )
    pred = int(_pipeline.predict(features)[0])
    labels = _metadata.get("target_names", [])
    label = labels[pred] if pred < len(labels) else str(pred)
    return PredictResponse(
        prediction=pred,
        prediction_label=label,
        model_version=_metadata.get("model_version", "unknown"),
    )
