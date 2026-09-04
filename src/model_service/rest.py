"""API REST — Mini-TP 1 (Sesión 1), sobre el servicio compartido.

Endpoints:
  GET  /health              -> estado del servicio y versión del modelo
  POST /v1/predict          -> predicción online (clase + etiqueta + versión)
  GET  /v1/model            -> metadatos del modelo (name, version)
  GET  /v1/model/metrics    -> métricas del modelo (auc, accuracy, f1)

Los dos últimos existen para la comparación REST vs GraphQL del Mini-TP 2:
armar "la vista" del modelo por REST cuesta 2 llamadas y trae campos de más.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import model as svc

app = FastAPI(title="model-service — REST", version="1.0.0")


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
        status="ok" if svc.is_loaded() else "degraded",
        model_loaded=svc.is_loaded(),
        model_version=svc.METADATA.get("model_version"),
    )


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    pred, label = svc.predict(
        [
            req.sepal_length_cm,
            req.sepal_width_cm,
            req.petal_length_cm,
            req.petal_width_cm,
        ]
    )
    return PredictResponse(
        prediction=pred,
        prediction_label=label,
        model_version=svc.METADATA.get("model_version", "unknown"),
    )


@app.get("/v1/model")
def model_info() -> dict:
    """Metadata completa del modelo (el cliente filtra lo que necesita)."""
    return {
        "name": svc.METADATA.get("algorithm", "unknown"),
        "model_version": svc.METADATA.get("model_version", "unknown"),
        "dataset": svc.METADATA.get("dataset"),
        "features": svc.FEATURES,
        "target_names": svc.METADATA.get("target_names", []),
        "trained_at": svc.METADATA.get("trained_at"),
    }


@app.get("/v1/model/metrics")
def model_metrics() -> dict:
    return svc.metrics()
