"""Carga del modelo y sus metadatos.

Módulo compartido: la API REST (Mini-TP 1), la API GraphQL (Mini-TP 2), el
servidor gRPC (Mini-TP 3) y el pipeline de streaming (Mini-TP 4) leen el
modelo desde acá.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Raíz del repo -> model/model.joblib
MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "model.joblib"

# Mini-TP 6: con MODEL_SOURCE=lake el artefacto se descarga del Data Lake
# (models/<MODEL_VERSION>/) en lugar de leerse de disco.
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "disk")


def _load() -> dict[str, Any] | None:
    if MODEL_SOURCE == "lake":
        from .datalake import load_model

        return load_model(os.environ.get("MODEL_VERSION", "v1"))
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


_artifact: dict[str, Any] | None = _load()
_pipeline = _artifact["pipeline"] if _artifact else None
METADATA: dict[str, Any] = _artifact["metadata"] if _artifact else {}

FEATURES = METADATA.get(
    "features",
    ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"],
)


def is_loaded() -> bool:
    return _pipeline is not None


def predict(features: list[float]) -> tuple[int, str]:
    """Devuelve (clase, etiqueta) para un vector de 4 features."""
    if _pipeline is None:
        raise RuntimeError("El modelo no está cargado; corré train.py primero.")
    pred = int(_pipeline.predict(np.array([features]))[0])
    labels = METADATA.get("target_names", [])
    label = labels[pred] if pred < len(labels) else str(pred)
    return pred, label


def predict_batch(features: list[list[float]]) -> list[int]:
    """Puntúa varios vectores de una sola vez (batch), sin ida y vuelta por evento."""
    if _pipeline is None:
        raise RuntimeError("El modelo no está cargado; corré train.py primero.")
    return [int(p) for p in _pipeline.predict(np.array(features))]


def metrics() -> dict[str, float]:
    """Métricas del modelo (de la metadata del artefacto).

    El artefacto del TP1 sólo guarda `test_accuracy`; exponemos también
    `auc` y `f1` como 0.0 si no están, para que el contrato sea estable.
    """
    return {
        "accuracy": float(METADATA.get("test_accuracy", 0.0)),
        "auc": float(METADATA.get("test_auc", 0.0)),
        "f1": float(METADATA.get("test_f1", 0.0)),
    }
