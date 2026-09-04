"""Carga del modelo y sus metadatos.

Módulo compartido: tanto la API REST (Mini-TP 1) como la API GraphQL
(Mini-TP 2) leen el modelo desde acá.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Raíz del repo -> model/model.joblib
MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "model.joblib"

_artifact: dict[str, Any] | None = (
    joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
)
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
