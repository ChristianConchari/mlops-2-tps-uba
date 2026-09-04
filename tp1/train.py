"""Entrena el modelo juguete y genera el artefacto versionado.

Modelo: RandomForestClassifier sobre el dataset Iris de scikit-learn.
Salida: model/model.joblib con el pipeline entrenado + metadata (features, clases, version).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "1.0.0"
MODEL_DIR = Path(__file__).parent / "model"

# Nombres "limpios" de las features (orden = orden de columnas de load_iris).
FEATURE_NAMES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]


def main() -> None:
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Accuracy en test: {acc:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestClassifier",
        "dataset": "sklearn.datasets.load_iris",
        "features": FEATURE_NAMES,
        "target_names": data.target_names.tolist(),
        "test_accuracy": round(float(acc), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Artefacto guardado en {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
