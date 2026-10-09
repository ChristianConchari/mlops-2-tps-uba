"""Data Lake (MinIO/S3) — Mini-TP 6 (Sesión 6), sobre el servicio compartido.

Un bucket con zonas por prefijo:
  raw/       dato tal cual llega (Iris con los nombres originales de sklearn)
  curated/   dato limpio y listo para entrenar (nombres de features del contrato)
  models/    artefactos versionados: models/v1/model.joblib + metadata.json

El servicio **carga el modelo desde el lake** (ver `model.py` con
MODEL_SOURCE=lake) en lugar de leerlo de disco o hornearlo en la imagen.

Credenciales y endpoint por variables de entorno, así el mismo código corre
contra MinIO local o contra S3 en la nube cambiando sólo la configuración.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import boto3
import joblib
from botocore.client import Config
from botocore.exceptions import ClientError

ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
BUCKET = os.environ.get("LAKE_BUCKET", "mlops2-lake")
ZONES = ("raw", "curated", "models")

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def client():
    """Cliente S3 apuntando a MinIO (o a S3 si S3_ENDPOINT_URL está vacío)."""
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio123")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT or None,
        config=Config(signature_version="s3v4"),
    )


def create_zones(s3=None) -> list[str]:
    """Crea el bucket (si no existe) y un marcador por zona.

    S3 no tiene carpetas: las zonas son prefijos. El objeto `<zona>/.keep`
    sólo hace que la zona se vea vacía en la consola antes de tener datos.
    """
    s3 = s3 or client()
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)
    for zone in ZONES:
        s3.put_object(Bucket=BUCKET, Key=f"{zone}/.keep", Body=b"")
    return [f"s3://{BUCKET}/{z}/" for z in ZONES]


def upload_dataset(s3=None) -> dict[str, Any]:
    """Sube Iris a raw/ (como viene) y a curated/ (con los nombres del contrato)."""
    from sklearn.datasets import load_iris

    from .model import FEATURES

    s3 = s3 or client()
    df = load_iris(as_frame=True).frame  # columnas "sepal length (cm)", ..., "target"
    raw_key = "raw/iris/iris.csv"
    s3.put_object(Bucket=BUCKET, Key=raw_key, Body=df.to_csv(index=False).encode())

    curated = df.rename(columns=dict(zip(df.columns[:4], FEATURES))).drop_duplicates()
    curated_key = "curated/iris/iris.csv"
    s3.put_object(Bucket=BUCKET, Key=curated_key, Body=curated.to_csv(index=False).encode())
    return {"raw": raw_key, "curated": curated_key, "filas_raw": len(df), "filas_curated": len(curated)}


def upload_model(version: str = "v1", s3=None) -> list[str]:
    """Sube el artefacto local (model/model.joblib + metadata.json) a models/<version>/."""
    s3 = s3 or client()
    keys = []
    for name in ("model.joblib", "metadata.json"):
        key = f"models/{version}/{name}"
        s3.upload_file(str(MODEL_DIR / name), BUCKET, key)
        keys.append(key)
    return keys


def list_versions(s3=None) -> list[str]:
    """Versiones de modelo publicadas en el lake (models/v1, models/v2, ...)."""
    s3 = s3 or client()
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="models/", Delimiter="/")
    return sorted(p["Prefix"].split("/")[1] for p in resp.get("CommonPrefixes", []))


def load_model(version: str = "v1", s3=None) -> dict[str, Any]:
    """Descarga models/<version>/model.joblib a memoria y lo deserializa.

    Devuelve el mismo dict que `train.py` guarda: {"pipeline", "metadata"}.
    No toca el disco: el artefacto va del lake a la RAM del servicio.
    """
    s3 = s3 or client()
    blob = s3.get_object(Bucket=BUCKET, Key=f"models/{version}/model.joblib")["Body"].read()
    return joblib.load(io.BytesIO(blob))


def read_metadata(version: str = "v1", s3=None) -> dict[str, Any]:
    s3 = s3 or client()
    body = s3.get_object(Bucket=BUCKET, Key=f"models/{version}/metadata.json")["Body"].read()
    return json.loads(body)


def list_objects(prefix: str = "", s3=None) -> list[tuple[str, int]]:
    """(key, bytes) de todo lo que hay bajo `prefix`."""
    s3 = s3 or client()
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [(o["Key"], o["Size"]) for o in resp.get("Contents", [])]
