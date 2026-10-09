"""Seguridad del Data Lake con SAIF — Mini-TP 7 (Sesión 7).

Endurece el lake del Mini-TP 6 contra **exfiltración** y **manipulación** de
artefactos:

  integridad   el modelo se publica con su SHA-256 y el servicio lo verifica
               antes de deserializarlo (un joblib alterado ejecuta código).
  enlaces      se comparte el modelo con una URL prefirmada que expira, nunca
               abriendo el bucket.
  auditoría    cada get/put pasa por `AuditedLake`, que registra quién, qué y
               cuándo, y persiste el registro en la zona `audit/` del lake.
  mínimo       usuarios de MinIO por rol con políticas IAM por prefijo: el
  privilegio   ingestor sólo escribe raw/, el serving sólo lee curated/ y models/.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import boto3
import joblib
from botocore.client import Config
from minio import MinioAdmin
from minio.credentials import StaticProvider

from . import datalake as dl


class IntegrityCheckError(Exception):
    """El artefacto del lake no coincide con su checksum publicado."""


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# --- auditoría ----------------------------------------------------------------

@dataclass
class AuditedLake:
    """Cliente S3 que deja registro de cada operación (gobernanza)."""

    actor: str
    s3: Any = None
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.s3 = self.s3 or dl.client()

    def _log(self, op: str, key: str, ok: bool, detail: str = "") -> None:
        self.records.append({
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "actor": self.actor, "op": op, "key": key, "ok": ok, "detail": detail,
        })

    def put(self, key: str, body: bytes, **extra) -> None:
        try:
            self.s3.put_object(Bucket=dl.BUCKET, Key=key, Body=body, **extra)
        except Exception as e:
            self._log("put", key, False, type(e).__name__)
            raise
        self._log("put", key, True, f"{len(body)} bytes")

    def get(self, key: str) -> bytes:
        try:
            body = self.s3.get_object(Bucket=dl.BUCKET, Key=key)["Body"].read()
        except Exception as e:
            self._log("get", key, False, type(e).__name__)
            raise
        self._log("get", key, True, f"{len(body)} bytes")
        return body

    def persist(self) -> str:
        """Guarda el registro como JSONL en audit/<actor>/<timestamp>.jsonl."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = f"audit/{self.actor}/{ts}.jsonl"
        body = "\n".join(json.dumps(r) for r in self.records).encode()
        self.s3.put_object(Bucket=dl.BUCKET, Key=key, Body=body)
        return key


# --- integridad ---------------------------------------------------------------

def publish_model(lake: AuditedLake, version: str = "v1") -> str:
    """Sube model.joblib + model.sha256 a models/<version>/ y devuelve el hash."""
    blob = (dl.MODEL_DIR / "model.joblib").read_bytes()
    digest = sha256(blob)
    lake.put(f"models/{version}/model.joblib", blob, Metadata={"sha256": digest})
    lake.put(f"models/{version}/model.sha256", digest.encode())
    return digest


def load_verified_model(lake: AuditedLake, version: str = "v1") -> dict[str, Any]:
    """Descarga el artefacto y lo deserializa **sólo** si el SHA-256 coincide."""
    blob = lake.get(f"models/{version}/model.joblib")
    expected = lake.get(f"models/{version}/model.sha256").decode().strip()
    if sha256(blob) != expected:
        lake._log("verify", f"models/{version}/model.joblib", False, "checksum distinto")
        raise IntegrityCheckError(f"models/{version}: sha256 {sha256(blob)[:12]}… != {expected[:12]}…")
    lake._log("verify", f"models/{version}/model.joblib", True, expected[:12])
    return joblib.load(io.BytesIO(blob))


# --- URLs prefirmadas ---------------------------------------------------------

def presigned_url(lake: AuditedLake, key: str, expires_in: int = 60) -> str:
    url = lake.s3.generate_presigned_url(
        "get_object", Params={"Bucket": dl.BUCKET, "Key": key}, ExpiresIn=expires_in
    )
    lake._log("presign", key, True, f"expira en {expires_in}s")
    return url


# --- mínimo privilegio por zona -----------------------------------------------

def _policy(read: list[str], write: list[str]) -> dict[str, Any]:
    arn = lambda p: f"arn:aws:s3:::{dl.BUCKET}/{p}*"  # noqa: E731
    stmts = []
    if read:
        stmts.append({"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": [arn(p) for p in read]})
    if write:
        stmts.append({"Effect": "Allow", "Action": ["s3:PutObject"], "Resource": [arn(p) for p in write]})
    return {"Version": "2012-10-17", "Statement": stmts}


# rol -> (secret, prefijos de lectura, prefijos de escritura)
ROLES: dict[str, tuple[str, list[str], list[str]]] = {
    "ingestor": ("ingestor-secret-123", [], ["raw/"]),
    "etl": ("etl-secret-12345", ["raw/"], ["curated/"]),
    "training": ("train-secret-1234", ["curated/"], ["models/"]),
    "serving": ("serving-secret-123", ["curated/", "models/"], ["audit/serving/"]),
}


def policies() -> dict[str, dict[str, Any]]:
    return {role: _policy(read, write) for role, (_, read, write) in ROLES.items()}


def create_roles() -> dict[str, dict[str, Any]]:
    """Crea en MinIO un usuario y una política por rol (idempotente)."""
    ep = urlparse(dl.ENDPOINT)
    admin = MinioAdmin(
        endpoint=ep.netloc,
        credentials=StaticProvider(
            os.environ.get("AWS_ACCESS_KEY_ID", "minio"),
            os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123"),
        ),
        secure=ep.scheme == "https",
    )
    pols = policies()
    for role, (secret, _, _) in ROLES.items():
        admin.user_add(role, secret)
        admin.policy_add(f"lake-{role}", policy=pols[role])
        try:
            admin.attach_policy([f"lake-{role}"], user=role)
        except Exception as e:  # ya estaba asociada
            if "already" not in str(e).lower():
                raise
    return pols


def role_client(role: str):
    """Cliente S3 con las credenciales del rol (no las de root)."""
    secret = ROLES[role][0]
    return boto3.client(
        "s3", endpoint_url=dl.ENDPOINT, aws_access_key_id=role,
        aws_secret_access_key=secret, region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )
