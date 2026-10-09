"""Seguridad del servicio gRPC con SAIF — Mini-TP 7 (Sesión 7), segunda capa.

Endurece el servidor del Mini-TP 3 (mismo `ScoringServicer`, mismo modelo)
contra acceso no autenticado, *man-in-the-middle* y abuso por mensajes enormes:

  autenticación  `AuthInterceptor` exige `authorization: Bearer <JWT>` en la
                 metadata y verifica firma (HS256) y vencimiento con
                 `jwt.decode`; si falla corta con UNAUTHENTICATED antes de
                 tocar el modelo.
  autorización   el claim `role` del JWT define qué RPCs puede llamar
                 (PERMISSION_DENIED si no alcanza).
  canal          TLS o mTLS con una CA propia (`generate_certs`), en lugar de
                 `add_insecure_port`.
  tamaño         `max_receive_message_length` limita el payload que acepta.
  auditoría      cada llamada queda registrada (actor, RPC, código, latencia).
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
import socket
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc
import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from . import scoring_pb2_grpc
from .grpc_server import ScoringServicer

PREDICT = "/scoring.Scoring/Predict"
PREDICT_BATCH = "/scoring.Scoring/PredictBatch"

# rol (claim `role` del JWT) -> RPCs permitidas
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "predictor": {PREDICT},
    "batch": {PREDICT, PREDICT_BATCH},
}

# En producción el secreto viene del entorno o de un secret manager, nunca del código.
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-secret-change-me-0123456789")
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "model-service"

MAX_MSG_BYTES = 64 * 1024  # un lote de ~1.800 vectores de 4 doubles


# --- auditoría ----------------------------------------------------------------

@dataclass
class AuditLog:
    records: list[dict[str, Any]] = field(default_factory=list)

    def log(self, actor: str, rpc: str, code: grpc.StatusCode, ms: float, detail: str = "") -> None:
        self.records.append({
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
            "actor": actor, "rpc": rpc.rsplit("/", 1)[-1], "code": code.name,
            "ms": round(ms, 2), "detail": detail,
        })


# --- JWT ----------------------------------------------------------------------

def issue_token(subject: str, role: str, ttl_seconds: int = 300, secret: str = JWT_SECRET) -> str:
    """Emite un JWT firmado con el actor (`sub`), su rol y un vencimiento corto."""
    now = dt.datetime.now(dt.timezone.utc)
    claims = {"sub": subject, "role": role, "iss": JWT_ISSUER,
              "iat": now, "exp": now + dt.timedelta(seconds=ttl_seconds)}
    return jwt.encode(claims, secret, algorithm=JWT_ALGORITHM)


def verify_token(token: str, secret: str = JWT_SECRET) -> dict[str, Any]:
    """Verifica firma, vencimiento y emisor. Lanza `jwt.InvalidTokenError` si algo falla."""
    return jwt.decode(
        token, secret, algorithms=[JWT_ALGORITHM], issuer=JWT_ISSUER,
        options={"require": ["sub", "role", "exp", "iss"]},
    )


# --- interceptor de autenticación + autorización ----------------------------

def _abort(code: grpc.StatusCode, detail: str):
    def handler(request, context):
        context.abort(code, detail)
    return grpc.unary_unary_rpc_method_handler(handler)


class AuthInterceptor(grpc.ServerInterceptor):
    """Verifica el JWT y el permiso de su rol antes de ejecutar la RPC."""

    def __init__(self, audit: AuditLog, secret: str = JWT_SECRET):
        self.audit = audit
        self.secret = secret

    def intercept_service(self, continuation, handler_call_details):
        rpc = handler_call_details.method
        meta = dict(handler_call_details.invocation_metadata or ())
        auth = meta.get("authorization", "")

        if not auth.startswith("Bearer "):
            self.audit.log("anónimo", rpc, grpc.StatusCode.UNAUTHENTICATED, 0.0, "sin token")
            return _abort(grpc.StatusCode.UNAUTHENTICATED, "falta el header authorization: Bearer <JWT>")
        try:
            claims = verify_token(auth.removeprefix("Bearer "), self.secret)
        except jwt.ExpiredSignatureError:
            self.audit.log("anónimo", rpc, grpc.StatusCode.UNAUTHENTICATED, 0.0, "token vencido")
            return _abort(grpc.StatusCode.UNAUTHENTICATED, "token vencido")
        except jwt.InvalidTokenError as e:
            self.audit.log("anónimo", rpc, grpc.StatusCode.UNAUTHENTICATED, 0.0, type(e).__name__)
            return _abort(grpc.StatusCode.UNAUTHENTICATED, f"token inválido ({type(e).__name__})")

        actor, role = claims["sub"], claims["role"]
        if rpc not in ROLE_PERMISSIONS.get(role, set()):
            self.audit.log(actor, rpc, grpc.StatusCode.PERMISSION_DENIED, 0.0, f"rol {role}")
            return _abort(grpc.StatusCode.PERMISSION_DENIED, f"el rol {role!r} no puede llamar {rpc}")

        return _audited(continuation(handler_call_details), self.audit, actor, rpc, role)


def _audited(handler, audit: AuditLog, actor: str, rpc: str, role: str):
    """Envuelve la RPC para registrar su resultado y su latencia real."""
    if handler is None:
        return None

    def _finish(t0, context, exc=None):
        code = context.code() or (grpc.StatusCode.INTERNAL if exc else grpc.StatusCode.OK)
        audit.log(actor, rpc, code, (time.perf_counter() - t0) * 1000, f"rol {role}")

    if handler.unary_unary:
        def unary(request, context):
            t0 = time.perf_counter()
            try:
                resp = handler.unary_unary(request, context)
            except Exception as e:
                _finish(t0, context, e)
                raise
            _finish(t0, context)
            return resp
        return grpc.unary_unary_rpc_method_handler(
            unary, handler.request_deserializer, handler.response_serializer)

    if handler.unary_stream:
        def stream(request, context):
            t0 = time.perf_counter()
            try:
                yield from handler.unary_stream(request, context)
            except Exception as e:
                _finish(t0, context, e)
                raise
            _finish(t0, context)
        return grpc.unary_stream_rpc_method_handler(
            stream, handler.request_deserializer, handler.response_serializer)

    return handler


# --- certificados (CA propia, servidor y cliente) ----------------------------

def _cert(subject: str, pub, issuer_name, issuer_key, *, ca=False, server=False):
    now = dt.datetime.now(dt.timezone.utc)
    b = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
        .issuer_name(issuer_name or x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
        .public_key(pub)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=1))
        .not_valid_after(now + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    if server:
        b = b.add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            ]),
            critical=False,
        ).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    elif not ca:
        b = b.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
    return b.sign(issuer_key, hashes.SHA256())


def generate_certs(directory: str | Path) -> dict[str, bytes]:
    """CA autofirmada + cert de servidor (localhost) + cert de cliente, en PEM."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    pem_key = lambda k: k.private_bytes(  # noqa: E731
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    pem_crt = lambda c: c.public_bytes(serialization.Encoding.PEM)  # noqa: E731

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = _cert("mlops2-ca", ca_key.public_key(), None, ca_key, ca=True)
    srv_key = ec.generate_private_key(ec.SECP256R1())
    srv = _cert("localhost", srv_key.public_key(), ca.subject, ca_key, server=True)
    cli_key = ec.generate_private_key(ec.SECP256R1())
    cli = _cert("batch-job", cli_key.public_key(), ca.subject, ca_key)

    out = {
        "ca.crt": pem_crt(ca),
        "server.crt": pem_crt(srv), "server.key": pem_key(srv_key),
        "client.crt": pem_crt(cli), "client.key": pem_key(cli_key),
    }
    for name, data in out.items():
        (d / name).write_bytes(data)
    return out


# --- proxy espía (para demostrar el man-in-the-middle) ------------------------

class SpyProxy:
    """Proxy TCP que reenvía el tráfico cliente <-> servidor y guarda una copia.

    Simula a alguien en la red (wifi, otro contenedor, un balanceador comprometido)
    que puede leer todo lo que pasa por el cable.
    """

    def __init__(self, listen_port: int, target_port: int):
        self.target = ("127.0.0.1", target_port)
        self.captured = bytearray()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", listen_port))
        self._sock.listen()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                cli, _ = self._sock.accept()
            except OSError:
                return
            srv = socket.create_connection(self.target)
            for a, b in ((cli, srv), (srv, cli)):
                threading.Thread(target=self._pump, args=(a, b), daemon=True).start()

    def _pump(self, source: socket.socket, target: socket.socket) -> None:
        try:
            while data := source.recv(65536):
                self.captured += data
                target.sendall(data)
        except OSError:
            pass
        finally:
            target.close()

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)  # despierta el accept() bloqueado
        except OSError:
            pass
        self._sock.close()


# --- servidor y cliente -------------------------------------------------------

def serve_secure(
    port: int,
    certs: dict[str, bytes],
    *,
    mtls: bool = True,
    secret: str = JWT_SECRET,
    max_msg_bytes: int = MAX_MSG_BYTES,
) -> tuple[grpc.Server, AuditLog]:
    """Servidor gRPC con interceptor de auth, TLS/mTLS y límite de tamaño."""
    audit = AuditLog()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[AuthInterceptor(audit, secret)],
        options=[("grpc.max_receive_message_length", max_msg_bytes)],
    )
    scoring_pb2_grpc.add_ScoringServicer_to_server(ScoringServicer(), server)
    creds = grpc.ssl_server_credentials(
        [(certs["server.key"], certs["server.crt"])],
        root_certificates=certs["ca.crt"] if mtls else None,
        require_client_auth=mtls,
    )
    server.add_secure_port(f"127.0.0.1:{port}", creds)
    server.start()
    return server, audit


def secure_channel(port: int, certs: dict[str, bytes], *, with_client_cert: bool = True) -> grpc.Channel:
    creds = grpc.ssl_channel_credentials(
        root_certificates=certs["ca.crt"],
        private_key=certs["client.key"] if with_client_cert else None,
        certificate_chain=certs["client.crt"] if with_client_cert else None,
    )
    return grpc.secure_channel(f"localhost:{port}", creds)


def with_token(token: str) -> list[tuple[str, str]]:
    return [("authorization", f"Bearer {token}")]


def free_port() -> int:
    """Un puerto TCP libre en localhost (para no chocar con otros servicios)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
