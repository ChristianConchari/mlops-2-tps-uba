"""API gRPC — Mini-TP 3 (Sesión 3), sobre el mismo servicio compartido.

Contrato tipado en `proto/scoring.proto` (mismos 4 features que `PredictRequest`
de REST). El modelo se carga **una sola vez** (vía `model.py`, igual que REST
y GraphQL) y se expone por dos métodos:
  Predict       (unary)            -> una predicción, equivalente a POST /v1/predict
  PredictBatch  (server-streaming) -> stream de predicciones para un lote

Levantar:  uv run python -m model_service.grpc_server   # puerto 50051
"""

from __future__ import annotations

from concurrent import futures

import grpc

from . import model as svc
from . import scoring_pb2, scoring_pb2_grpc


def _to_features(msg: scoring_pb2.Features) -> list[float]:
    return [
        msg.sepal_length_cm,
        msg.sepal_width_cm,
        msg.petal_length_cm,
        msg.petal_width_cm,
    ]


class ScoringServicer(scoring_pb2_grpc.ScoringServicer):
    def Predict(self, request: scoring_pb2.Features, context) -> scoring_pb2.Prediction:
        pred, label = svc.predict(_to_features(request))
        return scoring_pb2.Prediction(
            prediction=pred,
            prediction_label=label,
            model_version=svc.METADATA.get("model_version", "unknown"),
        )

    def PredictBatch(self, request: scoring_pb2.FeatureBatch, context):
        version = svc.METADATA.get("model_version", "unknown")
        for item in request.items:
            pred, label = svc.predict(_to_features(item))
            yield scoring_pb2.Prediction(
                prediction=pred, prediction_label=label, model_version=version
            )


def serve(port: int = 50051) -> grpc.Server:
    """Crea y arranca el servidor (no bloqueante); el llamador lo detiene."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    scoring_pb2_grpc.add_ScoringServicer_to_server(ScoringServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server


if __name__ == "__main__":
    server = serve()
    print("Servidor gRPC en localhost:50051")
    server.wait_for_termination()
