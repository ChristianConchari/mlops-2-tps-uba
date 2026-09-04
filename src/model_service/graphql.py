"""API GraphQL — Mini-TP 2 (Sesión 2), sobre el mismo servicio compartido.

Expone los metadatos del modelo con un tipo `Model` (name, version, metrics).
Una sola query trae exactamente los campos pedidos (sin over-fetching).

Levantar:  uv run uvicorn model_service.graphql:app --reload --port 8010
GraphiQL:  http://127.0.0.1:8010/graphql
    query { model { name version algorithm metrics { auc accuracy f1 } } }
"""

from __future__ import annotations

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

from . import model as svc


@strawberry.type
class Metrics:
    auc: float
    accuracy: float
    f1: float


@strawberry.type
class Model:
    name: str
    version: str
    algorithm: str

    @strawberry.field
    def metrics(self) -> Metrics:
        m = svc.metrics()
        return Metrics(auc=m["auc"], accuracy=m["accuracy"], f1=m["f1"])


@strawberry.type
class Query:
    @strawberry.field
    def model(self) -> Model:
        return Model(
            name=svc.METADATA.get("model_name", "unknown"),
            version=str(svc.METADATA.get("model_version", "0")),
            algorithm=svc.METADATA.get("algorithm", "unknown"),
        )


schema = strawberry.Schema(query=Query)

app = FastAPI(title="model-service — GraphQL")
app.include_router(GraphQLRouter(schema), prefix="/graphql")
