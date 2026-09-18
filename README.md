# MLOps II (UBA) — Entregas

Trabajos prácticos de la materia, como un **servicio que evoluciona** clase a
clase. Autor: **Christian Conchari**.

## Estructura

```
src/model_service/     el servicio (código compartido entre TPs)
  model.py             carga del modelo Iris + métricas
  rest.py              API REST      (Mini-TP 1)
  graphql.py           API GraphQL   (Mini-TP 2)
  proto/scoring.proto  contrato gRPC (Mini-TP 3)
  scoring_pb2*.py       stubs generados (Mini-TP 3)
  grpc_server.py        API gRPC      (Mini-TP 3)
train.py               entrena y genera model/model.joblib
model/                 artefacto entrenado + metadata
entregas/              un notebook por entrega (narrativa evaluable)
  tp1.ipynb            REST: contrato, /v1/predict, /health
  tp2.ipynb            GraphQL + comparación REST vs GraphQL
  tp3.ipynb            gRPC (unary + streaming) + comparación gRPC vs REST
```

Cada entrega queda fijada con un tag de git (`entrega-tp1`, `entrega-tp2`, …).
`main` es el estado actual del servicio.

## Bitácora

| TP | Sesión | Qué se agregó | Notebook | Tag | Estado |
|----|--------|---------------|----------|-----|--------|
| Mini-TP 1 | 1 | Servicio REST del modelo `iris-rf` (FastAPI + contrato Pydantic): `POST /v1/predict` versionado, `GET /health`, validación → 422 | `entregas/tp1.ipynb` | `entrega-tp1` | ✅ Entregado |
| Mini-TP 2 | 2 | Capa GraphQL (Strawberry) sobre el mismo servicio: tipo `Model` (`name`, `version`, `algorithm`, `metrics`) leído del artefacto + comparación REST vs GraphQL | `entregas/tp2.ipynb` | `entrega-tp2` | ✅ Entregado |
| Mini-TP 3 | 3 | Servicio gRPC (`scoring.proto` tipado a los 4 features del modelo): `Predict` unary y `PredictBatch` server-streaming, servidor carga el modelo una sola vez + comparación de latencia gRPC vs REST | `entregas/tp3.ipynb` | `entrega-tp3` | ✅ Entregado |

### Mini-TP 2 — REST vs GraphQL (medido en el notebook)

Armar *la vista del modelo* (`name`, `version`, `algorithm`, `auc`, `accuracy`, `f1`):

| | Llamadas | Bytes | Campos de más |
|---|---|---|---|
| REST | 2 (`/v1/model` + `/v1/model/metrics`) | 341 | 4 (`dataset`, `features`, `target_names`, `trained_at`) |
| GraphQL | 1 | 158 | 0 |

GraphQL: una request con la forma exacta del dato (2.2× menos bytes, sin
under- ni over-fetching). Costo: definir esquema y resolvers. Para una API chica
y estable REST alcanza; GraphQL gana con muchos consumidores o grafos de datos
con relaciones (linaje, MLflow).

### Mini-TP 3 — gRPC vs REST (medido en el notebook)

200 llamadas a `Predict`/`POST /v1/predict` con el mismo input, en localhost:

| | ms/llamada |
|---|---|
| gRPC | 15.05 |
| REST | 16.10 |

Solo ~1.1x más rápido (mucho menos que en el lab teórico con modelos de
juguete): el modelo real es un `RandomForestClassifier` de 200 árboles, así
que la **inferencia domina** el tiempo total y diluye el ahorro de transporte
de gRPC (HTTP/2 + Protobuf binario vs JSON/HTTP-1.1). gRPC seguiría siendo la
opción para comunicación interna de alto volumen o que necesite streaming
real; el costo de su contrato estricto no se justificó acá en latencia pura.

## Puesta en marcha

```bash
uv venv --python 3.11 .venv
uv pip install -e .
uv run python train.py          # genera model/model.joblib
uv run jupyter lab              # abrir entregas/*.ipynb
```

Las APIs también se levantan sueltas:

```bash
uv run uvicorn model_service.rest:app --port 8100       # REST
uv run uvicorn model_service.graphql:app --port 8110    # GraphQL -> /graphql
uv run python -m model_service.grpc_server               # gRPC -> :50051
```

Para regenerar los stubs de gRPC si cambia el `.proto`:

```bash
uv run python -m grpc_tools.protoc -I./src/model_service/proto \
    --python_out=./src/model_service --grpc_python_out=./src/model_service \
    ./src/model_service/proto/scoring.proto
```
