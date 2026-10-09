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
  streaming.py          scoring online + drift (Mini-TP 4)
  datalake.py           Data Lake MinIO/S3: zonas + modelo versionado (Mini-TP 6)
  lake_security.py      seguridad SAIF del lake (Mini-TP 7)
  grpc_security.py      seguridad SAIF de gRPC: auth, mTLS, límite de tamaño (Mini-TP 7)
train.py               entrena y genera model/model.joblib
model/                 artefacto entrenado + metadata
entregas/              un notebook por entrega (narrativa evaluable)
  tp1.ipynb            REST: contrato, /v1/predict, /health
  tp2.ipynb            GraphQL + comparación REST vs GraphQL
  tp3.ipynb            gRPC (unary + streaming) + comparación gRPC vs REST
  tp4.ipynb            streaming: inferencia online, ventanas, drift, batch vs streaming
  tp6.ipynb            Data Lake: zonas, models/v1, servir desde el lake, MLflow -> MinIO
  tp7.ipynb            seguridad SAIF del lake y de gRPC
```

Cada entrega queda fijada con un tag de git (`entrega-tp1`, `entrega-tp2`, …).
`main` es el estado actual del servicio.

## Bitácora

| TP | Sesión | Qué se agregó | Notebook | Tag | Estado |
|----|--------|---------------|----------|-----|--------|
| Mini-TP 1 | 1 | Servicio REST del modelo `iris-rf` (FastAPI + contrato Pydantic): `POST /v1/predict` versionado, `GET /health`, validación → 422 | `entregas/tp1.ipynb` | `entrega-tp1` | ✅ Entregado |
| Mini-TP 2 | 2 | Capa GraphQL (Strawberry) sobre el mismo servicio: tipo `Model` (`name`, `version`, `algorithm`, `metrics`) leído del artefacto + comparación REST vs GraphQL | `entregas/tp2.ipynb` | `entrega-tp2` | ✅ Entregado |
| Mini-TP 3 | 3 | Servicio gRPC (`scoring.proto` tipado a los 4 features del modelo): `Predict` unary y `PredictBatch` server-streaming, servidor carga el modelo una sola vez + comparación de latencia gRPC vs REST | `entregas/tp3.ipynb` | `entrega-tp3` | ✅ Entregado |
| Mini-TP 4 | 4 | Scoring online sobre un flujo de eventos (`streaming.py`): productor/consumidor en threads, inferencia evento a evento, métricas por ventana deslizante (throughput, p95), detector de drift por z-score + alertas, comparación contra scoring batch | `entregas/tp4.ipynb` | `entrega-tp4` | ✅ Entregado |
| Mini-TP 6 | 6 | Data Lake en MinIO (`datalake.py`): bucket `mlops2-lake` con zonas `raw`/`curated`/`models`, artefacto versionado en `models/v1/`, `MODEL_SOURCE=lake` hace que el servicio descargue el modelo al arrancar (la API REST lo informa en `/health`), experimento de MLflow con artefactos en `s3://mlops2-lake/mlflow/` | `entregas/tp6.ipynb` | `entrega-tp6` | 🟡 Listo, sin entregar |
| Mini-TP 7 | 7 | Seguridad SAIF en dos capas. **Data Lake** (`lake_security.py`): bucket público vs URL prefirmada con expiración, SHA-256 verificado antes de `joblib.load` (detecta el artefacto manipulado), un usuario de MinIO por rol con política IAM por prefijo (serving no puede escribir `models/`, ingestor no puede leerlo) y audit log persistido en `audit/`. **gRPC** (`grpc_security.py`): proxy espía que lee token y predicción del canal inseguro, interceptor que verifica un JWT (firma HS256, vencimiento, emisor) y autoriza por el claim `role` (`UNAUTHENTICATED`/`PERMISSION_DENIED`), mTLS con CA propia (el espía ya no lee nada), `max_receive_message_length` de 64 KB (`RESOURCE_EXHAUSTED`) y audit log por RPC | `entregas/tp7.ipynb` | `entrega-tp7` | 🟡 Listo, sin entregar |

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
| gRPC | 16.54 |
| REST | 16.80 |

Prácticamente empatados (~1.0x, mucho menos margen que en el lab teórico con
modelos de juguete): el modelo real es un `RandomForestClassifier` de 200
árboles, así que la **inferencia domina** el tiempo total y diluye el ahorro
de transporte de gRPC (HTTP/2 + Protobuf binario vs JSON/HTTP-1.1). gRPC
seguiría siendo la opción para comunicación interna de alto volumen o que
necesite streaming real; el costo de su contrato estricto no se justificó
acá en latencia pura.

### Mini-TP 4 — streaming vs batch (medido en el notebook)

800 eventos (features tipo Iris, drift +2.5σ inyectado a mitad del flujo),
mismo modelo `iris-rf` puntuado evento a evento vs en un solo batch:

| | ms/evento |
|---|---|
| Streaming (online) | 14.77 |
| Batch | 0.027 |

El batch resultó ~540x más barato por evento (una sola llamada vectorizada a
`pipeline.predict` vs. N llamadas individuales con overhead de cola/hilo en
el medio); las 800 predicciones coinciden entre ambos caminos. El detector de
drift (z-score de la media de la ventana deslizante de 100 eventos contra la
media/desvío de entrenamiento, revisado cada 50 eventos) disparó 7 alertas,
todas después del evento 500 — la primera ventana que queda enteramente del
lado corrido del flujo tras el cambio de distribución inyectado en el 400.

## Puesta en marcha

```bash
uv venv --python 3.11 .venv
uv pip install -e .
uv run python train.py          # genera model/model.joblib
uv run jupyter lab              # abrir entregas/*.ipynb
```

Los Mini-TPs 6 y 7 necesitan MinIO en `localhost:9000` (`minio` / `minio123`):

```bash
docker run -d --name minio -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minio -e MINIO_ROOT_PASSWORD=minio123 \
  quay.io/minio/minio server /data --console-address ":9001"

MODEL_SOURCE=lake uv run uvicorn model_service.rest:app --port 8100   # REST sirviendo desde el lake
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
