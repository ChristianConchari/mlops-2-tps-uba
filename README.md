# MLOps II (UBA) — Entregas

Mini-TPs de la materia, como un **servicio que evoluciona** clase a clase.
Autor: **Christian Conchari**.

Todo gira alrededor de un único modelo, `iris-rf` (RandomForest sobre Iris, 4
features, 3 clases), que cada TP expone, mueve o protege de una forma distinta:

| Capa | TP |
|---|---|
| Serving REST | 1 |
| GraphQL | 2 |
| gRPC | 3 |
| Streaming | 4 |
| Aprendizaje federado | 5 |
| Data Lake | 6 |
| Seguridad (SAIF) | 7 |

El modelo es deliberadamente simple: el foco es la **puesta en producción**,
no el modelado.

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
  federated.py          FedAvg, particiones IID/non-IID, DP-FedAvg (Mini-TP 5)
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
  tp5.ipynb            federado: FedAvg IID vs non-IID vs centralizado + privacidad diferencial
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
| Mini-TP 5 | 5 | Aprendizaje federado sobre el dataset de `iris-rf` (`federated.py`): FedAvg con un softmax (el RF no tiene parámetros promediables), 6 clientes con 1/3 por ronda, partición IID vs non-IID (una clase por cliente) vs centralizado, 30 semillas, y DP-FedAvg (recorte + ruido) barriendo sigma | `entregas/tp5.ipynb` | `entrega-tp5` | ✅ Entregado |
| Mini-TP 6 | 6 | Data Lake en MinIO (`datalake.py`): bucket `mlops2-lake` con zonas `raw`/`curated`/`models`, artefacto versionado en `models/v1/`, `MODEL_SOURCE=lake` hace que el servicio descargue el modelo al arrancar (la API REST lo informa en `/health`), experimento de MLflow con artefactos en `s3://mlops2-lake/mlflow/` | `entregas/tp6.ipynb` | `entrega-tp6` | ✅ Entregado |
| Mini-TP 7 | 7 | Seguridad SAIF en dos capas. **Data Lake** (`lake_security.py`): bucket público vs URL prefirmada con expiración, SHA-256 verificado antes de `joblib.load` (detecta el artefacto manipulado), un usuario de MinIO por rol con política IAM por prefijo (serving no puede escribir `models/`, ingestor no puede leerlo) y audit log persistido en `audit/`. **gRPC** (`grpc_security.py`): proxy espía que lee token y predicción del canal inseguro, interceptor que verifica un JWT (firma HS256, vencimiento, emisor) y autoriza por el claim `role` (`UNAUTHENTICATED`/`PERMISSION_DENIED`), mTLS con CA propia (el espía ya no lee nada), `max_receive_message_length` de 64 KB (`RESOURCE_EXHAUSTED`) y audit log por RPC | `entregas/tp7.ipynb` | `entrega-tp7` | ✅ Entregado |

### Mini-TP 1 — servicio REST (medido en el notebook)

`GET /health` informa `status`, `model_loaded` y `model_version`. El contrato
es `PredictRequest` (4 floats, `gt=0`, `extra="forbid"`):

| Caso | Resultado |
|---|---|
| `POST /v1/predict` con features válidas | 200 → `{"prediction": 0, "prediction_label": "setosa", "model_version": "1.0.0"}` |
| `petal_width_cm = -1.0` | 422 (`Input should be greater than 0`) |

Las entradas inválidas se rechazan antes de llegar al modelo y la respuesta
siempre trae la versión del modelo, para trazabilidad.

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

### Mini-TP 5 — federado (medido en el notebook)

FedAvg promedia parámetros y un RandomForest no los tiene, así que lo que viaja
entre clientes es un clasificador **softmax** sobre las mismas 4 features y el
mismo split que `train.py`. 6 clientes (20 filas c/u), 2 por ronda, 30 rondas,
10 épocas locales; media de 30 semillas sobre 30 filas de test:

| | Accuracy |
|---|---|
| Centralizado | 0.933 |
| FedAvg IID | 0.948 ± 0.017 |
| FedAvg non-IID (una clase por cliente) | 0.876 ± 0.086 |

En IID no hay costo medible (que supere al centralizado es ruido: una fila del
test vale 0.033). En non-IID el costo está en la **varianza**: la curva oscila
unas 5x más (0.073 vs 0.013 ronda a ronda) por *client drift*.

DP-FedAvg (recorte a norma 1 + ruido gaussiano, partición IID):

| sigma | 0 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| Accuracy | 0.948 | 0.939 | 0.893 | 0.782 | 0.650 | 0.549 |

El recorte solo es gratis; el costo viene del ruido, y pesa más porque se
promedian apenas 2 clientes por ronda. No se calculó (ε, δ): muestra la forma
del trade-off, no una garantía formal. En este dataset el federado es
**forzado** (Iris es público y cabe en memoria); sirve para entender el
mecanismo.

### Mini-TP 6 — Data Lake (medido en el notebook)

Bucket `mlops2-lake` en MinIO, con las zonas como prefijos:

| Zona | Contenido |
|---|---|
| `raw/` | `iris/iris.csv` tal cual lo entrega sklearn (150 filas) |
| `curated/` | mismo dataset con las columnas del contrato y sin duplicados (149 filas) |
| `models/v1/` | `model.joblib` + `metadata.json` |

`MODEL_SOURCE=lake` hace que `model.py` descargue el artefacto al arrancar, a
memoria y sin tocar el disco. `/health` lo confirma
(`"model_source": "lake"`). Así el ciclo de vida del modelo queda separado del
de la imagen: reentrenar no obliga a reconstruir el contenedor. Un experimento
de MLflow guarda sus artefactos en `s3://mlops2-lake/mlflow/` (el tracking va a
SQLite local).

### Mini-TP 7 — seguridad SAIF (medido en el notebook)

Se aseguran **dos** capas, las que más expone el servicio.

**Data Lake** (`lake_security.py`):

| Control | Resultado |
|---|---|
| Bucket público (vulnerable) | GET anónimo → 200, modelo completo robado |
| URL prefirmada con expiración | dentro del plazo 200; vencida o sin firma 403 |
| SHA-256 antes de `joblib.load` | artefacto alterado → bloqueado antes de deserializar |
| Un usuario por rol (IAM por prefijo) | `serving` no puede escribir `models/`; `ingestor` no puede leerlo |
| Audit log en `audit/` | registra también los accesos denegados |

**gRPC** (`grpc_security.py`):

| Control | Resultado |
|---|---|
| Proxy espía sobre el canal inseguro | captura 1127 bytes: ve la predicción y el JWT completo |
| JWT sin token / otro secreto / adulterado / vencido | `UNAUTHENTICATED` |
| Rol `predictor` llamando `PredictBatch` | `PERMISSION_DENIED` |
| Lote de 5000 (190 000 B) con límite de 64 KB | `RESOURCE_EXHAUSTED` |
| mTLS con CA propia | el espía ya no ve ni la predicción ni el JWT |
| Audit log por RPC | actor, método, código y latencia |

El checksum no frena a un atacante con escritura sobre todo `models/`, que
podría recalcularlo; por eso se combina con el mínimo privilegio del serving.

## Puesta en marcha

```bash
uv venv --python 3.11 .venv
uv pip install -e .
uv run python train.py          # genera model/model.joblib
uv run jupyter lab              # abrir entregas/*.ipynb
```

Qué necesita cada entrega:

| TP | Requisitos |
|---|---|
| 1, 2, 3, 4, 5 | solo el entorno de uv (el notebook levanta lo que necesite) |
| 6, 7 | además MinIO en `localhost:9000` |

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
