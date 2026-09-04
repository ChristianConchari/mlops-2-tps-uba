# Mini-TP 1 — Del modelo al servicio (Sesión 1)

Sirve un modelo por REST con **FastAPI**, con contrato Pydantic, endpoint de
predicción online versionado y `/health`.

## Modelo

Modelo juguete: **RandomForestClassifier** (con `StandardScaler`) sobre el
dataset **Iris** de scikit-learn. El artefacto entrenado se guarda en
`model/model.joblib` junto con su metadata (features, clases, versión, accuracy).

- **Versión del modelo:** `1.0.0`
- **Features reales (contrato):** `sepal_length_cm`, `sepal_width_cm`,
  `petal_length_cm`, `petal_width_cm` (todas en cm, `> 0`).
- **Salida:** clase (`0/1/2`), etiqueta (`setosa/versicolor/virginica`) y
  `model_version`.

## Requisitos

Python ≥ 3.10 y [uv](https://docs.astral.sh/uv/).

## Cómo ejecutar

```bash
# parado dentro de mini_tp1/
uv venv --python 3.11 .venv
uv pip install -r requirements.txt

# 1. Entrenar y generar el artefacto (model/model.joblib)
uv run python train.py

# 2. Levantar la API
uv run uvicorn app:app --reload --port 8000
#    docs interactivas: http://localhost:8000/docs

# 3. En otra terminal: cliente con un caso válido y uno inválido (422)
uv run python client.py
```

> Sin uv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`,
> luego reemplazar `uv run` por ejecución directa.

## Endpoints

| Método | Ruta          | Descripción                                              |
|--------|---------------|---------------------------------------------------------|
| GET    | `/health`     | Estado del servicio y si el modelo está cargado          |
| POST   | `/v1/predict` | Predicción online. Devuelve predicción + `model_version` |

### Ejemplo — caso válido

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length_cm":5.1,"sepal_width_cm":3.5,"petal_length_cm":1.4,"petal_width_cm":0.2}'
# {"prediction":0,"prediction_label":"setosa","model_version":"1.0.0"}
```

### Ejemplo — caso inválido (HTTP 422)

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length_cm":5.1,"sepal_width_cm":3.5,"petal_length_cm":1.4,"petal_width_cm":-1.0}'
# 422 Unprocessable Entity — "Input should be greater than 0"
```

La validación Pydantic (`gt=0`, `extra="forbid"`) rechaza valores fuera de rango,
tipos incorrectos o campos desconocidos antes de llegar al modelo.

## Estructura

```
mini_tp1/
├── train.py           # entrena y genera model/model.joblib + metadata.json
├── app.py             # API FastAPI: /health y /v1/predict
├── client.py          # cliente de prueba (caso válido + inválido)
├── requirements.txt
├── model/             # artefacto generado por train.py
│   ├── model.joblib
│   └── metadata.json
└── README.md
```

## Checklist del Mini-TP 1

- [x] Contrato Pydantic con las features reales del modelo
- [x] Endpoint `/v1/predict` (online) y `/health`
- [x] La respuesta devuelve la predicción y la versión del modelo
- [x] Cliente con un caso válido y uno inválido (muestra el 422)
- [x] Corre de punta a punta con `uv` + README
