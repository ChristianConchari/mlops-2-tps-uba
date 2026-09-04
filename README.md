# MLOps II (UBA) — Entregas

Trabajos prácticos de la materia, como un **servicio que evoluciona** clase a
clase. Autor: **Christian Conchari**.

## Estructura

```
src/model_service/     el servicio (código compartido entre TPs)
  model.py             carga del modelo Iris + métricas
  rest.py              API REST      (Mini-TP 1)
  graphql.py           API GraphQL   (Mini-TP 2)
train.py               entrena y genera model/model.joblib
model/                 artefacto entrenado + metadata
entregas/              un notebook por entrega (narrativa evaluable)
  tp1.ipynb            REST: contrato, /v1/predict, /health
  tp2.ipynb            GraphQL + comparación REST vs GraphQL
```

Cada entrega queda fijada con un tag de git (`entrega-tp1`, `entrega-tp2`, …).
`main` es el estado actual del servicio.

## Bitácora

| TP | Sesión | Qué se agregó | Notebook | Tag |
|----|--------|---------------|----------|-----|
| Mini-TP 1 | 1 | Servicio REST del modelo Iris (FastAPI + Pydantic) | `entregas/tp1.ipynb` | `entrega-tp1` |
| Mini-TP 2 | 2 | Capa GraphQL sobre el mismo servicio + comparación de llamadas/bytes vs REST | `entregas/tp2.ipynb` | `entrega-tp2` |

## Puesta en marcha

```bash
uv venv --python 3.11 .venv
uv pip install -e .
uv run python train.py          # genera model/model.joblib
uv run jupyter lab              # abrir entregas/*.ipynb
```

Las APIs también se levantan sueltas:

```bash
uv run uvicorn model_service.rest:app --port 8000       # REST
uv run uvicorn model_service.graphql:app --port 8010    # GraphQL -> /graphql
```
