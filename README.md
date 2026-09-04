# MLOps II (LSE · FIUBA) — Entregas

Trabajos prácticos de la materia. Autor: **Christian Conchari**.

| TP | Tema | Carpeta | Estado |
|----|------|---------|--------|
| Mini-TP 1 | Del modelo al servicio: modelo Iris servido por REST con FastAPI (contrato Pydantic, `/v1/predict`, `/health`) | [`tp1/`](tp1/) | Completo |
| Mini-TP 2 | Metadatos del modelo por GraphQL (Strawberry) y comparación REST vs GraphQL | [`tp2/`](tp2/) | En curso |

## Cómo correr

Cada TP tiene su propio README con instrucciones. En general se usa
[uv](https://docs.astral.sh/uv/):

```bash
cd tp1
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```
