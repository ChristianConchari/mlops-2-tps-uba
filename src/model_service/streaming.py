"""Scoring online de `iris-rf` sobre un flujo de eventos (Mini-TP 4).

Reutiliza el mismo modelo servido por REST (TP1), GraphQL (TP2) y gRPC (TP3)
(`model.predict` / `model.predict_batch`), ahora consumido evento a evento
desde una cola en lugar de por request. La cola simula un topic de Kafka:
reemplazarla por un `KafkaConsumer` real no cambia el resto del pipeline.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import load_iris

from .model import FEATURES, predict, predict_batch

STOP = object()

# Media y desvío de cada feature en el dataset de entrenamiento -> referencia
# de "comportamiento normal" para el detector de drift.
_iris = load_iris()
BASELINE_MEAN = _iris.data.mean(axis=0)
BASELINE_STD = _iris.data.std(axis=0)

WINDOW_SIZE = 100  # ventana deslizante (últimos N eventos) para las métricas
CHECK_EVERY = 50  # cada cuántos eventos se re-evalúa la ventana
DRIFT_Z_THRESHOLD = 2.5  # alerta si la media de la ventana se aleja > 2.5 sigma


@dataclass
class Evento:
    values: list[float]
    t: float


def generar_eventos(n: int, drift_at: float = 0.5, seed: int = 42) -> list[Evento]:
    """Genera `n` eventos con las 4 features del modelo. A partir de
    `drift_at` (fracción del flujo) corre la distribución +2.5 sigma por
    feature, simulando un sensor/población distinta, para poder disparar
    el detector de drift más adelante."""
    rng = random.Random(seed)
    corte = int(n * drift_at)
    eventos = []
    for i in range(n):
        centro = BASELINE_MEAN + (2.5 * BASELINE_STD if i >= corte else 0.0)
        values = [rng.gauss(mu, sigma) for mu, sigma in zip(centro, BASELINE_STD)]
        eventos.append(Evento(values=values, t=time.time()))
    return eventos


def productor(stream: "queue.Queue", eventos: list[Evento], delay: float = 0.0005) -> None:
    for ev in eventos:
        stream.put(ev)
        time.sleep(delay)
    stream.put(STOP)


@dataclass
class ResultadoStreaming:
    latencias_ms: list[float] = field(default_factory=list)
    predicciones: list[int] = field(default_factory=list)
    alertas: list[dict] = field(default_factory=list)
    throughput_ev_s: float = 0.0
    p95_ms: float = 0.0
    duracion_s: float = 0.0


def _revisar_drift(ventana: deque[Evento], procesados: int, resultado: ResultadoStreaming) -> None:
    medias = np.mean([w.values for w in ventana], axis=0)
    z = (medias - BASELINE_MEAN) / BASELINE_STD
    idx = int(np.argmax(np.abs(z)))
    z_max = float(z[idx])
    if abs(z_max) > DRIFT_Z_THRESHOLD:
        resultado.alertas.append(
            {"evento": procesados, "feature": FEATURES[idx], "z_score": round(z_max, 2)}
        )


def consumidor(stream: "queue.Queue", resultado: ResultadoStreaming) -> None:
    ventana: deque[Evento] = deque(maxlen=WINDOW_SIZE)  # últimos WINDOW_SIZE eventos
    procesados = 0
    while True:
        ev = stream.get()
        if ev is STOP:
            break

        t0 = time.perf_counter()
        pred, _ = predict(ev.values)  # inferencia online, un evento a la vez
        resultado.latencias_ms.append((time.perf_counter() - t0) * 1000)
        resultado.predicciones.append(pred)

        ventana.append(ev)
        procesados += 1
        if procesados % CHECK_EVERY == 0 and ventana:
            _revisar_drift(ventana, procesados, resultado)


def correr_streaming(eventos: list[Evento], delay: float = 0.0005) -> ResultadoStreaming:
    """Levanta productor + consumidor en hilos separados y devuelve las
    métricas por ventana (throughput, p95, alertas de drift)."""
    stream: queue.Queue = queue.Queue()
    resultado = ResultadoStreaming()

    t0 = time.time()
    pt = threading.Thread(target=productor, args=(stream, eventos, delay))
    ct = threading.Thread(target=consumidor, args=(stream, resultado))
    pt.start()
    ct.start()
    pt.join()
    ct.join()
    resultado.duracion_s = time.time() - t0

    resultado.throughput_ev_s = len(resultado.latencias_ms) / resultado.duracion_s
    resultado.p95_ms = float(np.percentile(resultado.latencias_ms, 95))
    return resultado


def puntuar_batch(eventos: list[Evento]) -> tuple[list[int], float]:
    """Puntúa todos los eventos de una sola vez. Devuelve (predicciones, ms_totales)."""
    X = [e.values for e in eventos]
    t0 = time.perf_counter()
    preds = predict_batch(X)
    ms = (time.perf_counter() - t0) * 1000
    return preds, ms
