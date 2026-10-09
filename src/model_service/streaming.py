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
class Event:
    values: list[float]
    t: float


def generate_events(n: int, drift_at: float = 0.5, seed: int = 42) -> list[Event]:
    """Genera `n` eventos con las 4 features del modelo. A partir de
    `drift_at` (fracción del flujo) corre la distribución +2.5 sigma por
    feature, simulando un sensor/población distinta, para poder disparar
    el detector de drift más adelante."""
    rng = random.Random(seed)
    cutoff = int(n * drift_at)
    events = []
    for i in range(n):
        center = BASELINE_MEAN + (2.5 * BASELINE_STD if i >= cutoff else 0.0)
        values = [rng.gauss(mu, sigma) for mu, sigma in zip(center, BASELINE_STD)]
        events.append(Event(values=values, t=time.time()))
    return events


def producer(stream: "queue.Queue", events: list[Event], delay: float = 0.0005) -> None:
    for ev in events:
        stream.put(ev)
        time.sleep(delay)
    stream.put(STOP)


@dataclass
class StreamingResult:
    latencies_ms: list[float] = field(default_factory=list)
    predictions: list[int] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    throughput_ev_s: float = 0.0
    p95_ms: float = 0.0
    duration_s: float = 0.0


def _check_drift(window: deque[Event], processed: int, result: StreamingResult) -> None:
    means = np.mean([w.values for w in window], axis=0)
    z = (means - BASELINE_MEAN) / BASELINE_STD
    idx = int(np.argmax(np.abs(z)))
    z_max = float(z[idx])
    if abs(z_max) > DRIFT_Z_THRESHOLD:
        result.alerts.append(
            {"event": processed, "feature": FEATURES[idx], "z_score": round(z_max, 2)}
        )


def consumer(stream: "queue.Queue", result: StreamingResult) -> None:
    window: deque[Event] = deque(maxlen=WINDOW_SIZE)  # últimos WINDOW_SIZE eventos
    processed = 0
    while True:
        ev = stream.get()
        if ev is STOP:
            break

        t0 = time.perf_counter()
        pred, _ = predict(ev.values)  # inferencia online, un evento a la vez
        result.latencies_ms.append((time.perf_counter() - t0) * 1000)
        result.predictions.append(pred)

        window.append(ev)
        processed += 1
        if processed % CHECK_EVERY == 0 and window:
            _check_drift(window, processed, result)


def run_streaming(events: list[Event], delay: float = 0.0005) -> StreamingResult:
    """Levanta productor + consumidor en hilos separados y devuelve las
    métricas por ventana (throughput, p95, alertas de drift)."""
    stream: queue.Queue = queue.Queue()
    result = StreamingResult()

    t0 = time.time()
    pt = threading.Thread(target=producer, args=(stream, events, delay))
    ct = threading.Thread(target=consumer, args=(stream, result))
    pt.start()
    ct.start()
    pt.join()
    ct.join()
    result.duration_s = time.time() - t0

    result.throughput_ev_s = len(result.latencies_ms) / result.duration_s
    result.p95_ms = float(np.percentile(result.latencies_ms, 95))
    return result


def score_batch(events: list[Event]) -> tuple[list[int], float]:
    """Puntúa todos los eventos de una sola vez. Devuelve (predicciones, ms_totales)."""
    X = [e.values for e in events]
    t0 = time.perf_counter()
    preds = predict_batch(X)
    ms = (time.perf_counter() - t0) * 1000
    return preds, ms
