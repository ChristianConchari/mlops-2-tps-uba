"""Cliente de prueba: dispara un caso válido y uno inválido (muestra el 422)."""

from __future__ import annotations

import requests

BASE_URL = "http://localhost:8000"


def main() -> None:
    print("== /health ==")
    print(requests.get(f"{BASE_URL}/health").json())

    print("\n== Caso VÁLIDO -> /v1/predict ==")
    ok = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
    r = requests.post(f"{BASE_URL}/v1/predict", json=ok)
    print(f"HTTP {r.status_code}")
    print(r.json())

    print("\n== Caso INVÁLIDO (petal_width_cm negativo) -> 422 esperado ==")
    bad = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": -1.0,
    }
    r = requests.post(f"{BASE_URL}/v1/predict", json=bad)
    print(f"HTTP {r.status_code}")
    print(r.json())
    assert r.status_code == 422, "Se esperaba un 422 para la entrada inválida"
    print("\nOK: la validación rechazó la entrada mala con 422.")


if __name__ == "__main__":
    main()
