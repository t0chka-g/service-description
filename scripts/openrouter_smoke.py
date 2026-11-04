"""Ручной smoke-test полного pipeline через OpenRouter.

Скрипт не ходит в сеть, пока явно не заданы ключ и подтверждение.
Запуск только вручную: запрос платный.

    $env:OPENAI_API_KEY = "<openrouter-key>"
    $env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
    $env:OPENAI_MODEL = "openai/gpt-4.1-mini"
    $env:SMOKE_OPENROUTER = "I_UNDERSTAND_THIS_IS_PAID"
    python scripts/openrouter_smoke.py
"""

from __future__ import annotations

import json
import os
import sys

import httpx

CONFIRM = "I_UNDERSTAND_THIS_IS_PAID"
GENERATE_PATH = "/api/v1/products/generate"

MANOMETER = {
    "name": "Манометр демонстрационный МП-Демо-01",
    "manufacturer": "Учебный стенд «Сигнал»",
    "description": (
        "Учебный пример, не паспорт реального прибора. "
        "Диапазон показаний 0...1,6 МПа. Класс точности 1,5. "
        "Присоединение M20x1,5. Степень защиты IP54."
    ),
    "specs": {
        "Диапазон показаний": "0...1,6 МПа",
        "Класс точности": "1,5",
        "Присоединение": "M20x1,5",
        "Степень защиты": "IP54",
    },
}

THERMOMETER = {
    "name": "Термометр демонстрационный ТС-Демо-02",
    "manufacturer": "Учебный стенд «Сигнал»",
    "description": (
        "Учебный пример с одной подтвержденной характеристикой. "
        "Диапазон измерений 0...100 °C."
    ),
    "specs": {"Диапазон измерений": "0...100 °C"},
}


def main() -> int:
    if os.environ.get("SMOKE_OPENROUTER") != CONFIRM:
        print(
            "Платный вызов не выполнен. "
            f"Для ручного запуска задайте SMOKE_OPENROUTER={CONFIRM}.",
            file=sys.stderr,
        )
        return 2
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("OPENAI_API_KEY пуст. Запрос не отправлен.", file=sys.stderr)
        return 2

    base_url = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000")
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        for name, payload in ("manometer", MANOMETER), ("thermometer", THERMOMETER):
            response = client.post(GENERATE_PATH, json=payload)
            print(name, response.status_code)
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
            response.raise_for_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
