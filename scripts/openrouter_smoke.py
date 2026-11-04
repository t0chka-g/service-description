"""Ручной E2E полного pipeline через OpenRouter.

Скрипт не ходит в сеть, пока явно не заданы ключ и подтверждение.
Запуск только вручную: запросы платные. Сервер должен быть запущен с тем же ключом.

    $env:OPENAI_API_KEY = "<openrouter-key>"
    $env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
    $env:OPENAI_MODEL = "openai/gpt-4.1-mini"
    uv run uvicorn app.main:app
    # в другом окне:
    $env:SMOKE_OPENROUTER = "I_UNDERSTAND_THIS_IS_PAID"
    uv run python scripts/openrouter_smoke.py

Сценарии A и B лежат в examples/. Оба должны вернуть status=complete.
Скрипт печатает результат и сам сверяет контракт; код выхода 1, если сценарий не прошел.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

CONFIRM = "I_UNDERSTAND_THIS_IS_PAID"
GENERATE_PATH = "/api/v1/products/generate"
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCENARIOS = ("manometer", "temperature_sensor")

TITLE_MAX = 70
DESCRIPTION_MIN = 800
DESCRIPTION_MAX = 1200
QUERIES_MIN = 3
QUERIES_MAX = 5


def check_card(card: dict[str, object]) -> list[str]:
    """Возвращает нарушения контракта полной карточки. Пустой список означает успех."""

    problems: list[str] = []
    if card.get("status") != "complete":
        problems.append(f"status={card.get('status')}, ожидался complete")
    title = card.get("seo_title")
    if not isinstance(title, str) or not title or len(title) > TITLE_MAX:
        problems.append(f"seo_title вне 1–{TITLE_MAX} символов")
    description = card.get("seo_description")
    if not isinstance(description, str):
        problems.append("seo_description отсутствует")
    elif not DESCRIPTION_MIN <= len(description) <= DESCRIPTION_MAX:
        problems.append(f"seo_description {len(description)} символов, нужно 800–1200")
    queries = card.get("search_queries")
    if not isinstance(queries, list) or not QUERIES_MIN <= len(queries) <= QUERIES_MAX:
        problems.append("search_queries: нужно 3–5 запросов")
    if not card.get("commercial_block"):
        problems.append("commercial_block пуст")
    if not card.get("specifications"):
        problems.append("specifications пусты")
    return problems


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
    failed = False
    with httpx.Client(base_url=base_url, timeout=150.0) as client:
        client.get("/health").raise_for_status()
        for name in SCENARIOS:
            payload = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))
            response = client.post(GENERATE_PATH, json=payload)
            print(f"== {name}: HTTP {response.status_code}")
            body = response.json()
            print(json.dumps(body, ensure_ascii=False, indent=2))
            problems = check_card(body) if response.status_code == 200 else ["HTTP не 200"]
            for problem in problems:
                print(f"НАРУШЕНИЕ: {problem}", file=sys.stderr)
            failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
