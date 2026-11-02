# AI Product Card Generator

AI-сервис на FastAPI для генерации SEO-карточек товаров B2B-интернет-магазина с использованием LLM.

Сервис принимает JSON с названием, производителем, характеристиками и техническим описанием товара, извлекает дополнительные характеристики, проверяет данные и формирует SEO-контент без выдуманных технических параметров.

## Стек

- Python 3.12+
- FastAPI
- Pydantic v2 / pydantic-settings
- OpenAI API
- pytest / pytest-asyncio / httpx
- Ruff

## Установка

Рекомендуется [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --extra dev
cp .env.example .env
```

`uv sync --locked` ставит зависимости строго по `uv.lock` и завершается ошибкой, если lock не соответствует `pyproject.toml`. Так локально и в CI получается одно и то же окружение.

Альтернатива через pip:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env
```

Заполните `OPENAI_API_KEY` в `.env`.

## Запуск

```bash
uvicorn app.main:app --reload
```

Проверка здоровья:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ: `{"status":"ok"}`.

## Тесты

```bash
uv run pytest
```

`pytest` по умолчанию исключает маркер `llm`. Тот же фильтр стоит в CI.

Реальный прогон оценки извлечения вызывает OpenAI и пишет отчёт в `reports/evals/` (каталог в `.gitignore`):

```bash
uv run pytest -m llm -v
```

Без `OPENAI_API_KEY` этот тест пропускается. Precision, Recall и F1 в отчёте относятся к характеристикам из описания; копирование `specs` в них не входит. Hallucination Rate считается по сырому ответу модели. Ошибки API лежат отдельно от метрик качества.

CI (`.github/workflows/ci.yml`) на каждом push/PR выполняет `uv sync --locked --extra dev`, затем ruff и pytest.

## Версионирование

Версия берется из `pyproject.toml` (`importlib.metadata`).  
При push в `main` GitHub Actions (`python-semantic-release`) бампит ее по conventional commits:

| Коммит | Бамп |
|--------|------|
| `fix: ...` / `perf: ...` | patch (`0.1.0` -> `0.1.1`) |
| `feat: ...` | minor (`0.1.0` -> `0.2.0`) |
| `BREAKING CHANGE:` в теле / `feat!:` | major (`0.1.0` -> `1.0.0`) |
| `chore:` / `docs:` / `refactor:` / … | без релиза |

Пример:

```bash
git commit -m "feat: extract attributes from description"
git push origin main
```

## Структура

```text
app/
├── api/          # HTTP-роуты
├── core/         # конфигурация (pydantic-settings)
├── schemas/      # Pydantic-модели запросов/ответов
├── services/     # бизнес-логика (extractor, generator, validators)
├── clients/      # LLM-клиент (dependency injection)
└── prompts/      # промпты отдельно от логики
tests/
├── unit/         # unit-тесты сервисов
├── integration/  # API-тесты
└── evals/        # оценочные тесты качества LLM
```
