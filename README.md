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

Реальный прогон оценки извлечения вызывает OpenAI и пишет отчет в `reports/evals/` (каталог в `.gitignore`):

```bash
uv run pytest -m llm -v
```

Без `OPENAI_API_KEY` этот тест пропускается. Клиент берет `OPENAI_MODEL` и `OPENAI_BASE_URL` из настроек. Precision, Recall и F1 в отчете относятся к характеристикам из описания; копирование `specs` в них не входит. Hallucination Rate считается по сырому ответу модели. Ошибки API лежат отдельно от метрик качества.

## Оценка извлечения, 3 ноября 2026

Один прогон на OpenRouter (`https://openrouter.ai/api/v1`), 16 сценариев из `tests/fixtures/extraction_cases.json`. Ошибок API не было. Хеш промпта `7f61e263c8b2`. Сохранение `specs` у всех моделей 12/12. Полные JSON-отчеты лежат локально в `reports/evals/`.

| Модель | F1 | Precision | Recall | Hallucination | Конфликты F1 | Числа |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `z-ai/glm-5.3-flash` | 1.00 | 1.00 | 1.00 | 0 | 1.00 | 1.00 |
| `deepseek/deepseek-v4.1-flash` | 1.00 | 1.00 | 1.00 | 0 | 1.00 | 1.00 |
| `xiaomi/mimo-v2.6-flash` | 0.94 | 0.94 | 0.94 | 0.04 | 0.50 | 0.94 |

GLM и DeepSeek совпали с разметкой по всем 17 парам из описания и по обоим конфликтам. В сырых ответах неподтвержденных цитат не было.

У `xiaomi/mimo-v2.6-flash` две ошибки:

- `empty_specs_rich_description`: вместо `Ду50` модель вернула `50`. Значение есть в тексте, поэтому это неверная пара, а не галлюцинация.
- `spec_key_case_mismatch`: конфликт `1,5` / `2,5` не зафиксирован. Модель процитировала описание целиком, и Extractor такую цитату не принял.

Это результат одного прогона на 16 сценариях. Повторение может дать другие ответы: клиент не фиксирует temperature.

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
