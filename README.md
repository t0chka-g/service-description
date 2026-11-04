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

## Настройка модели

| Переменная | Назначение | Рекомендация |
| --- | --- | --- |
| `OPENAI_MODEL`, `OPENAI_BASE_URL` | Модель и endpoint (OpenRouter) | `openai/gpt-4.1-mini` и `https://openrouter.ai/api/v1` |
| `OPENAI_TIMEOUT_SECONDS` | Таймаут одного запроса | 90 по умолчанию |
| `OPENAI_MAX_COMPLETION_TOKENS` | Лимит токенов ответа | 3000-4096, слишком малый лимит обрезает JSON |
| `OPENAI_REASONING_EFFORT` | Усилие рассуждения reasoning-модели | `low` для скорости |
| `OPENAI_TEMPERATURE` | Температура | пусто, если модель ее не принимает |

На OpenRouter усилие рассуждения уходит как `reasoning.effort`, на остальных endpoint - как `reasoning_effort`.

## Как строится SEO-карточка

1. Основной запрос к модели возвращает заголовок, описание, 3-5 запросов и коммерческий блок.
2. Ответ проходит проверку фактов: числа, единицы, привязка значений к параметрам, выдуманные характеристики, гарантии, сертификаты, отрасли, служебный текст, повторы.
3. Описание длиннее 1200 знаков обрезается по границе предложения без обращения к модели.
4. Описание короче 800 знаков или отклоненное проверкой исправляется одним дополнительным запросом, который переписывает только описание.
5. Если и после этого текст не годится, сервис возвращает `needs_review` с `seo_description: null` и причиной в `warnings`. Ответ `200`, не `502`.

Больше двух обращений к модели на карточку не бывает. LLM Judge в обычной генерации не вызывается.

Публикуемые поля: `seo_title`, `seo_description`, `specifications`, `search_queries`, `commercial_block`. Диагностика лежит отдельно: `status`, `warnings`, `missing_fields`. Если текста нет, поле равно `null`, а не сообщению об ошибке.

| Статус | Что значит |
| --- | --- |
| `complete` | Заголовок до 70 знаков, описание 800-1200, 3-5 запросов, коммерческий блок |
| `needs_review` | Часть полей не прошла проверку, подтвержденные характеристики сохранены |
| `insufficient_data` | Подтвержденных данных мало, модель не вызывалась |

## Ручной E2E через OpenRouter

Автотесты не обращаются к сети. Реальные запросы платные, вы запускаете их сами.

Готовые запросы лежат в `examples/`: `manometer.json` (сценарий A), `temperature_sensor.json` (сценарий B) и `thermometer_insufficient.json` (мало данных, ожидается `insufficient_data`).

**Swagger.** Запустите сервис с ключом, откройте `http://127.0.0.1:8000/docs`, выберите `POST /api/v1/products/generate`, в списке примеров запроса выберите `manometer` или `temperature_sensor` и нажмите Execute. Ожидается `status: complete`, `seo_title` до 70 знаков, `seo_description` 800-1200 знаков, 3-5 запросов в `search_queries`.

**PowerShell.**

```powershell
$env:OPENAI_API_KEY = "<openrouter-key>"
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
$env:OPENAI_MODEL = "openai/gpt-4.1-mini"
$env:OPENAI_MAX_COMPLETION_TOKENS = "4096"
$env:OPENAI_REASONING_EFFORT = "low"
uv run uvicorn app.main:app

# в другом окне
curl.exe -X POST http://127.0.0.1:8000/api/v1/products/generate `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@examples/manometer.json"
```

**Скрипт с проверкой контракта** (оба сценария, код выхода 1 при нарушении):

```powershell
$env:SMOKE_OPENROUTER = "I_UNDERSTAND_THIS_IS_PAID"
uv run python scripts/openrouter_smoke.py
```

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
