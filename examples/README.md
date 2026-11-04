# Демонстрационные примеры запросов

Файлы в этом каталоге — **синтетические учебные данные**, не паспорта реальных приборов.

| Файл | Назначение |
|------|------------|
| `manometer.json` | Манометр, сценарий полной карточки (smoke / E2E) |
| `manometer.request.json` | Тот же сценарий манометра (alias для docs) |
| `temperature_sensor.json` | Другой измерительный прибор |
| `thermometer_insufficient.json` | Мало данных → `insufficient_data` |
| `thermometer.request.json` | Короткий сценарий термометра |

Часть примеров совпадает с OpenAPI (`app/api/examples.py`).

## Как получить ответ

Готовых проверенных ответов реальной LLM в репозитории нет — не копируйте синтетические OpenAPI-примеры ответа как результат продакшен-прогона.

1. Заполните `.env` (`OPENAI_API_KEY`, при необходимости модель и timeout/reasoning).
2. Запустите сервис локально или через Docker Compose.
3. Откройте Swagger UI: <http://127.0.0.1:8000/docs>
4. Выберите `POST /api/v1/products/generate` и пример запроса, либо:

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/v1/products/generate" \
  -H "Content-Type: application/json" \
  --data-binary @examples/manometer.json
```

Вызов расходует квоту LLM-провайдера.

Платный smoke-скрипт:

```powershell
$env:SMOKE_OPENROUTER = "I_UNDERSTAND_THIS_IS_PAID"
uv run python scripts/openrouter_smoke.py
```
