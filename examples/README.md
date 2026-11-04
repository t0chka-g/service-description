# Демонстрационные примеры запросов

Файлы в этом каталоге — **синтетические учебные данные**, не паспорта реальных приборов.
Они совпадают с OpenAPI-примерами из `app/api/examples.py`.

| Файл | Назначение |
|------|------------|
| `manometer.request.json` | Манометр с несколькими характеристиками (сценарий полной карточки) |
| `thermometer.request.json` | Термометр с одной характеристикой (сценарий `insufficient_data`) |

## Как получить ответ

Готовых проверенных ответов реальной LLM в репозитории нет — не копируйте синтетические OpenAPI-примеры ответа как результат продакшен-прогона.

1. Заполните `.env` (`OPENAI_API_KEY`, при необходимости `OPENAI_BASE_URL` / `OPENAI_MODEL`).
2. Запустите сервис локально или через Docker Compose.
3. Откройте Swagger UI: <http://127.0.0.1:8000/docs>
4. Выберите `POST /api/v1/products/generate` и один из примеров запроса (`manometer` / `thermometer`), либо отправьте файл:

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/v1/products/generate" \
  -H "Content-Type: application/json" \
  --data-binary @examples/manometer.request.json
```

Вызов расходует квоту LLM-провайдера.
