# Архитектура AI Product Card Generator

Документ описывает фактически реализованный конвейер сервиса.
Детали внутренней политики SEO Generator (обрезка, fail-soft, `needs_review`) могут дорабатываться отдельно — ниже зафиксирован стабильный внешний контракт HTTP API.

## Component Diagram

```mermaid
flowchart LR
  UI["Web UI<br/>GET / + /static"]
  API["FastAPI<br/>app.main"]
  EXT["CharacteristicExtractor"]
  VAL["validators<br/>supported_quote / same_technical_value"]
  GEN["SEOGenerator"]
  RESP["GenerationResult<br/>JSON"]
  LLM["OpenRouter / OpenAI-compatible<br/>LLM API"]

  UI --> API
  API --> EXT
  EXT --> VAL
  EXT --> GEN
  GEN --> RESP
  API --> RESP

  EXT -.->|извлечение из description| LLM
  GEN -.->|генерация SEO текста<br/>только при открытом шлюзе| LLM
```

Слои приложения:

| Слой | Путь | Роль |
|------|------|------|
| HTTP | `app/api/` | маршруты, ошибки, OpenAPI-примеры |
| Config | `app/core/` | `pydantic-settings`, переменные окружения |
| Schemas | `app/schemas/` | контракты запроса/ответа |
| Services | `app/services/` | Extractor, Generator, validators |
| Clients | `app/clients/` | LLM-клиент (DI) |
| Prompts | `app/prompts/` | тексты промптов вне бизнес-логики |
| UI | `app/web/`, `app/static/` | страница контент-менеджера |

## Sequence Diagram

Канонические диаграммы запроса, grounding и статусов — в `README.md`. Ниже та же развилка, что в коде маршрута и `SEOGenerator`.

Неразобранный JSON **извлечения** — это `LLMError` и HTTP **502**. Неразобранный JSON **SEO черновика** — HTTP **200** и `needs_review` (`generation_failed`), без второго вызова. Повторный запрос бывает только как repair описания и только если отказ не `ungrounded`. Ошибка repair не поднимается в 502/503/504.

```mermaid
sequenceDiagram
  actor Client
  participant API as FastAPI routes
  participant Ext as CharacteristicExtractor
  participant Val as validators
  participant LLM as OpenRouter LLM API
  participant Gen as SEOGenerator

  Client->>API: POST /api/v1/products/generate
  alt тело не соответствует схеме
    API-->>Client: 422
  else схема валидна
    API->>API: нормализация полей
    API->>Ext: extract
    alt description пустое или нет полей
      Ext-->>API: attributes из specs
    else есть description и поля
      Ext->>LLM: промпт извлечения
      alt таймаут / API / JSON не разобран
        API-->>Client: 504 / 503 / 502
      else JSON извлечения разобран
        Ext->>Val: supported_quote, same_technical_value
        Ext-->>API: attributes, missing_fields, conflicts
      end
    end
    break ошибка извлечения
      Note over API: generate не вызывается
    end
    API->>Gen: generate
    alt шлюз закрыт
      Gen-->>API: 200 insufficient_data
    else шлюз открыт
      Gen->>LLM: промпт SEO карточки
      alt таймаут или сбой API
        API-->>Client: 504 или 503
      else сбой контракта вызова
        API-->>Client: 502
      else JSON черновика не разобран
        Gen-->>API: 200 needs_review
      else черновик разобран
        opt отказ описания не ungrounded
          Gen->>LLM: один repair описания
        end
        alt поля приняты
          Gen-->>API: 200 complete
        else описание или запросы не прошли
          Gen-->>API: 200 needs_review
        end
      end
    end
    API-->>Client: GenerationResult
  end
```

## Error Handling

| Ситуация | HTTP | Тело / статус | Источник в коде |
|----------|------|---------------|-----------------|
| Невалидное тело запроса | **422** | стандартный `ValidationError` FastAPI/Pydantic | схема `ProductInput` |
| Мало подтвержденных данных | **200** | `status: "insufficient_data"`, `warnings` | шлюз Generator; LLM не вызывается |
| Текст/поля не прошли проверку фактов | **200** | `status: "needs_review"`; `seo_description` / `commercial_block` могут быть `null` | политика Generator + grounding |
| JSON извлечения не разобран или сбой контракта вызова | **502** | `{"detail":"Модель вернула ответ, который не удалось принять."}` | `LLMErrorKind.MODEL` -> `llm_http_exception` |
| JSON SEO черновика не разобран | **200** | `status: "needs_review"`, код `generation_failed` | `SEOGenerator._failed_result`; повторного вызова нет |
| Провайдер LLM недоступен / ключ пуст при реальном вызове | **503** | `{"detail":"Сервис языковой модели временно недоступен."}` | `LLMErrorKind.API` |
| Таймаут ответа провайдера | **504** | `{"detail":"Превышено время ожидания ответа языковой модели."}` | `TimeoutError` / `APITimeoutError` |
| Успешная полная карточка | **200** | `status: "complete"` | Generator |

Важно:

- `insufficient_data` и `needs_review` — **не** ошибки HTTP: это результаты с кодом 200.
- Не больше двух обращений к модели на одну карточку (основной ответ + опциональный rewrite описания).
- Служебные проверки: `GET /health` -> `{"status":"ok"}`.
- Сообщения `ApiError.detail` не содержат стека, ключей и сырого текста провайдера.
