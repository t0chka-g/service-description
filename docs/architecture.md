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
  GEN -.->|генерация SEO-текста<br/>только при открытом шлюзе| LLM
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

Обработка `POST /api/v1/products/generate`:

```mermaid
sequenceDiagram
  actor Client
  participant API as FastAPI routes
  participant Ext as CharacteristicExtractor
  participant Val as validators
  participant LLM as OpenRouter LLM API
  participant Gen as SEOGenerator

  Client->>API: POST /api/v1/products/generate<br/>ProductInput
  alt тело не соответствует схеме
    API-->>Client: 422 Validation Error
  else схема валидна
    API->>API: нормализация name/manufacturer/description/specs
    API->>Ext: extract(prepared, fields=keys(specs))
    alt description непустое и есть поля
      Ext->>LLM: complete(extraction prompt)
      LLM-->>Ext: JSON атрибутов + evidence
      Ext->>Val: supported_quote / same_technical_value
    end
    Ext-->>API: attributes, missing_fields, conflicts
    API->>Gen: generate(GenerationInput)
    alt недостаточно подтвержденных данных
      Note over Gen: шлюз закрыт, LLM не вызывается
      Gen-->>API: 200 GenerationResult<br/>status=insufficient_data
    else данных достаточно для полного пути
      Gen->>LLM: complete(generation prompt)
      alt таймаут провайдера
        LLM-->>Gen: timeout
        Gen-->>API: LLMError / Timeout
        API-->>Client: 504
      else сбой API провайдера
        LLM-->>Gen: API error
        Gen-->>API: LLMError kind=api
        API-->>Client: 503
      else сырой ответ не разобрать / сбой контракта вызова
        LLM-->>Gen: unparseable / transport contract error
        Gen-->>API: LLMError kind=model
        API-->>Client: 502
      else факты не прошли / текст не готов
        Gen-->>API: 200 GenerationResult<br/>status=needs_review<br/>seo_description может быть null
      else черновик принят
        Gen-->>API: 200 GenerationResult<br/>status=complete
      end
    end
    API-->>Client: GenerationResult / ApiError
  end
```

## Error Handling

| Ситуация | HTTP | Тело / статус | Источник в коде |
|----------|------|---------------|-----------------|
| Невалидное тело запроса | **422** | стандартный `ValidationError` FastAPI/Pydantic | схема `ProductInput` |
| Мало подтвержденных данных | **200** | `status: "insufficient_data"`, `warnings` | шлюз Generator; LLM не вызывается |
| Текст/поля не прошли проверку фактов | **200** | `status: "needs_review"`; `seo_description` / `commercial_block` могут быть `null` | политика Generator + grounding |
| Ответ LLM не принят на уровне вызова/разбора | **502** | `{"detail":"Модель вернула ответ, который не удалось принять."}` | `LLMErrorKind.MODEL` → `llm_http_exception` |
| Провайдер LLM недоступен / ключ пуст при реальном вызове | **503** | `{"detail":"Сервис языковой модели временно недоступен."}` | `LLMErrorKind.API` |
| Таймаут ответа провайдера | **504** | `{"detail":"Превышено время ожидания ответа языковой модели."}` | `TimeoutError` / `APITimeoutError` |
| Успешная полная карточка | **200** | `status: "complete"` | Generator |

Важно:

- `insufficient_data` и `needs_review` — **не** ошибки HTTP: это результаты с кодом 200.
- Не больше двух обращений к модели на одну карточку (основной ответ + опциональный rewrite описания).
- Служебные проверки: `GET /health` → `{"status":"ok"}`.
- Сообщения `ApiError.detail` не содержат стека, ключей и сырого текста провайдера.
