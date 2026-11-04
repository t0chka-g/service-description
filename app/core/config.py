"""Конфигурация приложения."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-6.1-sol"
    openai_base_url: str | None = None
    # Пустое значение не ограничивает ответ. Лимит в токенах, не в символах.
    # У reasoning-моделей в лимит входят токены рассуждения: слишком малое значение
    # обрывает JSON. Для OpenRouter разумно от 3000 вместе с низким reasoning effort.
    openai_max_completion_tokens: int | None = None
    # Один запрос к модели. Reasoning-моделям нужно заметно больше 30 секунд.
    openai_timeout_seconds: float = 90.0
    # None — параметр не отправляется. Часть reasoning-моделей не принимает temperature.
    openai_temperature: float | None = None
    # minimal | low | medium | high. None — параметр не отправляется.
    # Для OpenRouter уходит как reasoning.effort, для OpenAI как reasoning_effort.
    openai_reasoning_effort: str | None = None
    # Список моделей генерации и отдельный судья. Пустые значения не вызывают API.
    eval_generator_models: str = ""
    eval_judge_model: str = ""


settings = Settings()
