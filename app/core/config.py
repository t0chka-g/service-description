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
    # Список моделей генерации и отдельный судья. Пустые значения не вызывают API.
    eval_generator_models: str = ""
    eval_judge_model: str = ""


settings = Settings()
