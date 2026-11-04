"""Зависимости FastAPI: LLM-клиент создается при первом реальном вызове."""

from typing import Annotated

from fastapi import Depends
from pydantic import BaseModel

from app.clients.llm import LLMClient, OpenAILLMClient
from app.core.config import Settings, settings
from app.services.extractor import CharacteristicExtractor
from app.services.generator import SEOGenerator


class LazyOpenAILLMClient:
    """Откладывает создание клиента до вызова модели.

    Карточка с заведомо неполными данными модель не вызывает, поэтому отсутствие
    ключа не мешает вернуть частичный результат.
    """

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._client: OpenAILLMClient | None = None

    async def complete(
        self,
        prompt: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        if self._client is None:
            self._client = OpenAILLMClient(
                api_key=self._settings.openai_api_key,
                model=self._settings.openai_model,
                base_url=self._settings.openai_base_url,
                max_completion_tokens=self._settings.openai_max_completion_tokens,
            )
        return await self._client.complete(prompt, response_model=response_model)


def get_settings() -> Settings:
    """Вернуть настройки процесса."""

    return settings


def get_llm_client(
    app_settings: Annotated[Settings, Depends(get_settings)],
) -> LLMClient:
    """Вернуть клиент языковой модели для текущего запроса."""

    return LazyOpenAILLMClient(app_settings)


def get_extractor(
    client: Annotated[LLMClient, Depends(get_llm_client)],
) -> CharacteristicExtractor:
    """Собрать извлекатель характеристик с клиентом запроса."""

    return CharacteristicExtractor(client)


def get_generator(
    client: Annotated[LLMClient, Depends(get_llm_client)],
) -> SEOGenerator:
    """Собрать генератор SEO-карточки с клиентом запроса."""

    return SEOGenerator(client)
