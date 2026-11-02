"""Клиент OpenAI API с возможностью подмены в тестах."""

from enum import StrEnum
from typing import Protocol

from openai import AsyncOpenAI, OpenAIError

from app.schemas.product import DescriptionExtraction

DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMErrorKind(StrEnum):
    """Технический сбой API отдельно от ответа, который не удалось разобрать."""

    API = "api"
    MODEL = "model"


class LLMError(Exception):
    """Ошибка вызова модели или разбора ее ответа."""

    def __init__(self, message: str, *, kind: LLMErrorKind = LLMErrorKind.MODEL) -> None:
        super().__init__(message)
        self.kind = kind


class LLMClient(Protocol):
    """Минимальный контракт LLM-клиента для dependency injection."""

    async def complete(self, prompt: str) -> str:
        """Вернуть текстовый ответ модели на промпт."""
        ...


class OpenAILLMClient:
    """Минимальный вызов OpenAI со Structured Outputs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key.strip():
            raise LLMError("OPENAI_API_KEY is empty", kind=LLMErrorKind.API)
        if not model.strip():
            raise LLMError("OPENAI_MODEL is empty", kind=LLMErrorKind.API)
        self._model = model
        self._timeout = timeout
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )

    async def complete(self, prompt: str) -> str:
        try:
            completion = await self._client.chat.completions.parse(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format=DescriptionExtraction,
                timeout=self._timeout,
            )
        except OpenAIError as exc:
            raise LLMError("OpenAI request failed", kind=LLMErrorKind.API) from exc

        if not completion.choices:
            raise LLMError("structured response is empty")
        choice = completion.choices[0].message
        if choice.refusal:
            raise LLMError("model refused the extraction request")
        if choice.parsed is None:
            raise LLMError("structured response is empty")
        return choice.parsed.model_dump_json()
