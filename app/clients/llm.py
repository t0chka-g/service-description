"""Клиент OpenAI API с возможностью подмены в тестах."""

from enum import StrEnum
from typing import Protocol, TypeVar

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict

from app.schemas.product import DescriptionExtraction

ModelT = TypeVar("ModelT", bound=BaseModel)

# Запас для reasoning-моделей: ответ со скрытым рассуждением идет дольше обычного.
DEFAULT_TIMEOUT_SECONDS = 90.0


class LLMErrorKind(StrEnum):
    """Технический сбой API отдельно от ответа, который не удалось разобрать."""

    API = "api"
    MODEL = "model"


class LLMError(Exception):
    """Ошибка вызова модели или разбора ее ответа."""

    def __init__(self, message: str, *, kind: LLMErrorKind = LLMErrorKind.MODEL) -> None:
        super().__init__(message)
        self.kind = kind


class LLMUsage(BaseModel):
    """Фактический usage ответа. Стоимость по нему считает evaluation, не клиент."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int | None = None


class LLMCallMeta(BaseModel):
    """Метаданные последнего вызова. Появляются только если провайдер их вернул."""

    model_config = ConfigDict(extra="ignore")

    response_model: str | None = None
    usage: LLMUsage | None = None


class LLMClient(Protocol):
    """Минимальный контракт LLM-клиента для dependency injection."""

    async def complete(
        self,
        prompt: str,
        *,
        response_model: type[ModelT] | None = None,
    ) -> str:
        """Вернуть JSON-ответ модели. Без схемы сохраняется извлечение характеристик."""
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
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMError("OPENAI_API_KEY is empty", kind=LLMErrorKind.API)
        if not model.strip():
            raise LLMError("OPENAI_MODEL is empty", kind=LLMErrorKind.API)
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._max_completion_tokens = max_completion_tokens
        self._reasoning_effort = (reasoning_effort or "").strip() or None
        self._openrouter = "openrouter.ai" in (base_url or "")
        self.last_call: LLMCallMeta | None = None
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )

    async def complete(
        self,
        prompt: str,
        *,
        response_model: type[ModelT] | None = None,
    ) -> str:
        schema: type[BaseModel] = response_model or DescriptionExtraction
        request: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": schema,
            "timeout": self._timeout,
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._max_completion_tokens is not None:
            request["max_completion_tokens"] = self._max_completion_tokens
        if self._reasoning_effort is not None:
            if self._openrouter:
                request["extra_body"] = {"reasoning": {"effort": self._reasoning_effort}}
            else:
                request["reasoning_effort"] = self._reasoning_effort
        try:
            completion = await self._client.chat.completions.parse(**request)
        except OpenAIError as exc:
            raise LLMError("OpenAI request failed", kind=LLMErrorKind.API) from exc

        self.last_call = _call_meta(completion)
        if not completion.choices:
            raise LLMError("structured response is empty")
        choice = completion.choices[0].message
        if choice.refusal:
            raise LLMError("model refused the extraction request")
        if choice.parsed is None:
            raise LLMError("structured response is empty")
        return choice.parsed.model_dump_json()


def _call_meta(completion: object) -> LLMCallMeta:
    usage = getattr(completion, "usage", None)
    parsed_usage: LLMUsage | None = None
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total = getattr(usage, "total_tokens", None)
            parsed_usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total if isinstance(total, int) else None,
            )
    response_model = getattr(completion, "model", None)
    return LLMCallMeta(
        response_model=response_model if isinstance(response_model, str) else None,
        usage=parsed_usage,
    )
