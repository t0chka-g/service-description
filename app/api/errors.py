"""Публичные HTTP-ошибки языковой модели."""

from typing import assert_never

from fastapi import HTTPException
from openai import APITimeoutError
from pydantic import BaseModel, Field

from app.clients.llm import LLMError, LLMErrorKind

BAD_RESPONSE_DETAIL = "Модель вернула ответ, который не удалось принять."
UNAVAILABLE_DETAIL = "Сервис языковой модели временно недоступен."
TIMEOUT_DETAIL = "Превышено время ожидания ответа языковой модели."


class ApiError(BaseModel):
    """Ответ об ошибке без стека, ключей и текста провайдера."""

    detail: str = Field(description="Что произошло, без внутренних деталей.")


def llm_http_exception(exc: LLMError) -> HTTPException:
    """Сопоставить статус с типом сбоя, а не с текстом сообщения."""

    if _is_timeout(exc):
        return HTTPException(status_code=504, detail=TIMEOUT_DETAIL)
    if exc.kind is LLMErrorKind.MODEL:
        return HTTPException(status_code=502, detail=BAD_RESPONSE_DETAIL)
    if exc.kind is LLMErrorKind.API:
        return HTTPException(status_code=503, detail=UNAVAILABLE_DETAIL)
    assert_never(exc.kind)


def _is_timeout(exc: BaseException) -> bool:
    cause = exc.__cause__
    return isinstance(exc, (TimeoutError, APITimeoutError)) or isinstance(
        cause, (TimeoutError, APITimeoutError)
    )
