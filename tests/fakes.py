"""Fake-зависимости для unit-тестов без реальных LLM-вызовов."""

import json
from typing import Any


class FakeLLMClient:
    """Возвращает заранее заданный ответ вместо вызова OpenAI."""

    def __init__(self, response: str | dict[str, Any]) -> None:
        if isinstance(response, dict):
            self._response = json.dumps(response, ensure_ascii=False)
        else:
            self._response = response
        self.calls: list[str] = []

    async def complete(self, prompt: str, *, response_model: object = None) -> str:
        del response_model
        self.calls.append(prompt)
        return self._response


class RaisingLLMClient:
    """Поднимает заданную ошибку вместо ответа модели."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls: list[str] = []

    async def complete(self, prompt: str, *, response_model: object = None) -> str:
        del response_model
        self.calls.append(prompt)
        raise self._error
