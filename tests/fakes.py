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


class ScriptedLLMClient:
    """Ответы по очереди для многошагового сценария: генерация, затем исправление.

    Лишний вызов модели — ошибка теста, а не молчаливый повтор последнего ответа.
    """

    def __init__(self, steps: list[str | dict[str, Any] | Exception]) -> None:
        self._steps = list(steps)
        self.calls: list[str] = []
        self.response_models: list[object] = []

    async def complete(self, prompt: str, *, response_model: object = None) -> str:
        self.calls.append(prompt)
        self.response_models.append(response_model)
        if not self._steps:
            raise AssertionError(f"unexpected LLM call #{len(self.calls)}")
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, dict):
            return json.dumps(step, ensure_ascii=False)
        return step


class RaisingLLMClient:
    """Поднимает заданную ошибку вместо ответа модели."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls: list[str] = []

    async def complete(self, prompt: str, *, response_model: object = None) -> str:
        del response_model
        self.calls.append(prompt)
        raise self._error
