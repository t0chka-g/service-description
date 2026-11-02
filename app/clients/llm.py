"""Клиент OpenAI API с возможностью подмены в тестах."""

from typing import Protocol


class LLMClient(Protocol):
    """Минимальный контракт LLM-клиента для dependency injection."""

    async def complete(self, prompt: str) -> str:
        """Вернуть текстовый ответ модели на промпт."""
        ...
