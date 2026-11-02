"""Unit-тесты OpenAI-клиента без сетевых вызовов."""

import pytest

from app.clients.llm import LLMError, OpenAILLMClient


async def test_empty_choices_raise_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAILLMClient(api_key="test-key", model="gpt-test")

    class _Completion:
        choices: list[object] = []

    async def _parse(**_kwargs: object) -> _Completion:
        return _Completion()

    monkeypatch.setattr(client._client.chat.completions, "parse", _parse)

    with pytest.raises(LLMError, match="structured response is empty"):
        await client.complete("prompt")
