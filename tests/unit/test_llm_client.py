"""Unit-тесты OpenAI-клиента без сетевых вызовов."""

import pytest
from openai import OpenAIError

from app.clients.llm import LLMError, LLMErrorKind, OpenAILLMClient


async def test_empty_choices_raise_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAILLMClient(api_key="test-key", model="gpt-test")

    class _Completion:
        choices: list[object] = []

    async def _parse(**_kwargs: object) -> _Completion:
        return _Completion()

    monkeypatch.setattr(client._client.chat.completions, "parse", _parse)

    with pytest.raises(LLMError, match="structured response is empty") as caught:
        await client.complete("prompt")

    assert caught.value.kind is LLMErrorKind.MODEL


async def test_openai_error_is_api_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAILLMClient(api_key="test-key", model="gpt-test")

    async def _parse(**_kwargs: object) -> None:
        raise OpenAIError("boom")

    monkeypatch.setattr(client._client.chat.completions, "parse", _parse)

    with pytest.raises(LLMError, match="OpenAI request failed") as caught:
        await client.complete("prompt")

    assert caught.value.kind is LLMErrorKind.API
