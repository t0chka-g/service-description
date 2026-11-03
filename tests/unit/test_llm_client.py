"""Unit-тесты OpenAI-клиента без сетевых вызовов."""

import pytest
from openai import OpenAIError
from pydantic import BaseModel

from app.clients.llm import LLMError, LLMErrorKind, OpenAILLMClient
from app.schemas.product import DescriptionExtraction


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


class _ProbeModel(BaseModel):
    answer: str = "ok"


def _completion_with(parsed: BaseModel) -> object:
    class _Completion:
        class _Choice:
            class _Message:
                refusal = None

                def __init__(self, body: BaseModel) -> None:
                    self.parsed = body

            def __init__(self, body: BaseModel) -> None:
                self.message = self._Message(body)

        def __init__(self, body: BaseModel) -> None:
            self.choices = [self._Choice(body)]

    return _Completion(parsed)


async def test_complete_forwards_response_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAILLMClient(api_key="test-key", model="gpt-test")
    seen: dict[str, object] = {}
    parsed = _ProbeModel()

    async def _parse(**kwargs: object) -> object:
        seen.update(kwargs)
        return _completion_with(parsed)

    monkeypatch.setattr(client._client.chat.completions, "parse", _parse)

    raw = await client.complete("prompt", response_model=_ProbeModel)

    assert seen["response_format"] is _ProbeModel
    assert raw == parsed.model_dump_json()


async def test_complete_defaults_to_extraction_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAILLMClient(api_key="test-key", model="gpt-test")
    seen: dict[str, object] = {}
    parsed = DescriptionExtraction()

    async def _parse(**kwargs: object) -> object:
        seen.update(kwargs)
        return _completion_with(parsed)

    monkeypatch.setattr(client._client.chat.completions, "parse", _parse)

    await client.complete("prompt")

    assert seen["response_format"] is DescriptionExtraction
