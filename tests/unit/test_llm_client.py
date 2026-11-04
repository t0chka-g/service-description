"""Unit-тесты OpenAI-клиента без сетевых вызовов."""

import pytest
from openai import OpenAIError
from pydantic import BaseModel

from app.api import dependencies
from app.clients.llm import DEFAULT_TIMEOUT_SECONDS, LLMError, LLMErrorKind, OpenAILLMClient
from app.core.config import Settings
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


async def _seen_request(
    client: OpenAILLMClient, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    seen: dict[str, object] = {}

    async def _parse(**kwargs: object) -> object:
        seen.update(kwargs)
        return _completion_with(_ProbeModel())

    monkeypatch.setattr(client._client.chat.completions, "parse", _parse)
    await client.complete("prompt", response_model=_ProbeModel)
    return seen


async def test_openrouter_request_carries_configured_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAILLMClient(
        api_key="test-key",
        model="vendor/model",
        base_url="https://openrouter.ai/api/v1",
        timeout=75.0,
        temperature=0.3,
        max_completion_tokens=3000,
        reasoning_effort="low",
    )

    seen = await _seen_request(client, monkeypatch)

    assert seen["timeout"] == 75.0
    assert seen["temperature"] == 0.3
    assert seen["max_completion_tokens"] == 3000
    assert seen["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "reasoning_effort" not in seen


async def test_direct_openai_uses_reasoning_effort_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAILLMClient(api_key="test-key", model="gpt-test", reasoning_effort="minimal")

    seen = await _seen_request(client, monkeypatch)

    assert seen["reasoning_effort"] == "minimal"
    assert "extra_body" not in seen


async def test_unset_options_are_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAILLMClient(api_key="test-key", model="gpt-test")

    seen = await _seen_request(client, monkeypatch)

    for name in ("temperature", "max_completion_tokens", "extra_body", "reasoning_effort"):
        assert name not in seen
    assert seen["timeout"] == DEFAULT_TIMEOUT_SECONDS


def test_default_timeout_fits_reasoning_models() -> None:
    assert DEFAULT_TIMEOUT_SECONDS >= 60.0
    assert Settings(_env_file=None).openai_timeout_seconds >= 60.0


async def test_lazy_client_passes_settings_to_provider_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Spy:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def complete(self, prompt: str, *, response_model: object = None) -> str:
            del prompt, response_model
            return "{}"

    monkeypatch.setattr(dependencies, "OpenAILLMClient", _Spy)
    app_settings = Settings(
        _env_file=None,
        openai_api_key="key",
        openai_model="vendor/model",
        openai_base_url="https://openrouter.ai/api/v1",
        openai_timeout_seconds=120.0,
        openai_temperature=0.2,
        openai_max_completion_tokens=4096,
        openai_reasoning_effort="low",
    )

    await dependencies.LazyOpenAILLMClient(app_settings).complete("prompt")

    assert captured["timeout"] == 120.0
    assert captured["temperature"] == 0.2
    assert captured["max_completion_tokens"] == 4096
    assert captured["reasoning_effort"] == "low"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
