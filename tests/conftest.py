"""Общие фикстуры для тестирования."""

import json
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from tests.fakes import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
async def client() -> AsyncClient:
    """HTTP-клиент для integration-тестов API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="session")
def extraction_cases() -> list[dict[str, Any]]:
    """Загрузить датасет сценариев извлечения характеристик."""
    path = FIXTURES_DIR / "extraction_cases.json"
    with path.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def extraction_cases_by_id(extraction_cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Индекс сценариев извлечения по id."""
    return {case["id"]: case for case in extraction_cases}


@pytest.fixture
def fake_llm_factory():
    """Фабрика Fake LLM с ответом из сценария или явным payload."""

    def _factory(case_or_response: dict[str, Any] | str) -> FakeLLMClient:
        if isinstance(case_or_response, str):
            return FakeLLMClient(case_or_response)
        if "llm_response" in case_or_response:
            return FakeLLMClient(case_or_response["llm_response"])
        return FakeLLMClient(case_or_response)

    return _factory
