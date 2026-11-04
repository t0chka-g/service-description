"""Интеграционные тесты HTTP-слоя без сетевых вызовов LLM."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx import AsyncClient
from openai import APITimeoutError

from app.clients.llm import LLMError, LLMErrorKind
from app.main import app
from app.schemas.product import (
    SEO_DESCRIPTION_MIN_LENGTH,
    SEO_TITLE_MAX_LENGTH,
    GenerationResult,
)

API_ROOT = Path(__file__).resolve().parents[2] / "app" / "api"
GENERATE_PATH = "/api/v1/products/generate"
BAD_RESPONSE = "Модель вернула ответ, который не удалось принять."
UNAVAILABLE = "Сервис языковой модели временно недоступен."
TIMEOUT = "Превышено время ожидания ответа языковой модели."

_NAME = "Манометр демонстрационный МП-Демо-01"
_MANUFACTURER = "Учебный стенд «Сигнал»"
_SPECS = {
    "Диапазон показаний": "0...1,6 МПа",
    "Класс точности": "1,5",
    "Присоединение": "M20x1,5",
    "Степень защиты": "IP54",
}
_PAD = " Сведения приведены только по подтвержденным характеристикам."


class ScriptedLLM:
    """Ответы или ошибки по очереди. Сеть не используется."""

    def __init__(self, steps: list[str | Exception]) -> None:
        self._steps = list(steps)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, response_model: object = None) -> str:
        del response_model
        self.prompts.append(prompt)
        if not self._steps:
            raise AssertionError("unexpected LLM call")
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class RejectingLLM:
    """Падает, если генерация обратилась к модели."""

    async def complete(self, prompt: str, *, response_model: object = None) -> str:
        del prompt, response_model
        raise AssertionError("LLM must not be called")


@pytest.fixture
def install_llm():
    """Подменить LLM-клиент на время одного теста."""

    def _install(client: object) -> None:
        from app.api.dependencies import get_llm_client

        app.dependency_overrides[get_llm_client] = lambda: client

    yield _install
    app.dependency_overrides.clear()


def _product(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": _NAME,
        "manufacturer": _MANUFACTURER,
        "description": "",
        "specs": dict(_SPECS),
    }
    payload.update(overrides)
    return payload


def _grounded_generation(specs: dict[str, str]) -> str:
    """Черновик полной карточки, заземленный на переданных характеристиках."""

    confirmed = [(name, value) for name, value in specs.items() if value.strip()]
    parts = [_NAME, f"Производитель: {_MANUFACTURER}."]
    parts.extend(f"{name}: {value}." for name, value in confirmed)
    description = " ".join(parts)
    while len(description) < 850:
        description += _PAD
    assert SEO_DESCRIPTION_MIN_LENGTH <= len(description) <= 1200
    first_name, first_value = confirmed[0]
    draft = {
        "status": "complete",
        "seo_title": _NAME,
        "seo_description": description,
        "search_queries": [
            f"{_NAME} купить",
            f"{_NAME} характеристики",
            f"{_NAME} цена",
        ],
        "commercial_block": (
            f"{_NAME}: {first_name} {first_value}. Производитель: {_MANUFACTURER}."
        ),
        "specifications": [
            {
                "name": "погрешность",
                "value": "0,01 %",
                "source": "description",
                "evidence": "выдумано",
            }
        ],
        "missing_fields": ["выдуманное поле"],
        "warnings": [{"code": "insufficient_data", "field": "чужое"}],
    }
    assert len(draft["seo_title"]) <= SEO_TITLE_MAX_LENGTH
    return json.dumps(draft, ensure_ascii=False)


def _assert_public_error(response: httpx.Response, detail: str) -> None:
    assert response.json() == {"detail": detail}
    text = response.text.casefold()
    assert "traceback" not in text
    assert "openai" not in text
    assert "api_key" not in text
    assert "sk-" not in text


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_index_returns_application_page(client: AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    page = response.text
    assert "Сгенерировать карточку" in page
    assert "/static/js/app.js" in page
    assert "/static/css/styles.css" in page
    assert "OPENAI_API_KEY" not in page
    assert "sk-" not in page


async def test_static_assets_are_served(client: AsyncClient) -> None:
    styles = await client.get("/static/css/styles.css")
    script = await client.get("/static/js/app.js")

    assert styles.status_code == 200
    assert "text/css" in styles.headers["content-type"]
    assert script.status_code == 200
    assert GENERATE_PATH in script.text
    assert "Генерируем карточку товара" in script.text
    assert "innerHTML" not in script.text
    assert "insertAdjacentHTML" not in script.text


async def test_docs_and_openapi_schema(client: AsyncClient) -> None:
    from importlib.metadata import version

    docs = await client.get("/docs")
    redoc = await client.get("/redoc")
    schema_response = await client.get("/openapi.json")

    assert docs.status_code == 200
    assert "swagger" in docs.text.casefold()
    assert redoc.status_code == 200
    assert schema_response.status_code == 200
    schema = schema_response.json()
    assert schema["info"]["title"] == "AI Product Card Generator"
    assert schema["info"]["version"] == version("service-description")
    assert schema["info"]["description"]
    operation = schema["paths"][GENERATE_PATH]["post"]
    assert "products" in operation["tags"]
    assert operation["responses"]["200"]
    assert operation["responses"]["422"]
    assert operation["responses"]["502"]
    assert operation["responses"]["503"]
    assert operation["responses"]["504"]
    encoded = json.dumps(schema, ensure_ascii=False)
    assert "МП-Демо-01" in encoded
    assert "ТС-Демо-02" in encoded
    assert "не характеристики реального прибора" in encoded


def test_generation_api_does_not_import_evaluation() -> None:
    sources = [path.read_text(encoding="utf-8") for path in API_ROOT.glob("*.py")]
    sources.append((API_ROOT.parent / "main.py").read_text(encoding="utf-8"))
    combined = "\n".join(sources)
    lowered = combined.casefold()
    assert "app.evals" not in combined
    assert "judge" not in lowered
    assert "generation_runner" not in combined
    assert "evaluation runner" not in lowered


async def test_generate_returns_confirmed_card(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    script = ScriptedLLM([_grounded_generation(_SPECS)])
    install_llm(script)
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["seo_title"] == _NAME
    assert [(item["name"], item["value"], item["source"]) for item in body["specifications"]] == [
        (name, value, "specs") for name, value in _SPECS.items()
    ]
    assert body["missing_fields"] == []
    assert "0,01 %" not in response.text
    assert "выдуманное поле" not in response.text
    assert script.prompts
    assert script.prompts[0].startswith("Собери SEO-карточку")
    assert all(not prompt.startswith("Извлеки") for prompt in script.prompts)


async def test_generate_calls_extractor_before_generator(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    script = ScriptedLLM(
        [
            json.dumps({"attributes": []}, ensure_ascii=False),
            _grounded_generation(_SPECS),
        ]
    )
    install_llm(script)
    response = await client.post(
        GENERATE_PATH,
        json=_product(description="Учебный пример для проверки порядка вызовов."),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert len(script.prompts) == 2
    assert script.prompts[0].startswith("Извлеки технические характеристики")
    assert script.prompts[1].startswith("Собери SEO-карточку")


async def test_missing_spec_value_is_reported(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    specs = dict(_SPECS)
    specs["Степень защиты"] = ""
    install_llm(ScriptedLLM([_grounded_generation(specs)]))
    response = await client.post(GENERATE_PATH, json=_product(specs=specs))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["missing_fields"] == ["Степень защиты"]
    assert {
        "code": "missing_field",
        "field": "Степень защиты",
        "message": "",
    } in body["warnings"]
    names = [item["name"] for item in body["specifications"]]
    assert "Степень защиты" not in names
    assert "Диапазон показаний" in names


async def test_omitted_fields_return_insufficient_data_without_llm(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    install_llm(RejectingLLM())
    response = await client.post(GENERATE_PATH, json={})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_data"
    assert body["seo_title"] == "Товар"
    assert body["search_queries"] == []
    assert len(body["seo_description"]) < SEO_DESCRIPTION_MIN_LENGTH
    assert any(item["code"] == "insufficient_data" for item in body["warnings"])


async def test_sparse_product_is_insufficient_data(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    install_llm(RejectingLLM())
    response = await client.post(
        GENERATE_PATH,
        json={
            "name": "Термометр демонстрационный ТС-Демо-02",
            "manufacturer": _MANUFACTURER,
            "specs": {"Диапазон измерений": "0...100 °C"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_data"
    assert body["specifications"][0]["value"] == "0...100 °C"
    assert body["specifications"][0]["source"] == "specs"
    assert any(item["code"] == "insufficient_data" for item in body["warnings"])


async def test_invalid_payload_returns_422(client: AsyncClient) -> None:
    response = await client.post(GENERATE_PATH, json={"name": 15, "specs": ["диапазон"]})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert "traceback" not in response.text.casefold()


async def test_model_error_returns_partial_card(client: AsyncClient, install_llm: Any) -> None:
    install_llm(ScriptedLLM(["{"]))
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 200
    card = _assert_matches_response_model(response.json())
    assert card.status.value == "needs_review"
    assert [item.value for item in card.specifications] == list(_SPECS.values())
    assert any(item.code.value == "needs_review" for item in card.warnings)
    assert "traceback" not in response.text.casefold()


async def test_provider_error_returns_503(client: AsyncClient, install_llm: Any) -> None:
    install_llm(
        ScriptedLLM([LLMError("OpenAI request failed", kind=LLMErrorKind.API)])
    )
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 503
    _assert_public_error(response, UNAVAILABLE)


async def test_timeout_message_without_timeout_type_stays_503(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    install_llm(
        ScriptedLLM([LLMError("LLM request timed out", kind=LLMErrorKind.API)])
    )
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 503
    _assert_public_error(response, UNAVAILABLE)


async def test_generator_timeout_returns_504(client: AsyncClient, install_llm: Any) -> None:
    install_llm(ScriptedLLM([TimeoutError()]))
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 504
    _assert_public_error(response, TIMEOUT)


async def test_extractor_timeout_returns_504(client: AsyncClient, install_llm: Any) -> None:
    install_llm(ScriptedLLM([TimeoutError()]))
    response = await client.post(
        GENERATE_PATH,
        json=_product(description="Учебное описание без вызова генератора."),
    )

    assert response.status_code == 504
    _assert_public_error(response, TIMEOUT)


async def test_api_timeout_cause_returns_504(client: AsyncClient, install_llm: Any) -> None:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    error = LLMError("OpenAI request failed", kind=LLMErrorKind.API)
    error.__cause__ = APITimeoutError(request)
    install_llm(ScriptedLLM([error]))
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 504
    _assert_public_error(response, TIMEOUT)


def _description_with_specs(
    specs: dict[str, str],
    *,
    override: dict[str, str] | None = None,
) -> str:
    """Собрать описание, где каждое значение встречается дословной цитатой."""

    shown = dict(specs)
    if override:
        shown.update(override)
    parts = ["Учебный пример, не паспорт реального прибора."]
    parts.extend(f"{name}: {value}." for name, value in shown.items() if value.strip())
    return " ".join(parts)


def _extraction_payload(name: str, value: str, description: str) -> str:
    evidence = f"{name}: {value}."
    assert evidence in description
    return json.dumps(
        {"attributes": [{"name": name, "value": value, "evidence": evidence}]},
        ensure_ascii=False,
    )


def _assert_matches_response_model(body: dict[str, Any]) -> GenerationResult:
    parsed = GenerationResult.model_validate(body)
    assert parsed.model_dump(mode="json") == body
    return parsed


async def test_description_value_fills_empty_spec(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    specs = dict(_SPECS)
    specs["Степень защиты"] = ""
    description = _description_with_specs(specs, override={"Степень защиты": "IP54"})
    confirmed = {**specs, "Степень защиты": "IP54"}
    script = ScriptedLLM(
        [
            _extraction_payload("Степень защиты", "IP54", description),
            _grounded_generation(confirmed),
        ]
    )
    install_llm(script)
    response = await client.post(
        GENERATE_PATH,
        json=_product(description=description, specs=specs),
    )

    assert response.status_code == 200
    card = _assert_matches_response_model(response.json())
    protection = next(item for item in card.specifications if item.name == "Степень защиты")
    assert protection.value == "IP54"
    assert protection.source.value == "description"
    assert protection.evidence == "Степень защиты: IP54."
    preserved = {
        item.name: item.value
        for item in card.specifications
        if item.name != "Степень защиты"
    }
    assert preserved == {
        "Диапазон показаний": "0...1,6 МПа",
        "Класс точности": "1,5",
        "Присоединение": "M20x1,5",
    }
    assert all(
        item.source.value == "specs"
        for item in card.specifications
        if item.name != "Степень защиты"
    )
    assert script.prompts[0].startswith("Извлеки технические характеристики")
    assert '"Диапазон показаний": "0...1,6 МПа"' not in script.prompts[0]


async def test_conflicting_description_keeps_spec_value(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    description = _description_with_specs(_SPECS, override={"Класс точности": "2,5"})
    script = ScriptedLLM(
        [
            _extraction_payload("Класс точности", "2,5", description),
            _grounded_generation(_SPECS),
        ]
    )
    install_llm(script)
    response = await client.post(GENERATE_PATH, json=_product(description=description))

    assert response.status_code == 200
    card = _assert_matches_response_model(response.json())
    accuracy = next(item for item in card.specifications if item.name == "Класс точности")
    assert accuracy.value == "1,5"
    assert accuracy.source.value == "specs"
    assert any(
        item.code.value == "conflict" and item.field == "Класс точности"
        for item in card.warnings
    )
    assert "2,5" not in response.text


async def test_conflicting_value_in_llm_copy_is_rejected(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    description = _description_with_specs(_SPECS, override={"Класс точности": "2,5"})
    hostile = json.loads(_grounded_generation(_SPECS))
    hostile["seo_description"] = hostile["seo_description"].replace("1,5", "2,5", 1)
    install_llm(
        ScriptedLLM(
            [
                _extraction_payload("Класс точности", "2,5", description),
                json.dumps(hostile, ensure_ascii=False),
                json.dumps(hostile, ensure_ascii=False),
            ]
        )
    )
    response = await client.post(GENERATE_PATH, json=_product(description=description))

    assert response.status_code == 200
    card = _assert_matches_response_model(response.json())
    assert card.status.value == "needs_review"
    assert "2,5" not in response.text
    assert any(item.value == "1,5" for item in card.specifications)


async def test_extractor_invalid_json_returns_502(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    install_llm(ScriptedLLM(["{"]))
    response = await client.post(
        GENERATE_PATH,
        json=_product(description="Учебное описание без подтверждающей цитаты."),
    )

    assert response.status_code == 502
    _assert_public_error(response, BAD_RESPONSE)
    assert "JSONDecodeError" not in response.text
    assert "schema" not in response.text.casefold()


async def test_extractor_pydantic_error_returns_502(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    install_llm(ScriptedLLM([json.dumps({"attributes": [{"name": 1}]})]))
    response = await client.post(
        GENERATE_PATH,
        json=_product(description="Учебное описание для неверной схемы извлечения."),
    )

    assert response.status_code == 502
    _assert_public_error(response, BAD_RESPONSE)
    assert "ValidationError" not in response.text
    assert "validation error" not in response.text.casefold()


async def test_generator_pydantic_error_returns_partial_card(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    install_llm(ScriptedLLM([json.dumps({"seo_title": 15, "seo_description": ""})]))
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 200
    card = _assert_matches_response_model(response.json())
    assert card.status.value == "needs_review"
    assert [item.value for item in card.specifications] == list(_SPECS.values())
    assert "ValidationError" not in response.text


async def test_provider_secret_does_not_leak(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    secret = "sk-live-secret-do-not-leak"
    install_llm(
        ScriptedLLM([LLMError(f"OpenRouter failed for key {secret}", kind=LLMErrorKind.API)])
    )
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 503
    _assert_public_error(response, UNAVAILABLE)
    assert secret not in response.text
    assert "OpenRouter" not in response.text


async def test_direct_provider_timeout_returns_504(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    install_llm(ScriptedLLM([APITimeoutError(request)]))
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 504
    _assert_public_error(response, TIMEOUT)
    assert "openrouter" not in response.text.casefold()


async def test_successful_card_matches_declared_schema(
    client: AsyncClient,
    install_llm: Any,
) -> None:
    install_llm(ScriptedLLM([_grounded_generation(_SPECS)]))
    response = await client.post(GENERATE_PATH, json=_product())

    assert response.status_code == 200
    card = _assert_matches_response_model(response.json())
    assert card.status.value == "complete"
    assert SEO_DESCRIPTION_MIN_LENGTH <= len(card.seo_description) <= 1200
    assert len(card.search_queries) == 3
    assert card.commercial_block.startswith(_NAME)
