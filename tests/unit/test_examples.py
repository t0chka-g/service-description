"""Демо-примеры: файлы совпадают с OpenAPI и проходят собственные проверки."""

import importlib.util
import json
from pathlib import Path

from app.api.examples import (
    COMPLETE_EXAMPLE,
    INSUFFICIENT_EXAMPLE,
    MANOMETER_REQUEST,
    NEEDS_REVIEW_EXAMPLE,
    REQUEST_EXAMPLES,
    TEMPERATURE_SENSOR_REQUEST,
    THERMOMETER_REQUEST,
)
from app.schemas.product import GenerationInput, GenerationResult, GenerationStatus
from app.services.generator import SEOGenerator
from tests.fakes import ScriptedLLMClient
from tests.texts import attribute, full_draft, with_length

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "openrouter_smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("openrouter_smoke", SMOKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(name: str) -> dict[str, object]:
    return json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))


def test_example_files_match_openapi_examples() -> None:
    assert _read("manometer") == MANOMETER_REQUEST
    assert _read("temperature_sensor") == TEMPERATURE_SENSOR_REQUEST
    assert _read("thermometer_insufficient") == THERMOMETER_REQUEST
    assert REQUEST_EXAMPLES["manometer"]["value"] == MANOMETER_REQUEST
    assert REQUEST_EXAMPLES["temperature_sensor"]["value"] == TEMPERATURE_SENSOR_REQUEST


def test_response_examples_are_valid_cards() -> None:
    assert GenerationResult.model_validate(COMPLETE_EXAMPLE).status is GenerationStatus.COMPLETE
    review = GenerationResult.model_validate(NEEDS_REVIEW_EXAMPLE)
    assert review.status is GenerationStatus.NEEDS_REVIEW
    assert review.seo_description is None
    insufficient = GenerationResult.model_validate(INSUFFICIENT_EXAMPLE)
    assert insufficient.seo_description is None
    assert insufficient.commercial_block is None


def test_complete_example_passes_the_smoke_contract() -> None:
    smoke = _load_smoke()
    assert smoke.check_card(COMPLETE_EXAMPLE) == []
    assert smoke.check_card(NEEDS_REVIEW_EXAMPLE) != []
    assert smoke.check_card(INSUFFICIENT_EXAMPLE) != []


async def test_complete_example_description_survives_full_validation() -> None:
    data = GenerationInput(
        name=str(MANOMETER_REQUEST["name"]),
        manufacturer=str(MANOMETER_REQUEST["manufacturer"]),
        attributes=[attribute(k, v) for k, v in MANOMETER_REQUEST["specs"].items()],  # type: ignore[attr-defined]
    )
    description = str(COMPLETE_EXAMPLE["seo_description"])
    llm = ScriptedLLMClient([full_draft(data, description)])
    result = await SEOGenerator(llm).generate(data)

    assert result.status is GenerationStatus.COMPLETE
    assert result.seo_description == description
    assert len(llm.calls) == 1


async def test_second_scenario_builds_a_complete_card() -> None:
    specs: dict[str, str] = TEMPERATURE_SENSOR_REQUEST["specs"]  # type: ignore[assignment]
    data = GenerationInput(
        name=str(TEMPERATURE_SENSOR_REQUEST["name"]),
        manufacturer=str(TEMPERATURE_SENSOR_REQUEST["manufacturer"]),
        attributes=[attribute(k, v) for k, v in specs.items()],
    )
    core = (
        f"{data.name} производства {data.manufacturer} предназначен для измерения температуры. "
        "Диапазон измерений -50...+150 °C определяет рабочие границы датчика. "
        "Тип чувствительного элемента Pt100 важно учесть при выборе преобразователя. "
        "Класс допуска B описывает допустимое отклонение показаний. "
        "Выходной сигнал 4...20 мА передается на систему сбора данных. "
        "Степень защиты IP65 относится к защите корпуса."
    )
    draft = full_draft(
        data,
        with_length(core, 950),
        search_queries=[
            f"{data.name} купить",
            f"{data.name} характеристики",
            "датчик температуры Pt100",
        ],
    )
    result = await SEOGenerator(ScriptedLLMClient([draft])).generate(data)

    assert result.status is GenerationStatus.COMPLETE
    assert len(result.specifications) == 5
    assert 800 <= len(result.seo_description or "") <= 1200
