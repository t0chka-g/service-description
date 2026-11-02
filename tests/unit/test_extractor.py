"""Unit-тесты CharacteristicExtractor."""

from typing import Any, assert_never

import pytest

from app.schemas.product import (
    AttributeSource,
    ExtractedAttribute,
    ExtractionResult,
    ProductInput,
)
from app.services.extractor import CharacteristicExtractor


def _attributes_by_name(result: ExtractionResult) -> dict[str, Any]:
    return {item.name: item for item in result.attributes}


def _assert_evidence_from_source(
    product: ProductInput,
    actual: ExtractedAttribute,
    expected_evidence: str,
) -> None:
    assert expected_evidence in actual.evidence
    if actual.source is AttributeSource.SPECS:
        assert actual.value == product.specs[actual.name]
        assert actual.evidence in product.specs[actual.name]
    elif actual.source is AttributeSource.DESCRIPTION:
        assert actual.evidence in product.description
        assert actual.evidence != product.description
    else:
        assert_never(actual.source)


def _conflicts_by_name(result: ExtractionResult) -> dict[str, Any]:
    return {item.name: item for item in result.conflicts}


@pytest.mark.parametrize(
    "case_id",
    [
        "specs_complete",
        "partial_from_description",
        "missing_in_both",
        "empty_specs_rich_description",
        "specs_description_conflict",
        "numeric_units_formats",
        "nonstandard_order",
        "synonym_names",
        "empty_invalid_input",
        "prompt_injection_no_hallucination",
        "unsupported_llm_attribute",
    ],
)
async def test_extract_matches_case_contract(
    case_id: str,
    extraction_cases_by_id: dict[str, dict[str, Any]],
    fake_llm_factory,
) -> None:
    case = extraction_cases_by_id[case_id]
    product = ProductInput.model_validate(case["input"])
    extractor = CharacteristicExtractor(llm_client=fake_llm_factory(case))

    result = await extractor.extract(
        product=product,
        required_fields=case["required_fields"],
    )

    assert isinstance(result, ExtractionResult)
    by_name = _attributes_by_name(result)
    assert len(result.attributes) == len(by_name)

    assert set(by_name) == set(case["expected"])
    for name, expected_attr in case["expected"].items():
        actual = by_name[name]
        assert actual.value == expected_attr["value"]
        assert actual.source == AttributeSource(expected_attr["source"])
        _assert_evidence_from_source(product, actual, expected_attr["evidence"])

    assert sorted(result.missing_fields) == sorted(case["missing_fields"])

    conflicts = _conflicts_by_name(result)
    assert set(conflicts) == {item["name"] for item in case["conflicts"]}
    for expected_conflict in case["conflicts"]:
        actual_conflict = conflicts[expected_conflict["name"]]
        assert actual_conflict.specs_value == expected_conflict["specs_value"]
        assert actual_conflict.description_value == expected_conflict["description_value"]
        assert expected_conflict["evidence"] in actual_conflict.evidence


async def test_preserves_specs_values_unchanged(
    extraction_cases_by_id: dict[str, dict[str, Any]],
    fake_llm_factory,
) -> None:
    case = extraction_cases_by_id["specs_complete"]
    extractor = CharacteristicExtractor(llm_client=fake_llm_factory(case))

    result = await extractor.extract(
        product=ProductInput.model_validate(case["input"]),
        required_fields=case["required_fields"],
    )

    for name, raw_value in case["input"]["specs"].items():
        attr = _attributes_by_name(result)[name]
        assert attr.value == raw_value
        assert attr.source is AttributeSource.SPECS


async def test_adds_attributes_extracted_from_description(
    extraction_cases_by_id: dict[str, dict[str, Any]],
    fake_llm_factory,
) -> None:
    case = extraction_cases_by_id["partial_from_description"]
    extractor = CharacteristicExtractor(llm_client=fake_llm_factory(case))

    result = await extractor.extract(
        product=ProductInput.model_validate(case["input"]),
        required_fields=case["required_fields"],
    )

    by_name = _attributes_by_name(result)
    assert by_name["диапазон измерений"].source is AttributeSource.DESCRIPTION
    assert by_name["выходной сигнал"].source is AttributeSource.DESCRIPTION
    assert "Диапазон измерений температуры −50...+150 °C" in by_name["диапазон измерений"].evidence


async def test_does_not_include_unconfirmed_attributes(
    extraction_cases_by_id: dict[str, dict[str, Any]],
    fake_llm_factory,
) -> None:
    case = extraction_cases_by_id["prompt_injection_no_hallucination"]
    extractor = CharacteristicExtractor(llm_client=fake_llm_factory(case))

    result = await extractor.extract(
        product=ProductInput.model_validate(case["input"]),
        required_fields=case["required_fields"],
    )

    values = {attr.name: attr.value for attr in result.attributes}
    for name, forbidden_value in case["forbidden_values"].items():
        assert values.get(name) != forbidden_value
        assert name not in values


async def test_detects_conflicts_between_specs_and_description(
    extraction_cases_by_id: dict[str, dict[str, Any]],
    fake_llm_factory,
) -> None:
    case = extraction_cases_by_id["specs_description_conflict"]
    extractor = CharacteristicExtractor(llm_client=fake_llm_factory(case))

    result = await extractor.extract(
        product=ProductInput.model_validate(case["input"]),
        required_fields=case["required_fields"],
    )

    attribute = _attributes_by_name(result)["диапазон измерений"]
    assert attribute.value == "0...1,6 МПа"
    assert attribute.source is AttributeSource.SPECS
    conflict = _conflicts_by_name(result)["диапазон измерений"]
    assert conflict.specs_value == "0...1,6 МПа"
    assert conflict.description_value == "0...2,5 МПа"


async def test_reports_missing_fields(
    extraction_cases_by_id: dict[str, dict[str, Any]],
    fake_llm_factory,
) -> None:
    case = extraction_cases_by_id["missing_in_both"]
    extractor = CharacteristicExtractor(llm_client=fake_llm_factory(case))

    result = await extractor.extract(
        product=ProductInput.model_validate(case["input"]),
        required_fields=case["required_fields"],
    )

    assert sorted(result.missing_fields) == sorted(case["missing_fields"])


async def test_preserves_numeric_values_and_units(
    extraction_cases_by_id: dict[str, dict[str, Any]],
    fake_llm_factory,
) -> None:
    case = extraction_cases_by_id["numeric_units_formats"]
    extractor = CharacteristicExtractor(llm_client=fake_llm_factory(case))

    result = await extractor.extract(
        product=ProductInput.model_validate(case["input"]),
        required_fields=case["required_fields"],
    )

    by_name = _attributes_by_name(result)
    assert by_name["диапазон измерений"].value == "0–10 bar"
    assert by_name["погрешность"].value == "±0,25 %"
    assert by_name["выходной сигнал"].value == "4…20mA"
    assert by_name["температура среды"].value == "-40...+85°C"


async def test_handles_empty_input(
    extraction_cases_by_id: dict[str, dict[str, Any]],
    fake_llm_factory,
) -> None:
    case = extraction_cases_by_id["empty_invalid_input"]
    extractor = CharacteristicExtractor(llm_client=fake_llm_factory(case))

    result = await extractor.extract(
        product=ProductInput.model_validate(case["input"]),
        required_fields=case["required_fields"],
    )

    assert result.attributes == []
    assert sorted(result.missing_fields) == sorted(case["required_fields"])
    assert result.conflicts == []
