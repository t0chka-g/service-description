"""Оценочные тесты качества извлечения характеристик через LLM.

Запуск: pytest -m llm
Без OPENAI_API_KEY сетевые запросы не выполняются.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import pytest

from app.schemas.product import ExtractedAttribute, ExtractionResult


def normalize_value(value: str) -> str:
    """Нормализация записи значения без изменения технического смысла."""
    normalized = value.strip().lower().replace(",", ".")
    return " ".join(normalized.split())


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def attribute_pairs(attributes: Iterable[ExtractedAttribute]) -> set[tuple[str, str]]:
    return {(normalize_name(item.name), normalize_value(item.value)) for item in attributes}


def precision_recall_f1(
    expected: set[tuple[str, str]],
    actual: set[tuple[str, str]],
) -> tuple[float, float, float]:
    """Precision / Recall / F1 по множеству (name, value)."""
    if not actual and not expected:
        return 1.0, 1.0, 1.0
    if not actual:
        return 0.0, 0.0, 0.0
    if not expected:
        return 0.0, 1.0 if not actual else 0.0, 0.0

    true_positive = len(expected & actual)
    precision = true_positive / len(actual)
    recall = true_positive / len(expected)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def hallucination_rate(
    result: ExtractionResult,
    ground_truth: set[tuple[str, str]],
) -> float:
    """Доля извлеченных значений, отсутствующих в ground truth."""
    actual = attribute_pairs(result.attributes)
    if not actual:
        return 0.0
    hallucinated = actual - ground_truth
    return len(hallucinated) / len(actual)


def test_metric_helpers_on_ground_truth_comparison() -> None:
    """Проверка формул метрик на детерминированных множествах (без LLM)."""
    expected = {("диапазон измерений", "0...1.0 мпа"), ("класс точности", "1.5")}
    actual = {
        ("диапазон измерений", "0...1.0 мпа"),
        ("степень защиты", "ip68"),
    }

    precision, recall, f1 = precision_recall_f1(expected, actual)

    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)

    result = ExtractionResult(
        attributes=[
            ExtractedAttribute(
                name="диапазон измерений",
                value="0...1,0 МПа",
                source="specs",
                evidence="0...1,0 МПа",
            ),
            ExtractedAttribute(
                name="степень защиты",
                value="IP68",
                source="description",
                evidence="IP68",
            ),
        ]
    )
    assert hallucination_rate(result, expected) == pytest.approx(0.5)


@pytest.mark.llm
async def test_llm_extraction_against_ground_truth(
    extraction_cases_by_id: dict,
) -> None:
    """Сравнение реальной модели с ground truth (отдельный opt-in запуск)."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    _ = extraction_cases_by_id
    pytest.skip("Real LLM extraction evaluation is not implemented yet")
