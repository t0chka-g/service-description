"""Пробел между числом и единицей и границы технических кодов.

Случаи взяты из DEBUG-лога генератора, но проверяются на отдельных парах
«характеристика — значение», а не на тексте одного манометра.
"""

import re

import pytest

from app.schemas.product import GenerationInput, GenerationStatus
from app.services.generator import SEOGenerator
from app.services.grounding import FindingKind, grounding_findings, text_grounded
from tests.fakes import ScriptedLLMClient
from tests.texts import attribute, full_draft, repair_reply, with_length

LENGTH = "10см"
WIDTH = "2см"
THREAD = "M20x1,5"
OTHER_CODE = "DN50"
OTHER_WIDTH = "5мм"


def _input(*pairs: tuple[str, str]) -> GenerationInput:
    return GenerationInput(
        name="Прибор демонстрационный",
        manufacturer="Стенд Сигнал",
        attributes=[attribute(name, value) for name, value in pairs],
    )


def _logged_input() -> GenerationInput:
    """Пара, на которой лог отклонил запись с пробелом и связал 2 с резьбой."""

    return _input(
        ("Длина", LENGTH),
        ("Ширина", WIDTH),
        ("Присоединение", THREAD),
        ("Диапазон", "0...1,6 МПа"),
    )


def _findings(data: GenerationInput, text: str) -> list:
    return grounding_findings(data, text, field_name="seo_description")


@pytest.mark.parametrize(
    ("confirmed", "written"),
    [
        ("10см", "10 см"),
        ("10 см", "10см"),
        ("25кг", "25 кг"),
        ("4мм", "4 мм"),
        ("10см", "10\u00a0см"),
    ],
)
def test_space_between_number_and_unit_is_the_same_value(confirmed: str, written: str) -> None:
    data = _input(("Длина", confirmed), ("Масса", "3кг"), ("Материал", "латунь"))
    text = f"Длина корпуса {written}, масса указана отдельно."

    assert text_grounded(data, text)
    assert _findings(data, text) == []


@pytest.mark.parametrize(
    ("confirmed", "written"),
    [
        ("10см", "10 мм"),
        ("10 мм", "10см"),
        ("25кг", "25 г"),
        ("4мм", "4 см"),
    ],
)
def test_different_unit_is_not_the_same_value(confirmed: str, written: str) -> None:
    data = _input(("Длина", confirmed), ("Масса", "3кг"), ("Материал", "латунь"))
    text = f"Длина корпуса {written}."
    findings = _findings(data, text)
    number, unit = _measure(written)
    confirmed_unit = _measure(confirmed.replace(" ", ""))[1]

    assert findings
    assert findings[0].kind is FindingKind.NUMBER
    assert findings[0].value == number
    assert f"«{unit}»".casefold() in findings[0].reason.casefold()
    assert f"«{confirmed_unit}»".casefold() in findings[0].reason.casefold()
    assert THREAD not in findings[0].reason


def _measure(written: str) -> tuple[str, str]:
    match = re.match(r"(\d+(?:[,.]\d+)?)\s*([^\W\d_]+)", written)
    assert match is not None, written
    return match.group(1), match.group(2)


def test_bare_number_does_not_inherit_a_unit() -> None:
    data = _logged_input()
    findings = _findings(data, "Длина корпуса 10 без единицы измерения.")

    assert any(item.value == "10" for item in findings)


def test_width_with_space_is_not_read_from_thread_code() -> None:
    """2 в «M20x1,5» — часть кода, а не значение ширины «2см»."""

    data = _logged_input()
    text = "Присоединение M20x1,5. Ширина 2 см."
    findings = _findings(data, text)

    assert findings == []
    assert all("M20x1,5" not in (item.reason or "") or item.value != "2" for item in findings)


def test_inner_digit_of_another_code_is_not_a_separate_width() -> None:
    data = _input(
        ("Ширина", OTHER_WIDTH),
        ("Условный проход", OTHER_CODE),
        ("Материал", "сталь"),
    )
    text = "Условный проход DN50. Ширина 5 мм."

    assert _findings(data, text) == []


def test_inflected_width_in_the_next_sentence_is_not_bound_to_the_thread() -> None:
    data = _logged_input()
    text = (
        "Присоединение выполнено с резьбой M20x1,5 и задает ответную деталь. "
        "Корпус шириной 2 см занимает мало места."
    )
    findings = _findings(data, text)

    assert findings == []


def test_inflected_size_in_the_same_sentence_stays_with_its_attribute() -> None:
    data = _input(
        ("Высота", "8см"),
        ("Присоединение", THREAD),
        ("Материал", "сталь"),
    )
    text = "Присоединение M20x1,5 и корпус высотой 8 см стоят в одной фразе."

    assert _findings(data, text) == []


def test_swapped_value_is_still_rejected() -> None:
    data = _logged_input()
    text = "Присоединение 2 см задает резьбу, а не габарит."
    findings = _findings(data, text)
    swapped = [item for item in findings if item.kind is FindingKind.SWAPPED_VALUE]

    assert swapped
    assert swapped[0].value == "2"
    assert "Ширина" in swapped[0].reason
    assert "2 см" in swapped[0].snippet


def test_swapped_snippet_points_at_the_separate_number_not_the_code() -> None:
    data = _logged_input()
    text = (
        "Присоединение описано кодом M20x1,5 в начале фразы, "
        "а затем ошибочно указано 2 см как размер этой резьбы."
    )
    swapped = [
        item for item in _findings(data, text) if item.kind is FindingKind.SWAPPED_VALUE
    ]

    assert swapped
    assert "2 см" in swapped[0].snippet


def _good_description(data: GenerationInput) -> str:
    core = (
        f"{data.name} производства {data.manufacturer} нужен для сравнения моделей по параметрам. "
        "Длина 10 см и ширина 2 см задают габарит корпуса перед монтажом. "
        "Присоединение M20x1,5 описывает резьбу ответной детали. "
        "Диапазон 0...1,6 МПа ограничивает шкалу прибора."
    )
    return with_length(core, 1000)


async def test_spaced_units_pass_on_the_first_call() -> None:
    data = _logged_input()
    text = _good_description(data)
    assert 800 <= len(text) <= 1200
    client = ScriptedLLMClient([full_draft(data, text), repair_reply(text)])

    result = await SEOGenerator(client).generate(data)

    assert result.status is GenerationStatus.COMPLETE
    assert result.seo_description == text
    assert len(client.calls) == 1


async def test_grounding_error_does_not_call_the_model_again() -> None:
    data = _logged_input()
    text = with_length(f"{_good_description(data)[:400]} Длина 10 мм.", 1000)
    client = ScriptedLLMClient([full_draft(data, text), repair_reply(_good_description(data))])

    result = await SEOGenerator(client).generate(data)

    assert result.status is GenerationStatus.NEEDS_REVIEW
    assert result.seo_description is None
    assert len(client.calls) == 1
    assert "10 мм" not in " ".join(item.message for item in result.warnings)


async def test_warranty_and_invented_number_stay_blocked() -> None:
    data = _logged_input()
    invented = with_length(f"{_good_description(data)} Рабочее давление до 2,5 МПа.", 1100)
    warranty = with_length(
        f"{_good_description(data)} На прибор действует гарантия производителя.",
        1100,
    )

    invented_result = await SEOGenerator(
        ScriptedLLMClient([full_draft(data, invented)])
    ).generate(data)
    warranty_client = ScriptedLLMClient(
        [full_draft(data, warranty), repair_reply(_good_description(data))]
    )
    warranty_result = await SEOGenerator(warranty_client).generate(data)

    assert invented_result.seo_description is None
    published = warranty_result.seo_description or ""
    assert "гарантия" not in published.casefold()
    assert "2,5" not in " ".join(item.message for item in invented_result.warnings)
    assert len(warranty_client.calls) == 2


async def test_commercial_block_keeps_spaced_units() -> None:
    data = _logged_input()
    text = _good_description(data)
    spaced = "Длина 10 см и ширина 2 см, присоединение M20x1,5."
    client = ScriptedLLMClient([full_draft(data, text, commercial_block=spaced)])

    result = await SEOGenerator(client).generate(data)

    assert result.commercial_block == spaced
    assert len(client.calls) == 1
