"""Диагностика grounding в DEBUG-логах: попытка, поле, значение, фрагмент, причина.

Публичный ответ при этом не меняется: подробности остаются только во внутренних логах.
"""

import logging

import pytest

from app.schemas.product import AttributeConflict, GenerationInput, GenerationStatus
from app.services.description import check_description
from app.services.generator import SEOGenerator
from app.services.grounding import (
    Attempt,
    FindingKind,
    grounding_findings,
    log_findings,
)
from tests.fakes import ScriptedLLMClient
from tests.texts import (
    MANOMETER_SENTENCES,
    full_draft,
    manometer_description,
    manometer_input,
    repair_reply,
    with_length,
)

LOGGER = "uvicorn.error"
DATA = manometer_input()
GOOD = manometer_description()
FOREIGN_SENTENCE = "Класс точности 2,5 подходит для контрольных измерений."


def _with_foreign_number() -> str:
    """Валидное по длине описание, где у класса точности стоит чужое число."""

    sentences = [
        FOREIGN_SENTENCE if sentence.startswith("Класс точности") else sentence
        for sentence in MANOMETER_SENTENCES
    ]
    return " ".join(sentences)


def _rejections(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("grounding rejected")
    ]


@pytest.fixture
def debug_log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    return caplog


def test_findings_name_value_snippet_attribute_and_reason() -> None:
    findings = grounding_findings(DATA, _with_foreign_number(), field_name="seo_description")

    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind is FindingKind.NUMBER
    assert finding.field == "seo_description"
    assert finding.value == "2,5"
    assert "Класс точности 2,5 подходит" in finding.snippet
    assert finding.attribute is not None
    assert "ближайшая перед числом: Класс точности = 1,5" in finding.attribute
    assert "нет ни в одном подтвержденном значении" in finding.reason


def test_initial_rejection_is_logged_with_every_detail(
    debug_log: pytest.LogCaptureFixture,
) -> None:
    check = check_description(DATA, _with_foreign_number())

    assert not check.ok
    [line] = _rejections(debug_log)
    assert "attempt=initial" in line
    assert "field=seo_description" in line
    assert "kind=number" in line
    assert "value='2,5'" in line
    assert "Класс точности = 1,5" in line
    assert "нет ни в одном подтвержденном значении" in line
    assert "Класс точности 2,5 подходит" in line


def test_summary_and_full_text_are_logged_for_a_rejected_description(
    debug_log: pytest.LogCaptureFixture,
) -> None:
    text = _with_foreign_number()
    check_description(DATA, text)

    messages = [record.getMessage() for record in debug_log.records]
    assert any(
        "description rejected attempt=initial field=seo_description issue=ungrounded" in line
        and "число «2,5» не подтверждено" in line
        for line in messages
    )
    assert any(
        line.startswith("description rejected attempt=initial text=") and "2,5" in line
        for line in messages
    )


async def test_grounding_rejection_does_not_start_repair(
    debug_log: pytest.LogCaptureFixture,
) -> None:
    client = ScriptedLLMClient([full_draft(DATA, _with_foreign_number()), repair_reply(GOOD)])

    result = await SEOGenerator(client).generate(DATA)

    assert result.status is GenerationStatus.NEEDS_REVIEW
    assert result.seo_description is None
    assert len(client.calls) == 1
    lines = _rejections(debug_log)
    assert lines
    assert all("attempt=initial" in line for line in lines)
    assert all("attempt=repair" not in line for line in lines)


async def test_accepted_repair_is_logged_as_accepted(
    debug_log: pytest.LogCaptureFixture,
) -> None:
    short = with_length(MANOMETER_SENTENCES[0], 700)
    client = ScriptedLLMClient([full_draft(DATA, short), repair_reply(GOOD)])

    result = await SEOGenerator(client).generate(DATA)

    assert result.status is GenerationStatus.COMPLETE
    assert len(client.calls) == 2
    messages = [record.getMessage() for record in debug_log.records]
    assert f"description accepted attempt=repair length={len(GOOD)}" in messages


def test_value_inside_known_range_with_other_unit_explains_the_unit_mismatch() -> None:
    text = GOOD.replace("0...1,6 МПа", "0...1,6 бар")
    [finding] = grounding_findings(DATA, text, field_name="seo_description")

    assert finding.value == "0...1,6"
    assert finding.attribute is not None and "Диапазон показаний = 0...1,6 МПа" in finding.attribute
    assert "значение «0...1,6 МПа»" in finding.reason
    assert "другая единица" in finding.reason


def test_range_component_with_wrong_unit_is_explained() -> None:
    text = GOOD.replace("Диапазон показаний 0...1,6 МПа", "Диапазон показаний до 1,6 бар")
    [finding] = grounding_findings(DATA, text, field_name="seo_description")

    assert finding.value == "1,6"
    assert "компонент диапазона «0...1,6 МПа»" in finding.reason
    assert "«бар»" in finding.reason and "«МПа»" in finding.reason


def test_broken_code_names_the_whole_code_run() -> None:
    """Кириллическая «М» в резьбе: диагностика показывает код целиком и юникод-значение."""

    text = GOOD.replace("M20x1,5", "\u041c20x1,5")
    findings = grounding_findings(DATA, text, field_name="seo_description")

    assert [item.value for item in findings] == ["20"]
    assert "внутри кода «\u041c20x1,5»" in findings[0].reason
    assert findings[0].attribute is not None
    assert "Присоединение" in findings[0].attribute


def test_swapped_value_names_both_attributes() -> None:
    text = GOOD.replace("Класс точности 1,5", "Класс точности IP54").replace(
        "Степень защиты IP54", "Степень защиты 1,5"
    )
    findings = grounding_findings(DATA, text, field_name="seo_description")
    swapped = [item for item in findings if item.kind is FindingKind.SWAPPED_VALUE]

    assert swapped, findings
    first = swapped[0]
    assert first.field == "seo_description"
    assert first.attribute is not None
    assert "подтверждено только для" in first.reason
    assert first.value in first.snippet


def test_conflict_value_is_reported_with_the_confirmed_one() -> None:
    data = GenerationInput(
        name=DATA.name,
        manufacturer=DATA.manufacturer,
        attributes=DATA.attributes,
        conflicts=[
            AttributeConflict(
                name="Класс точности",
                specs_value="1,5",
                description_value="Класс С",
                evidence="Класс С",
            )
        ],
    )
    text = GOOD.replace("Класс точности 1,5", "Класс точности Класс С")
    [finding] = [
        item
        for item in grounding_findings(data, text, field_name="seo_description")
        if item.kind is FindingKind.CONFLICT_VALUE
    ]

    assert finding.value == "Класс С"
    assert finding.attribute == "Класс точности = 1,5"
    assert "Класс С" in finding.snippet


async def test_commercial_block_and_queries_report_their_own_field(
    debug_log: pytest.LogCaptureFixture,
) -> None:
    draft = full_draft(
        DATA,
        GOOD,
        commercial_block="Манометр МП-Демо-01 выдерживает 99 МПа.",
        search_queries=[
            "Манометр демонстрационный МП-Демо-01 купить",
            "Манометр демонстрационный МП-Демо-01 250 бар",
            "Манометр демонстрационный МП-Демо-01 характеристики",
            "Учебный стенд «Сигнал» манометр",
            "купить датчик давления",
        ],
    )
    result = await SEOGenerator(ScriptedLLMClient([draft])).generate(DATA)

    assert result.status is GenerationStatus.COMPLETE
    lines = _rejections(debug_log)
    assert any("field=commercial_block" in line and "value='99'" in line for line in lines)
    assert any("field=search_query" in line and "value='250'" in line for line in lines)
    messages = [record.getMessage() for record in debug_log.records]
    assert any(
        line.startswith("query rejected") and "купить датчик давления" in line
        for line in messages
    )


async def test_unsupported_claim_in_title_is_logged(
    debug_log: pytest.LogCaptureFixture,
) -> None:
    draft = full_draft(DATA, GOOD, seo_title="Лучший манометр МП-Демо-01")
    result = await SEOGenerator(ScriptedLLMClient([draft])).generate(DATA)

    assert result.seo_title == "Манометр демонстрационный МП-Демо-01"
    assert any(
        record.getMessage().startswith("claim rejected")
        and "field=seo_title" in record.getMessage()
        for record in debug_log.records
    )


def test_details_are_written_only_in_debug_mode(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    check_description(DATA, _with_foreign_number())
    log_findings(
        grounding_findings(DATA, _with_foreign_number(), field_name="seo_description"),
        Attempt.INITIAL,
    )

    assert not [r for r in caplog.records if "2,5" in r.getMessage()]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_debug_log_has_no_secrets(
    debug_log: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-or-v1-secret-for-test-0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    client = ScriptedLLMClient([full_draft(DATA, _with_foreign_number())])

    await SEOGenerator(client).generate(DATA)

    assert debug_log.records
    assert secret not in debug_log.text
    assert "OPENAI_API_KEY" not in debug_log.text
    assert "Authorization" not in debug_log.text


async def test_public_message_does_not_leak_the_rejected_value() -> None:
    client = ScriptedLLMClient(
        [full_draft(DATA, _with_foreign_number())]
    )

    result = await SEOGenerator(client).generate(DATA)

    published = " ".join(item.message for item in result.warnings)
    assert "2,5" not in published
    assert "не подтвержденное характеристиками" in published


def test_long_filler_text_is_still_rejected_for_a_foreign_number() -> None:
    """Диагностика не ослабляет проверку: чужое число по-прежнему отклоняется."""

    text = with_length(GOOD[:200], 900).replace("Перед", "Перед 77", 1)
    assert not check_description(DATA, text).ok
