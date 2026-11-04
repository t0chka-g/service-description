"""Pipeline SEO-генерации: длина, исправление, fail-soft, grounding.

LLM подменен очередью детерминированных ответов. Лишний вызов модели
падает с AssertionError, поэтому предел «не больше двух запросов» проверяется
самим фейком. Реальный API не используется.
"""

from pathlib import Path
from typing import Any

import pytest

from app.clients.llm import LLMError, LLMErrorKind
from app.schemas.product import (
    SEARCH_QUERIES_MAX,
    SEARCH_QUERIES_MIN,
    SEO_DESCRIPTION_MAX_LENGTH,
    SEO_DESCRIPTION_MIN_LENGTH,
    SEO_TITLE_MAX_LENGTH,
    GenerationInput,
    GenerationResult,
    GenerationStatus,
    GenerationWarningCode,
)
from app.services.generator import DescriptionRepair, SEOGenerator
from app.services.grounding import grounding_findings
from tests.fakes import ScriptedLLMClient
from tests.texts import (
    MANOMETER_SENTENCES,
    NEUTRAL_SENTENCES,
    attribute,
    full_draft,
    long_text,
    manometer_description,
    manometer_input,
    repair_reply,
    with_length,
)

DATA = manometer_input()
GOOD = manometer_description()
SHORT = " ".join(MANOMETER_SENTENCES[:3])
SERVICE_MARKERS = (
    "не собрано",
    "недостаточно",
    "подтвержд",
    "не добавля",
    "needs_review",
    "insufficient",
    "неподтвержд",
)
TOO_SHORT = GenerationWarningCode.DESCRIPTION_TOO_SHORT
REJECTED = GenerationWarningCode.DESCRIPTION_REJECTED


def _run(*steps: str | dict[str, Any] | Exception) -> tuple[SEOGenerator, ScriptedLLMClient]:
    client = ScriptedLLMClient(list(steps))
    return SEOGenerator(client), client


async def _generate(
    *steps: str | dict[str, Any] | Exception,
    data: GenerationInput = DATA,
) -> tuple[GenerationResult, ScriptedLLMClient]:
    generator, client = _run(*steps)
    return await generator.generate(data), client


def _published(result: GenerationResult) -> str:
    parts = [
        result.seo_title,
        result.seo_description or "",
        result.commercial_block or "",
        *result.search_queries,
    ]
    return "\n".join(parts)


def _codes(result: GenerationResult) -> set[GenerationWarningCode]:
    return {item.code for item in result.warnings}


def _assert_complete_card(result: GenerationResult) -> None:
    assert result.status is GenerationStatus.COMPLETE
    assert result.seo_description is not None
    assert SEO_DESCRIPTION_MIN_LENGTH <= len(result.seo_description) <= SEO_DESCRIPTION_MAX_LENGTH
    assert 0 < len(result.seo_title) <= SEO_TITLE_MAX_LENGTH
    assert SEARCH_QUERIES_MIN <= len(result.search_queries) <= SEARCH_QUERIES_MAX
    assert result.commercial_block


def _assert_specs_kept(result: GenerationResult, data: GenerationInput = DATA) -> None:
    assert [(a.name, a.value, a.source, a.evidence) for a in result.specifications] == [
        (a.name, a.value, a.source, a.evidence) for a in data.attributes
    ]


def _assert_no_service_text(result: GenerationResult) -> None:
    published = _published(result).casefold()
    for marker in SERVICE_MARKERS:
        assert marker not in published, marker


def test_sanity_of_test_texts() -> None:
    assert 900 <= len(GOOD) <= 1050
    assert len(set(MANOMETER_SENTENCES)) == len(MANOMETER_SENTENCES)
    assert len(set(NEUTRAL_SENTENCES)) == len(NEUTRAL_SENTENCES)


# 1–2. Четыре подтвержденные характеристики и целевая длина.


async def test_four_confirmed_characteristics_make_full_card() -> None:
    result, client = await _generate(full_draft(DATA))

    _assert_complete_card(result)
    assert result.seo_description == GOOD
    assert result.warnings == []
    assert result.missing_fields == []
    assert len(client.calls) == 1
    for item in DATA.attributes:
        assert item.value in client.calls[0]


@pytest.mark.parametrize("length", [900, 1000, 1050])
async def test_target_length_description_is_complete_without_repair(length: int) -> None:
    text = with_length(MANOMETER_SENTENCES[0], length)
    result, client = await _generate(full_draft(DATA, text))

    _assert_complete_card(result)
    assert result.seo_description == text
    assert len(client.calls) == 1


@pytest.mark.parametrize("length", [SEO_DESCRIPTION_MIN_LENGTH, SEO_DESCRIPTION_MAX_LENGTH])
async def test_allowed_range_borders_are_complete(length: int) -> None:
    text = with_length(MANOMETER_SENTENCES[0], length)
    result, client = await _generate(full_draft(DATA, text))

    _assert_complete_card(result)
    assert len(result.seo_description or "") == length
    assert len(client.calls) == 1


# 3. Длиннее 1200: безопасное сокращение без второго вызова.


async def test_overlong_description_is_shortened_at_sentence_boundary() -> None:
    original = long_text(GOOD, 1500)
    result, client = await _generate(full_draft(DATA, original))

    _assert_complete_card(result)
    shortened = result.seo_description or ""
    assert original.startswith(shortened)
    assert shortened.endswith(".")
    assert original[len(shortened)] == " "
    for item in DATA.attributes:
        assert item.value in shortened
    assert len(client.calls) == 1


async def test_shortening_drops_hallucination_in_cut_tail() -> None:
    tail = "Гарантия производителя составляет 24 месяца."
    original = f"{GOOD} {' '.join(NEUTRAL_SENTENCES[:3])} {tail}"
    assert len(original) > SEO_DESCRIPTION_MAX_LENGTH
    result, client = await _generate(full_draft(DATA, original))

    _assert_complete_card(result)
    assert "24" not in _published(result)
    assert len(client.calls) == 1


async def test_overlong_text_without_safe_cut_uses_single_repair() -> None:
    blob = "слово, " * 250
    result, client = await _generate(full_draft(DATA, blob), repair_reply(GOOD))

    _assert_complete_card(result)
    assert result.seo_description == GOOD
    assert len(client.calls) == 2
    assert "1200" in client.calls[1]


# 4. Короче 800: один целевой repair только описания.


async def test_short_description_is_repaired_once() -> None:
    draft = full_draft(DATA, SHORT)
    result, client = await _generate(draft, repair_reply(GOOD))

    _assert_complete_card(result)
    assert result.seo_description == GOOD
    assert result.seo_title == draft["seo_title"]
    assert result.search_queries == draft["search_queries"]
    assert result.commercial_block == draft["commercial_block"]
    assert len(client.calls) == 2
    assert client.response_models[1] is DescriptionRepair
    repair_prompt = client.calls[1]
    assert SHORT in repair_prompt
    for item in DATA.attributes:
        assert item.value in repair_prompt
    for target in ("900", "1050", "800", "1200"):
        assert target in repair_prompt
    assert not repair_prompt.startswith("Собери SEO-карточку")
    assert "search_queries" not in repair_prompt


# 5. Неуспешный repair → needs_review.


@pytest.mark.parametrize(
    "second",
    [
        repair_reply(SHORT),
        repair_reply(""),
        "{",
        {"unexpected": "shape"},
        LLMError("OpenAI request failed", kind=LLMErrorKind.API),
    ],
    ids=["still_short", "empty", "bad_json", "wrong_shape", "provider_error"],
)
async def test_failed_repair_returns_needs_review(second: str | dict[str, Any] | Exception) -> None:
    draft = full_draft(DATA, SHORT)
    result, client = await _generate(draft, second)

    assert result.status is GenerationStatus.NEEDS_REVIEW
    assert result.seo_description is None
    assert TOO_SHORT in _codes(result)
    assert GenerationWarningCode.NEEDS_REVIEW in _codes(result)
    assert len(client.calls) == 2
    _assert_specs_kept(result)
    assert result.search_queries == draft["search_queries"]
    assert result.commercial_block == draft["commercial_block"]


async def test_reason_is_reported_outside_published_text() -> None:
    result, _client = await _generate(full_draft(DATA, SHORT), repair_reply(SHORT))

    reasons = [item for item in result.warnings if item.code is TOO_SHORT]
    assert len(reasons) == 1
    assert reasons[0].field == "seo_description"
    assert str(len(SHORT)) in reasons[0].message
    assert reasons[0].message not in _published(result)


# 6. Частичный результат без служебного текста в публикуемых полях.


def _two_attribute_input() -> GenerationInput:
    return GenerationInput(
        name=DATA.name,
        manufacturer=DATA.manufacturer,
        attributes=DATA.attributes[:2],
        missing_fields=["Присоединение", "Степень защиты"],
    )


PARTIAL_SCENARIOS: list[tuple[str, list[Any], GenerationInput, GenerationWarningCode]] = [
    ("unparseable", ["{"], DATA, GenerationWarningCode.GENERATION_FAILED),
    (
        "short_after_repair",
        [full_draft(DATA, SHORT), repair_reply(SHORT)],
        DATA,
        TOO_SHORT,
    ),
    (
        "invented_after_repair",
        [
            full_draft(DATA, f"{GOOD} Рабочее давление до 2,5 МПа."),
            repair_reply(f"{GOOD} Рабочее давление до 2,5 МПа."),
        ],
        DATA,
        REJECTED,
    ),
    ("closed_gate", [], _two_attribute_input(), GenerationWarningCode.INSUFFICIENT_DATA),
]


@pytest.mark.parametrize(
    ("steps", "data", "code"),
    [(steps, data, code) for _, steps, data, code in PARTIAL_SCENARIOS],
    ids=[scenario_id for scenario_id, *_ in PARTIAL_SCENARIOS],
)
async def test_partial_result_has_no_service_text_in_published_fields(
    steps: list[Any],
    data: GenerationInput,
    code: GenerationWarningCode,
) -> None:
    result, _client = await _generate(*steps, data=data)

    assert result.status is not GenerationStatus.COMPLETE
    assert result.seo_description is None
    _assert_no_service_text(result)
    _assert_specs_kept(result, data)
    assert code in _codes(result)
    assert all(item.message for item in result.warnings if item.code is code)
    assert result.model_dump(mode="json")["seo_description"] is None


async def test_unparseable_response_is_not_retried() -> None:
    result, client = await _generate("{")

    assert result.status is GenerationStatus.NEEDS_REVIEW
    assert len(client.calls) == 1


async def test_closed_gate_does_not_call_model_and_keeps_facts() -> None:
    data = _two_attribute_input()
    result, client = await _generate(data=data)

    assert result.status is GenerationStatus.INSUFFICIENT_DATA
    assert client.calls == []
    assert result.search_queries == []
    assert result.missing_fields == ["Присоединение", "Степень защиты"]
    _assert_specs_kept(result, data)


# 7. Повторы ради длины.


def _padded_by_repetition() -> str:
    text = MANOMETER_SENTENCES[0]
    while len(text) < 900:
        text += f" {NEUTRAL_SENTENCES[0]}"
    return text


def _value_stuffed() -> str:
    openers = ["Первое", "Второе", "Третье", "Четвертое", "Пятое", "Шестое", "Седьмое"]
    stuffed = [f"{word} соображение при монтаже: IP54 нужно учитывать." for word in openers]
    return with_length(" ".join([MANOMETER_SENTENCES[0], *stuffed]), 950)


@pytest.mark.parametrize(
    "text",
    [_padded_by_repetition(), _value_stuffed()],
    ids=["repeated_sentence", "repeated_characteristic"],
)
async def test_repetition_is_not_accepted_as_length(text: str) -> None:
    assert SEO_DESCRIPTION_MIN_LENGTH <= len(text) <= SEO_DESCRIPTION_MAX_LENGTH
    result, client = await _generate(full_draft(DATA, text), repair_reply(text))

    assert result.status is GenerationStatus.NEEDS_REVIEW
    assert result.seo_description is None
    assert REJECTED in _codes(result)
    assert len(client.calls) == 2


async def test_repair_may_fix_repetition() -> None:
    result, client = await _generate(
        full_draft(DATA, _padded_by_repetition()),
        repair_reply(GOOD),
    )

    _assert_complete_card(result)
    assert result.seo_description == GOOD
    assert len(client.calls) == 2


# 8. Характеристики не теряются ни в одном исходе.


@pytest.mark.parametrize(
    "steps",
    [
        [full_draft(DATA)],
        [full_draft(DATA, SHORT), repair_reply(GOOD)],
        [full_draft(DATA, SHORT), repair_reply(SHORT)],
        ["{"],
        [{"seo_title": 1}],
    ],
    ids=["complete", "repaired", "repair_failed", "unparseable", "wrong_type"],
)
async def test_confirmed_specifications_are_never_lost(steps: list[Any]) -> None:
    result, _client = await _generate(*steps)

    _assert_specs_kept(result)
    assert result.missing_fields == []


# 9–11. Grounding сохранен: числа, единицы, гарантии, сертификаты.


def _with_sentence(sentence: str) -> str:
    return f"{GOOD} {sentence}"


FORBIDDEN_SENTENCES = [
    ("invented_pressure", "Рабочее давление прибора составляет 2,5 МПа."),
    ("invented_accuracy", "Класс точности повышенный и равен 0,6."),
    ("invented_year", "Модель выпускается с 2019 года."),
    ("unit_swap_bar", "Диапазон показаний 0...1,6 бар подходит для типовых задач."),
    ("unit_swap_kpa", "Диапазон показаний до 1,6 кПа охватывает типовые задачи."),
    ("swapped_values", "Степень защиты 1,5 и класс точности IP54."),
    ("warranty_digits", "Гарантия производителя составляет 12 месяцев."),
    ("warranty_words", "На прибор предоставляется гарантия производителя."),
    ("certificate", "Прибор сертифицирован и имеет сертификат соответствия."),
    ("standard", "Прибор изготовлен по ГОСТ и допущен к эксплуатации."),
    ("delivery", "Бесплатная доставка по всей стране."),
    ("superlative", "Это лучший манометр на рынке."),
    ("industry", "Манометр применяется в нефтяной и химической промышленности."),
]


@pytest.mark.parametrize(
    "sentence",
    [sentence for _, sentence in FORBIDDEN_SENTENCES],
    ids=[case_id for case_id, _ in FORBIDDEN_SENTENCES],
)
async def test_ungrounded_claims_are_rejected_after_single_repair(sentence: str) -> None:
    bad = _with_sentence(sentence)
    result, client = await _generate(full_draft(DATA, bad), repair_reply(bad))

    assert result.status is GenerationStatus.NEEDS_REVIEW
    assert result.seo_description is None
    assert sentence not in _published(result)
    assert REJECTED in _codes(result)
    # Числа и подмена единиц — ошибка grounding, повторного запроса нет.
    # Обещание без числа по-прежнему уходит в одно исправление.
    fact_error = bool(grounding_findings(DATA, bad, field_name="seo_description"))
    assert len(client.calls) == (1 if fact_error else 2)
    _assert_specs_kept(result)


@pytest.mark.parametrize(
    "sentence",
    [sentence for _, sentence in FORBIDDEN_SENTENCES],
    ids=[case_id for case_id, _ in FORBIDDEN_SENTENCES],
)
async def test_clean_repair_replaces_rejected_description(sentence: str) -> None:
    bad = _with_sentence(sentence)
    result, client = await _generate(full_draft(DATA, bad), repair_reply(GOOD))

    if grounding_findings(DATA, bad, field_name="seo_description"):
        assert result.seo_description is None
        assert len(client.calls) == 1
        return
    _assert_complete_card(result)
    assert result.seo_description == GOOD
    assert len(client.calls) == 2


async def test_ungrounded_number_is_not_sent_to_repair() -> None:
    bad = _with_sentence("Рабочее давление прибора составляет 2,5 МПа.")
    _result, client = await _generate(full_draft(DATA, bad), repair_reply(GOOD))

    assert len(client.calls) == 1


async def test_service_phrases_in_description_are_rejected() -> None:
    bad = _with_sentence("Описание составлено только по подтвержденным данным.")
    result, _client = await _generate(full_draft(DATA, bad), repair_reply(bad))

    assert result.status is GenerationStatus.NEEDS_REVIEW
    assert result.seo_description is None


# Ложные срабатывания: безопасные переформулировки подтвержденных значений.

SAFE_REWRITES = [
    ("range_words", "Прибор рассчитан на давление от 0 до 1,6 МПа."),
    ("upper_limit", "Верхний предел шкалы составляет 1,6 МПа."),
    ("thread_head", "Для подключения нужна резьба M20."),
    ("short_model_code", "Модель Демо-01 удобна для сравнения."),
    ("decimal_point", "Класс точности равен 1.5, это указано в маркировке."),
]


@pytest.mark.parametrize(
    "sentence",
    [sentence for _, sentence in SAFE_REWRITES],
    ids=[case_id for case_id, _ in SAFE_REWRITES],
)
async def test_safe_rewrites_of_confirmed_values_are_accepted(sentence: str) -> None:
    core = f"{MANOMETER_SENTENCES[0]} {sentence}"
    text = with_length(core, 950)
    result, client = await _generate(full_draft(DATA, text))

    _assert_complete_card(result)
    assert result.seo_description == text
    assert len(client.calls) == 1


UNSAFE_REWRITES = [
    "Прибор рассчитан на давление от 0 до 1,6 бар.",
    "Прибор рассчитан на давление от 0 до 16 МПа.",
    "Гарантия 0 лет.",
    "Для подключения нужна резьба M30.",
    "Для подключения нужна резьба M20x1.",
]


@pytest.mark.parametrize("sentence", UNSAFE_REWRITES)
async def test_unsafe_rewrites_stay_rejected(sentence: str) -> None:
    text = with_length(f"{MANOMETER_SENTENCES[0]} {sentence}", 950)
    result, _client = await _generate(full_draft(DATA, text), repair_reply(text))

    assert result.status is GenerationStatus.NEEDS_REVIEW
    assert sentence not in _published(result)


# Остальные поля карточки.


async def test_invented_number_in_title_or_queries_is_dropped() -> None:
    draft = full_draft(
        DATA,
        seo_title=f"{DATA.name} 2,5 МПа",
        search_queries=[
            f"{DATA.name} купить",
            f"{DATA.name} 0,05 класс",
            f"{DATA.manufacturer} манометр",
            f"{DATA.name} цена",
        ],
    )
    result, _client = await _generate(draft)

    _assert_complete_card(result)
    assert "2,5" not in result.seo_title
    assert all("0,05" not in query for query in result.search_queries)
    assert result.search_queries == [
        f"{DATA.name} купить",
        f"{DATA.manufacturer} манометр",
        f"{DATA.name} цена",
    ]


# 11, 14. Коммерческий блок без неподтвержденных условий.

COMMERCIAL_CLAIMS = [
    "Доставка за 3 дня, гарантия 5 лет, скидка при заказе от 10 штук.",
    "Гарантия производителя и сертификат соответствия прилагаются.",
    "Прибор всегда в наличии на складе.",
    "Бесплатная доставка и самые низкие цены.",
]


@pytest.mark.parametrize("claim", COMMERCIAL_CLAIMS)
async def test_commercial_block_does_not_keep_unconfirmed_terms(claim: str) -> None:
    draft = full_draft(DATA, commercial_block=f"{DATA.name}. {claim}")
    result, client = await _generate(draft)

    _assert_complete_card(result)
    commercial = (result.commercial_block or "").casefold()
    for stem in ("гаранти", "сертиф", "доставк", "скидк", "наличи", "3 дня", "5 лет", "10 штук"):
        assert stem not in commercial
    assert DATA.attributes[0].value in (result.commercial_block or "")
    assert len(client.calls) == 1


# 12–13. Заголовок и поисковые запросы.

LONG_NAME = "Манометр демонстрационный МП-Демо-01 для учебных стендов и лабораторных показов"


def _long_name_input() -> GenerationInput:
    return GenerationInput(
        name=LONG_NAME,
        manufacturer=DATA.manufacturer,
        attributes=list(DATA.attributes),
    )


async def test_seo_title_never_exceeds_limit() -> None:
    data = _long_name_input()
    assert len(data.name) > SEO_TITLE_MAX_LENGTH
    draft = full_draft(data, seo_title=LONG_NAME)
    result, _client = await _generate(draft, data=data)

    _assert_complete_card(result)
    assert 0 < len(result.seo_title) <= SEO_TITLE_MAX_LENGTH
    assert data.name.startswith(result.seo_title)


@pytest.mark.parametrize("count", [3, 4, 5])
async def test_search_queries_in_range_are_kept(count: int) -> None:
    tails = ("купить", "цена", "характеристики", "паспорт", "заказ")
    queries = [f"{DATA.name} {tail}" for tail in tails]
    result, _client = await _generate(full_draft(DATA, search_queries=queries[:count]))

    _assert_complete_card(result)
    assert result.search_queries == queries[:count]


async def test_extra_search_queries_are_cut_to_maximum() -> None:
    queries = [f"{DATA.name} {tail}" for tail in "абвгдеж"]
    result, _client = await _generate(full_draft(DATA, search_queries=queries))

    _assert_complete_card(result)
    assert result.search_queries == queries[:SEARCH_QUERIES_MAX]


async def test_too_few_grounded_queries_need_review_but_keep_description() -> None:
    draft = full_draft(DATA, search_queries=[f"{DATA.name} купить", "купить дешево", "цена"])
    result, client = await _generate(draft)

    assert result.status is GenerationStatus.NEEDS_REVIEW
    assert result.seo_description == GOOD
    assert result.search_queries == [f"{DATA.name} купить"]
    assert GenerationWarningCode.SEARCH_QUERIES_INSUFFICIENT in _codes(result)
    assert len(client.calls) == 1


# Нет бесконечных повторов и нет судьи в обычной генерации.


@pytest.mark.parametrize(
    "steps",
    [
        [full_draft(DATA, SHORT), repair_reply(SHORT)],
        [full_draft(DATA, long_text(GOOD, 1500))],
        ["{"],
    ],
)
async def test_model_is_called_at_most_twice(steps: list[Any]) -> None:
    _result, client = await _generate(*steps)

    assert len(client.calls) <= 2


def test_generation_services_do_not_use_evaluation_or_judge() -> None:
    root = Path(__file__).resolve().parents[2] / "app"
    for path in [*(root / "services").glob("*.py"), root / "prompts" / "generation.py"]:
        source = path.read_text(encoding="utf-8")
        assert "app.evals" not in source, path
        assert "judge" not in source.casefold(), path


# Контракт схемы: отсутствие описания явное, фиктивный текст не нужен.


def test_schema_allows_missing_description_only_for_partial_status() -> None:
    base: dict[str, Any] = {
        "seo_title": "Манометр",
        "seo_description": None,
        "commercial_block": None,
        "search_queries": [],
    }
    partial = GenerationResult.model_validate({**base, "status": "needs_review"})
    assert partial.seo_description is None
    closed = GenerationResult.model_validate({**base, "status": "insufficient_data"})
    assert closed.commercial_block is None
    with pytest.raises(ValueError, match="seo_description"):
        GenerationResult.model_validate(
            {
                **base,
                "status": "complete",
                "commercial_block": "Блок",
                "search_queries": ["а", "б", "в"],
            }
        )


def test_schema_rejects_blank_description_text() -> None:
    with pytest.raises(ValueError, match="seo_description"):
        GenerationResult.model_validate(
            {
                "status": "needs_review",
                "seo_title": "Манометр",
                "seo_description": "   ",
                "commercial_block": None,
            }
        )


def test_input_attribute_helper_is_valid() -> None:
    item = attribute("Класс точности", "1,5")
    assert item.value == "1,5"
