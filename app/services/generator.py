"""Генерация SEO-контента карточки товара по проверенному результату Extractor."""

import json
import logging
import re
from collections import Counter
from typing import assert_never

from pydantic import BaseModel, ConfigDict, ValidationError

from app.clients.llm import LLMClient, LLMError, LLMErrorKind
from app.prompts.generation import build_generation_prompt
from app.schemas.product import (
    SEARCH_QUERIES_MAX,
    SEARCH_QUERIES_MIN,
    SEO_DESCRIPTION_MAX_LENGTH,
    SEO_DESCRIPTION_MIN_LENGTH,
    SEO_TITLE_MAX_LENGTH,
    GenerationInput,
    GenerationResult,
    GenerationStatus,
    GenerationWarning,
    GenerationWarningCode,
    assess_generation_status,
    trusted_product_title,
)

_CLOSED_TITLE = "Товар"
_CLOSED_NOTE = (
    "Полное описание не собрано: подтвержденных данных недостаточно, "
    "неподтвержденные характеристики не добавляются."
)
_CLOSED_COMMERCIAL = "Коммерческие условия не указаны: подтвержденных характеристик нет."
# Целый числовой токен: цифры и разделители диапазона. Части разных значений не склеиваются.
_NUMBER = re.compile(
    r"(?:[+\-−±])?\d+(?:[,.]\d+)?(?:(?:\.{2,3}|…|[/\-–−])(?:[+\-−])?\d+(?:[,.]\d+)?)*"
)
_REPEATED = "seo_description is a repeated block"
_UNGROUNDED = "ungrounded content in the model response"
_UNGROUNDED_REASON = "число не лежит целиком внутри подтвержденного значения или названия"
logger = logging.getLogger("uvicorn.error")


class GenerationDraft(BaseModel):
    """Сырой ответ модели. Статус, характеристики и предупреждения подставляет сервис."""

    model_config = ConfigDict(extra="ignore")

    status: GenerationStatus
    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial_block: str


class SEOGenerator:
    """Черновик SEO-карточки из подтвержденных характеристик.

    Генератор не читает исходное техническое описание и не извлекает
    характеристики заново. При закрытом шлюзе модель не вызывается.
    Короткий, длинный или повторяющийся ответ модели — ошибка контракта,
    а не признак нехватки входных данных.

    Детерминированная проверка чисел смотрит на подтвержденные пары
    attribute→value и на название как идентификатор: короткий токен из
    маркировки не делает чужое утверждение допустимым, смена единицы при
    том же числе отклоняется. Семантика произвольного текста — зона LLM Evaluation.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def generate(self, data: GenerationInput) -> GenerationResult:
        """Собрать карточку из подтвержденных полей и ответа модели."""

        status = assess_generation_status(data)
        if status is GenerationStatus.INSUFFICIENT_DATA:
            return _closed_result(data)
        if status is GenerationStatus.COMPLETE:
            return await self._generate_complete(data)
        assert_never(status)

    async def _generate_complete(self, data: GenerationInput) -> GenerationResult:
        draft = await self._complete_draft(build_generation_prompt(data))
        if draft is None:
            return _review_result(data)
        return _from_complete_gate(data, draft)

    async def _complete_draft(self, prompt: str) -> GenerationDraft | None:
        try:
            raw = await self._llm_client.complete(prompt, response_model=GenerationDraft)
        except TimeoutError as exc:
            raise LLMError("LLM request timed out", kind=LLMErrorKind.API) from exc
        try:
            return _parse_generation_response(raw)
        except LLMError:
            return None


def _parse_generation_response(raw: str) -> GenerationDraft:
    try:
        return GenerationDraft.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMError(
            "LLM response does not match the generation schema",
            kind=LLMErrorKind.MODEL,
        ) from exc


def _from_complete_gate(data: GenerationInput, draft: GenerationDraft) -> GenerationResult:
    """Собрать карточку. Длину чинит обрезка, не второй вызов модели."""

    description = _candidate_description(data, draft.seo_description)
    title = _candidate_title(data, draft.seo_title)
    commercial = _candidate_commercial(data, draft.commercial_block)
    queries = _candidate_queries(data, draft.search_queries)
    if (
        description is not None
        and SEO_DESCRIPTION_MIN_LENGTH <= len(description) <= SEO_DESCRIPTION_MAX_LENGTH
        and len(queries) >= SEARCH_QUERIES_MIN
        and _package_ok(data, title, description, commercial, queries)
    ):
        return _result(
            data,
            status=GenerationStatus.COMPLETE,
            title=title,
            description=description,
            queries=queries,
            commercial=commercial,
        )
    if description is None:
        description, fallback_commercial = _closed_copy(data)
        if commercial == draft.commercial_block.strip() and not _field_ok(data, commercial):
            commercial = fallback_commercial
    return _result(
        data,
        status=GenerationStatus.NEEDS_REVIEW,
        title=title,
        description=description,
        queries=queries,
        commercial=commercial,
    )


def _candidate_description(data: GenerationInput, raw: str) -> str | None:
    if not raw.strip() or _is_repeated_block(raw.strip()):
        return None
    text = raw
    if len(text) > SEO_DESCRIPTION_MAX_LENGTH:
        text = _trim_at_sentence(text.strip()) or ""
        if not text:
            return None
    if not _field_ok(data, text):
        return None
    return text


def _candidate_title(data: GenerationInput, raw: str) -> str:
    title = raw.strip()
    fallback = _closed_title(data)
    catalog = trusted_product_title(data.name)
    if title and len(title) <= SEO_TITLE_MAX_LENGTH and (
        title == catalog or _is_cut_title(title, catalog) or title == fallback
    ):
        return title
    if not title or len(title) > SEO_TITLE_MAX_LENGTH or not _field_ok(data, title):
        return fallback
    return title


def _candidate_commercial(data: GenerationInput, raw: str) -> str:
    text = raw.strip()
    if text and _field_ok(data, text):
        return text
    return _closed_copy(data)[1]


def _candidate_queries(data: GenerationInput, queries: list[str]) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = query.strip()
        folded = text.casefold()
        if not text or folded in seen or not _query_is_grounded(text, data):
            continue
        if not _query_numbers_ok(data, text):
            continue
        seen.add(folded)
        kept.append(text)
        if len(kept) == SEARCH_QUERIES_MAX:
            break
    return kept


def _query_numbers_ok(data: GenerationInput, text: str) -> bool:
    title = trusted_product_title(data.name)
    free_tokens = {_fold_technical(token) for token in _NUMBER.findall(title)}
    try:
        _reject_ungrounded_numbers(
            "search_query",
            text,
            _fact_blobs(data),
            _identity_blobs(data),
            free_tokens=free_tokens,
        )
    except LLMError:
        return False
    return not _has_conflict_value(data, [text]) and not _has_swapped_value(data, [text])


def _package_ok(
    data: GenerationInput,
    title: str,
    description: str,
    commercial: str,
    queries: list[str],
) -> bool:
    try:
        _reject_ungrounded(data, title, description, commercial, queries)
    except LLMError:
        return False
    return True


def _field_ok(data: GenerationInput, text: str) -> bool:
    try:
        _reject_ungrounded_numbers("field", text, _fact_blobs(data), _identity_blobs(data))
    except LLMError:
        return False
    return not _has_conflict_value(data, [text]) and not _has_swapped_value(data, [text])


def _trim_at_sentence(text: str) -> str | None:
    """Последний конец предложения внутри 800–1200 символов, без разрезания значений."""

    ends = [
        pos
        for pos in _sentence_ends(text)
        if SEO_DESCRIPTION_MIN_LENGTH <= pos <= SEO_DESCRIPTION_MAX_LENGTH
    ]
    if not ends:
        return None
    trimmed = text[: max(ends)].rstrip()
    if not SEO_DESCRIPTION_MIN_LENGTH <= len(trimmed) <= SEO_DESCRIPTION_MAX_LENGTH:
        return None
    return trimmed


def _sentence_ends(text: str) -> list[int]:
    ends: list[int] = []
    for match in re.finditer(r"[.!?…](?=\s|$)", text):
        punct_at = match.start()
        if text[punct_at] == "." and _is_internal_dot(text, punct_at):
            continue
        ends.append(match.end())
    return ends


def _is_internal_dot(text: str, index: int) -> bool:
    if index > 0 and text[index - 1] == ".":
        return True
    if (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    ):
        return True
    word = text[:index].split()[-1] if text[:index].split() else ""
    token = word.casefold().strip(".")
    if token in _ABBREVIATIONS:
        return True
    return len(word) <= 2 and word.isalpha()


_ABBREVIATIONS = frozenset(
    {
        "т",
        "см",
        "мм",
        "кг",
        "мг",
        "им",
        "др",
        "пр",
        "рис",
        "ул",
        "стр",
        "напр",
        "гост",
    }
)


def _reject_ungrounded(
    data: GenerationInput,
    title: str,
    description: str,
    commercial: str,
    queries: list[str],
) -> None:
    prose = [
        ("seo_title", title),
        ("seo_description", description),
        ("commercial_block", commercial),
    ]
    fields = [text for _, text in prose]
    fields.extend(queries)
    catalog_title = trusted_product_title(data.name)
    fact_blobs = _fact_blobs(data)
    identity_blobs = _identity_blobs(data)
    title_numbers = {_fold_technical(token) for token in _NUMBER.findall(catalog_title)}
    for name, field in prose:
        if _is_cut_title(field, catalog_title):
            continue
        _reject_ungrounded_numbers(name, field, fact_blobs, identity_blobs)
    for index, query in enumerate(queries):
        _reject_ungrounded_numbers(
            f"search_queries[{index}]",
            query,
            fact_blobs,
            identity_blobs,
            free_tokens=title_numbers,
        )
    if _has_swapped_value(data, fields) or _has_conflict_value(data, fields):
        raise LLMError(_UNGROUNDED, kind=LLMErrorKind.MODEL)
    if any(not _query_is_grounded(query, data) for query in queries):
        raise LLMError(_UNGROUNDED, kind=LLMErrorKind.MODEL)


def _reject_ungrounded_numbers(
    field_name: str,
    field: str,
    fact_blobs: list[str],
    identity_blobs: list[str],
    *,
    free_tokens: set[str] | None = None,
) -> None:
    """Отклонить число, которое не входит в подтвержденный фрагмент.

    Сверка идет по нормализованной записи: запятая и точка в десятичной дроби,
    разделители диапазона, знаки умножения в резьбе и пробел в IP54 эквивалентны.
    Единицы и чужие числа из-за этого не становятся допустимыми.
    """

    folded = _fold_technical(field)
    folded_facts = [_fold_technical(blob) for blob in fact_blobs]
    folded_identity = [_fold_technical(blob) for blob in identity_blobs]
    allowed_free = free_tokens or set()
    for start, end, token in _number_spans(folded):
        if _span_covered(folded, start, end, folded_facts):
            continue
        if _span_covered(folded, start, end, folded_identity):
            continue
        if token in allowed_free:
            continue
        snippet = folded[max(0, start - 30) : min(len(folded), end + 30)]
        logger.warning(
            "ungrounded number field=%s token=%s snippet=%s reason=%s allowed=%s",
            field_name,
            token,
            snippet,
            _UNGROUNDED_REASON,
            fact_blobs,
        )
        raise LLMError(
            f"{_UNGROUNDED}: {field_name} contains {token}",
            kind=LLMErrorKind.MODEL,
        )


def _fold_technical(text: str) -> str:
    """Свести эквивалентную техническую запись к одному виду, не меняя цифры."""

    folded = text.translate(str.maketrans({"×": "x", "х": "x", "Х": "x"}))
    folded = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "x", folded)
    folded = re.sub(r"(?<=\d)\s*(?:\.{2,3}|…)\s*(?=\d)", "...", folded)
    folded = re.sub(r"(?<=\d)\s*[-–—−]\s*(?=\d)", "...", folded)
    folded = re.sub(r"(?<=\d)\.(?=\d)", ",", folded)
    return re.sub(r"IP[\s-]*(\d{2})\b", r"IP\1", folded, flags=re.IGNORECASE)


def _closed_result(data: GenerationInput) -> GenerationResult:
    description, commercial = _closed_copy(data)
    return _result(
        data,
        status=GenerationStatus.INSUFFICIENT_DATA,
        title=_closed_title(data),
        description=description,
        queries=[],
        commercial=commercial,
    )


def _review_result(data: GenerationInput) -> GenerationResult:
    description, commercial = _closed_copy(data)
    return _result(
        data,
        status=GenerationStatus.NEEDS_REVIEW,
        title=_closed_title(data),
        description=description,
        queries=[],
        commercial=commercial,
    )


def _result(
    data: GenerationInput,
    *,
    status: GenerationStatus,
    title: str,
    description: str,
    queries: list[str],
    commercial: str,
) -> GenerationResult:
    return GenerationResult(
        status=status,
        seo_title=title,
        seo_description=description,
        specifications=list(data.attributes),
        search_queries=queries,
        commercial_block=commercial,
        missing_fields=list(data.missing_fields),
        warnings=_warnings(data, status),
    )


def _warnings(data: GenerationInput, status: GenerationStatus) -> list[GenerationWarning]:
    warnings = [
        GenerationWarning(code=GenerationWarningCode.CONFLICT, field=item.name)
        for item in data.conflicts
    ]
    warnings.extend(
        GenerationWarning(code=GenerationWarningCode.MISSING_FIELD, field=name)
        for name in data.missing_fields
    )
    if status is GenerationStatus.INSUFFICIENT_DATA:
        warnings.append(GenerationWarning(code=GenerationWarningCode.INSUFFICIENT_DATA, field=""))
    elif status is GenerationStatus.NEEDS_REVIEW:
        warnings.append(GenerationWarning(code=GenerationWarningCode.NEEDS_REVIEW, field=""))
    return warnings


def _closed_title(data: GenerationInput) -> str:
    for candidate in (trusted_product_title(data.name), data.manufacturer):
        title = candidate.strip()[:SEO_TITLE_MAX_LENGTH].rstrip()
        if title:
            return title
    return _CLOSED_TITLE


def _closed_commercial(data: GenerationInput) -> str:
    if not data.attributes:
        return _CLOSED_COMMERCIAL
    item = data.attributes[0]
    product = trusted_product_title(data.name) or _CLOSED_TITLE
    text = f"{product}: {item.name} {item.value}."
    manufacturer = data.manufacturer.strip()
    if manufacturer:
        text += f" Производитель: {manufacturer}."
    return text


def _closed_copy(data: GenerationInput) -> tuple[str, str]:
    """Короткая сводка. Значение характеристики не обрезается посередине."""

    facts = [f"{item.name}: {item.value}." for item in data.attributes if item.value.strip()]
    parts: list[str] = []
    title = trusted_product_title(data.name)
    if title and len(title) < SEO_DESCRIPTION_MIN_LENGTH:
        parts.append(title)
    manufacturer = data.manufacturer.strip()
    if manufacturer:
        parts.append(f"Производитель: {manufacturer}.")
    if data.missing_fields:
        parts.append("Не подтверждено: " + ", ".join(data.missing_fields) + ".")
    parts.append(_CLOSED_NOTE)
    description = " ".join(parts).strip()
    overflow: list[str] = []
    for fact in facts:
        candidate = f"{description} {fact}".strip()
        if len(candidate) < SEO_DESCRIPTION_MIN_LENGTH:
            description = candidate
        else:
            overflow.append(fact)
    commercial = _closed_commercial(data)
    if overflow:
        commercial = f"{commercial} {' '.join(overflow)}".strip()
    if len(description) >= SEO_DESCRIPTION_MIN_LENGTH:
        description = _CLOSED_NOTE
    return description, commercial


def _is_repeated_block(description: str) -> bool:
    """Описание из одного повторяющегося блока в 1–2 символа."""

    text = description.strip()
    for size in (1, 2):
        if len(text) % size:
            continue
        unit = text[:size]
        if unit and text == unit * (len(text) // size):
            return True
    return False


def _number_spans(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group()) for match in _NUMBER.finditer(text)]


def _fact_blobs(data: GenerationInput) -> list[str]:
    """Подтвержденные значения характеристик целиком, с единицами."""

    return [item.value.strip() for item in data.attributes if item.value.strip()]


def _identity_blobs(data: GenerationInput) -> list[str]:
    """Название и производитель как идентификаторы, не как склад технических чисел."""

    blobs: list[str] = []
    title = trusted_product_title(data.name)
    if title:
        blobs.append(title)
    manufacturer = data.manufacturer.strip()
    if manufacturer:
        blobs.append(manufacturer)
    return blobs


def _span_covered(text: str, start: int, end: int, blobs: list[str]) -> bool:
    """Число допустимо, только если целиком лежит внутри копии подтвержденного фрагмента."""

    for blob in blobs:
        if not blob:
            continue
        index = 0
        while True:
            pos = text.find(blob, index)
            if pos < 0:
                break
            if pos <= start and end <= pos + len(blob):
                return True
            index = pos + 1
    return False


def _is_cut_title(field: str, catalog_title: str) -> bool:
    """Поле — начало названия, обрезанное лимитом заголовка, а не новое число."""

    compact = field.strip()
    return bool(compact) and compact != catalog_title and catalog_title.startswith(compact)


def _exclusive_numbers(data: GenerationInput) -> dict[str, set[str]]:
    per_name = {
        item.name: set(_NUMBER.findall(item.value))
        for item in data.attributes
        if item.name.strip() and item.value.strip()
    }
    counts = Counter(token for tokens in per_name.values() for token in tokens)
    return {
        name: {token for token in tokens if counts[token] == 1}
        for name, tokens in per_name.items()
    }


def _has_swapped_value(data: GenerationInput, fields: list[str]) -> bool:
    exclusive = _exclusive_numbers(data)
    names = [item.name for item in data.attributes if item.name.strip()]
    for field in fields:
        for name, window in _attribute_windows(field, names):
            for token in _NUMBER.findall(window):
                owner = next(
                    (item for item, tokens in exclusive.items() if token in tokens),
                    None,
                )
                if owner is not None and owner.casefold() != name.casefold():
                    return True
    return False


def _attribute_windows(text: str, names: list[str]) -> list[tuple[str, str]]:
    occupied = bytearray(len(text))
    hits: list[tuple[int, int, str]] = []
    for name in sorted(names, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            if any(occupied[match.start() : match.end()]):
                continue
            occupied[match.start() : match.end()] = b"\x01" * (match.end() - match.start())
            hits.append((match.start(), match.end(), name))
    hits.sort()
    windows: list[tuple[str, str]] = []
    for index, (_start, end, name) in enumerate(hits):
        next_start = hits[index + 1][0] if index + 1 < len(hits) else len(text)
        windows.append((name, text[end:next_start]))
    return windows


def _has_conflict_value(data: GenerationInput, fields: list[str]) -> bool:
    for conflict in data.conflicts:
        foreign = conflict.description_value.strip()
        confirmed = conflict.specs_value.strip()
        if not foreign or foreign.casefold() == confirmed.casefold():
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(foreign)}(?!\w)", re.IGNORECASE)
        if any(pattern.search(field) for field in fields):
            return True
    return False


def _query_is_grounded(query: str, data: GenerationInput) -> bool:
    folded = query.casefold()
    manufacturer = data.manufacturer.strip()
    if manufacturer and manufacturer.casefold() in folded:
        return True
    title = trusted_product_title(data.name)
    words = title.split()
    if words:
        head = words[0].strip(".,;:").casefold()
        if len(head) >= 4 and head in folded:
            return True
    return any(token.casefold() in folded for token in _NUMBER.findall(title))
