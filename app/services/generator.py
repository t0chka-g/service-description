"""Генерация SEO-контента карточки товара по проверенному результату Extractor."""

import json
import re
from collections import Counter
from typing import assert_never

from pydantic import BaseModel, ConfigDict, ValidationError

from app.clients.llm import LLMClient, LLMError, LLMErrorKind
from app.prompts.generation import build_generation_prompt
from app.schemas.product import (
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
        try:
            raw = await self._llm_client.complete(
                build_generation_prompt(data),
                response_model=GenerationDraft,
            )
        except TimeoutError as exc:
            raise LLMError("LLM request timed out", kind=LLMErrorKind.API) from exc
        return _from_complete_gate(data, _parse_generation_response(raw))


def _parse_generation_response(raw: str) -> GenerationDraft:
    try:
        return GenerationDraft.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMError(
            "LLM response does not match the generation schema",
            kind=LLMErrorKind.MODEL,
        ) from exc


def _from_complete_gate(data: GenerationInput, draft: GenerationDraft) -> GenerationResult:
    if not draft.seo_description.strip():
        raise LLMError("seo_description is empty", kind=LLMErrorKind.MODEL)
    _reject_invalid_shape(draft, description=draft.seo_description)
    if _is_repeated_block(draft.seo_description):
        raise LLMError(_REPEATED, kind=LLMErrorKind.MODEL)
    _reject_ungrounded(
        data,
        draft.seo_title,
        draft.seo_description,
        draft.commercial_block,
        draft.search_queries,
    )
    return _result(
        data,
        status=GenerationStatus.COMPLETE,
        title=draft.seo_title,
        description=draft.seo_description,
        queries=list(draft.search_queries),
        commercial=draft.commercial_block,
    )


def _reject_invalid_shape(draft: GenerationDraft, *, description: str) -> None:
    """Проверить ответ по схеме полной карточки, без подмены текста."""

    try:
        GenerationResult(
            status=GenerationStatus.COMPLETE,
            seo_title=draft.seo_title,
            seo_description=description,
            search_queries=list(draft.search_queries),
            commercial_block=draft.commercial_block,
        )
    except ValidationError as exc:
        raise LLMError(str(exc), kind=LLMErrorKind.MODEL) from exc


def _reject_ungrounded(
    data: GenerationInput,
    title: str,
    description: str,
    commercial: str,
    queries: list[str],
) -> None:
    prose = [title, description, commercial]
    fields = [*prose, *queries]
    catalog_title = trusted_product_title(data.name)
    fact_blobs = _fact_blobs(data)
    identity_blobs = _identity_blobs(data)
    title_numbers = set(_NUMBER.findall(catalog_title))
    for field in prose:
        if _is_cut_title(field, catalog_title):
            continue
        _reject_ungrounded_numbers(field, fact_blobs, identity_blobs)
    for query in queries:
        _reject_ungrounded_numbers(
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
    field: str,
    fact_blobs: list[str],
    identity_blobs: list[str],
    *,
    free_tokens: set[str] | None = None,
) -> None:
    allowed_free = free_tokens or set()
    for start, end, token in _number_spans(field):
        if _span_covered(field, start, end, fact_blobs):
            continue
        if _span_covered(field, start, end, identity_blobs):
            continue
        if token in allowed_free:
            continue
        raise LLMError(_UNGROUNDED, kind=LLMErrorKind.MODEL)


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
