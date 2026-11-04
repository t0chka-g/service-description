"""Генерация SEO-контента карточки товара по проверенному результату Extractor.

Конвейер: один запрос к модели, разбор и проверка полей, при необходимости один
точечный запрос на исправление только seo_description. Публикуемые поля отделены
от диагностики: если готового текста нет, поле остается пустым, а причина
попадает в warnings.
"""

import json
import logging
from typing import assert_never

from pydantic import BaseModel, ConfigDict, ValidationError

from app.clients.llm import LLMClient, LLMError, LLMErrorKind
from app.prompts.generation import build_generation_prompt, build_repair_prompt
from app.schemas.product import (
    COMPLETE_MIN_ATTRIBUTES,
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
from app.services.description import DescriptionCheck, DescriptionIssue, check_description
from app.services.grounding import (
    Attempt,
    grounding_findings,
    is_cut_title,
    log_findings,
    query_findings,
    query_grounded,
    unsupported_claim,
)

_CLOSED_TITLE = "Товар"
logger = logging.getLogger("uvicorn.error")

_DESCRIPTION_CODES: dict[DescriptionIssue, GenerationWarningCode] = {
    DescriptionIssue.EMPTY: GenerationWarningCode.DESCRIPTION_REJECTED,
    DescriptionIssue.UNGROUNDED: GenerationWarningCode.DESCRIPTION_REJECTED,
    DescriptionIssue.UNSUPPORTED_CLAIM: GenerationWarningCode.DESCRIPTION_REJECTED,
    DescriptionIssue.SERVICE_TEXT: GenerationWarningCode.DESCRIPTION_REJECTED,
    DescriptionIssue.REPEATED: GenerationWarningCode.DESCRIPTION_REJECTED,
    DescriptionIssue.TOO_SHORT: GenerationWarningCode.DESCRIPTION_TOO_SHORT,
    DescriptionIssue.TOO_LONG: GenerationWarningCode.DESCRIPTION_TOO_LONG,
}

_PUBLIC_REASONS: dict[DescriptionIssue, str] = {
    DescriptionIssue.EMPTY: "описание пустое",
    DescriptionIssue.UNGROUNDED: "в тексте есть значение, не подтвержденное характеристиками",
    DescriptionIssue.UNSUPPORTED_CLAIM: (
        "в тексте есть обещание или область применения, которых нет в данных"
    ),
    DescriptionIssue.SERVICE_TEXT: "в тексте есть служебная фраза",
    DescriptionIssue.REPEATED: "текст набирает объем повторами",
    DescriptionIssue.TOO_SHORT: "длина {length} символов, нужно {low}–{high}",
    DescriptionIssue.TOO_LONG: (
        "длина {length} символов, нужно {low}–{high}, безопасно сократить текст не удалось"
    ),
}


class GenerationDraft(BaseModel):
    """Сырой ответ модели. Статус, характеристики и предупреждения подставляет сервис."""

    model_config = ConfigDict(extra="ignore")

    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial_block: str


class DescriptionRepair(BaseModel):
    """Ответ модели на точечное исправление описания."""

    model_config = ConfigDict(extra="ignore")

    seo_description: str


class SEOGenerator:
    """Черновик SEO-карточки из подтвержденных характеристик.

    Генератор не читает исходное техническое описание и не извлекает
    характеристики заново. При закрытом шлюзе модель не вызывается.
    Не больше двух запросов к модели. Второй запрос бывает только если описание
    не прошло проверку длины, повторов или служебных фраз. Ошибка grounding
    повторный запрос не вызывает.
    Слишком длинный текст сначала сокращается по границе предложения без запроса.

    Детерминированная проверка чисел смотрит на подтвержденные пары
    attribute->value и на название как идентификатор: короткий токен из
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
            return _failed_result(data)
        check = check_description(data, draft.seo_description)
        repair_attempted = False
        if not check.ok and check.issue is not DescriptionIssue.UNGROUNDED:
            repair_attempted = True
            check = await self._repair(data, draft.seo_description, check)
        return _assemble(data, draft, check, repair_attempted=repair_attempted)

    async def _complete_draft(self, prompt: str) -> GenerationDraft | None:
        try:
            raw = await self._llm_client.complete(prompt, response_model=GenerationDraft)
        except TimeoutError as exc:
            raise LLMError("LLM request timed out", kind=LLMErrorKind.API) from exc
        try:
            return GenerationDraft.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError):
            logger.warning("generation response does not match the schema")
            return None

    async def _repair(
        self,
        data: GenerationInput,
        description: str,
        failed: DescriptionCheck,
    ) -> DescriptionCheck:
        """Один запрос на исправление описания. Сбой исправления не роняет карточку."""

        prompt = build_repair_prompt(data, description, failed.problems)
        try:
            raw = await self._llm_client.complete(prompt, response_model=DescriptionRepair)
            repaired = DescriptionRepair.model_validate(json.loads(raw))
        except (LLMError, TimeoutError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("description repair failed: %s", type(exc).__name__)
            logger.debug(
                "repair attempt produced no text; keeping initial rejection issue=%s",
                failed.issue.value if failed.issue else None,
            )
            return failed
        if not repaired.seo_description.strip():
            logger.debug("repair attempt returned an empty description")
            return failed
        return check_description(data, repaired.seo_description, attempt=Attempt.REPAIR)


def _assemble(
    data: GenerationInput,
    draft: GenerationDraft,
    check: DescriptionCheck,
    *,
    repair_attempted: bool,
) -> GenerationResult:
    """Собрать результат из независимо проверенных полей."""

    queries = _candidate_queries(data, draft.search_queries)
    reasons: list[GenerationWarning] = []
    if not check.ok:
        reasons.append(_description_warning(check, repair_attempted=repair_attempted))
    if len(queries) < SEARCH_QUERIES_MIN:
        reasons.append(
            GenerationWarning(
                code=GenerationWarningCode.SEARCH_QUERIES_INSUFFICIENT,
                field="search_queries",
                message=(
                    f"Проверку прошли поисковых запросов: {len(queries)}, "
                    f"нужно {SEARCH_QUERIES_MIN}–{SEARCH_QUERIES_MAX}."
                ),
            )
        )
    status = GenerationStatus.NEEDS_REVIEW if reasons else GenerationStatus.COMPLETE
    return _result(
        data,
        status=status,
        title=_candidate_title(data, draft.seo_title),
        description=check.text,
        queries=queries,
        commercial=_candidate_commercial(data, draft.commercial_block),
        reasons=reasons,
    )


def _description_warning(check: DescriptionCheck, *, repair_attempted: bool) -> GenerationWarning:
    """Причина без цитат: чужое число или конфликтное значение в ответ не попадает."""

    assert check.issue is not None
    reason = _PUBLIC_REASONS[check.issue].format(
        length=check.length,
        low=SEO_DESCRIPTION_MIN_LENGTH,
        high=SEO_DESCRIPTION_MAX_LENGTH,
    )
    message = f"Описание не принято: {reason}."
    if repair_attempted:
        message += " Автоматическое исправление не помогло."
    return GenerationWarning(
        code=_DESCRIPTION_CODES[check.issue],
        field="seo_description",
        message=message,
    )


def _candidate_title(data: GenerationInput, raw: str) -> str:
    title = raw.strip()
    fallback = _closed_title(data)
    catalog = trusted_product_title(data.name)
    if title and len(title) <= SEO_TITLE_MAX_LENGTH and (
        title == catalog or is_cut_title(title, catalog) or title == fallback
    ):
        return title
    if not title or len(title) > SEO_TITLE_MAX_LENGTH or not _prose_ok(data, title, "seo_title"):
        return fallback
    return title


def _candidate_commercial(data: GenerationInput, raw: str) -> str:
    text = raw.strip()
    if text and _prose_ok(data, text, "commercial_block"):
        return text
    return _fact_commercial(data)


def _candidate_queries(data: GenerationInput, queries: list[str]) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = query.strip()
        folded = text.casefold()
        if not text or folded in seen:
            continue
        if not query_grounded(text, data):
            logger.debug(
                "query rejected attempt=%s field=search_query reason=not tied to the product "
                "(no manufacturer, first title word or model code) text=%r",
                Attempt.INITIAL.value,
                text,
            )
            continue
        findings = query_findings(data, text)
        if findings:
            log_findings(findings, Attempt.INITIAL)
            continue
        seen.add(folded)
        kept.append(text)
        if len(kept) == SEARCH_QUERIES_MAX:
            break
    return kept


def _prose_ok(data: GenerationInput, text: str, field: str) -> bool:
    findings = grounding_findings(data, text, field_name=field)
    claim = unsupported_claim(data, text)
    log_findings(findings, Attempt.INITIAL)
    if claim is not None:
        logger.debug(
            "claim rejected attempt=%s field=%s value=%r reason=%s snippet=%r",
            Attempt.INITIAL.value,
            field,
            claim,
            "обещание или область применения, которых нет в названии и характеристиках",
            text[:200],
        )
    return not findings and claim is None


def _closed_result(data: GenerationInput) -> GenerationResult:
    confirmed = sum(1 for item in data.attributes if item.value.strip())
    reason = GenerationWarning(
        code=GenerationWarningCode.INSUFFICIENT_DATA,
        message=(
            f"Подтверждено характеристик: {confirmed}, для полной карточки нужно не меньше "
            f"{COMPLETE_MIN_ATTRIBUTES} и название товара. Модель не вызывалась."
        ),
    )
    return _result(
        data,
        status=GenerationStatus.INSUFFICIENT_DATA,
        title=_closed_title(data),
        description=None,
        queries=[],
        commercial=None,
        reasons=[reason],
    )


def _failed_result(data: GenerationInput) -> GenerationResult:
    reason = GenerationWarning(
        code=GenerationWarningCode.GENERATION_FAILED,
        message=(
            "Ответ модели не удалось разобрать как карточку. "
            "Повторный запрос не выполнялся, публикуемых текстов нет."
        ),
    )
    return _result(
        data,
        status=GenerationStatus.NEEDS_REVIEW,
        title=_closed_title(data),
        description=None,
        queries=[],
        commercial=None,
        reasons=[reason],
    )


def _result(
    data: GenerationInput,
    *,
    status: GenerationStatus,
    title: str,
    description: str | None,
    queries: list[str],
    commercial: str | None,
    reasons: list[GenerationWarning],
) -> GenerationResult:
    return GenerationResult(
        status=status,
        seo_title=title,
        seo_description=description,
        specifications=list(data.attributes),
        search_queries=queries,
        commercial_block=commercial,
        missing_fields=list(data.missing_fields),
        warnings=_warnings(data, status, reasons),
    )


def _warnings(
    data: GenerationInput,
    status: GenerationStatus,
    reasons: list[GenerationWarning],
) -> list[GenerationWarning]:
    warnings = [
        GenerationWarning(code=GenerationWarningCode.CONFLICT, field=item.name)
        for item in data.conflicts
    ]
    warnings.extend(
        GenerationWarning(code=GenerationWarningCode.MISSING_FIELD, field=name)
        for name in data.missing_fields
    )
    warnings.extend(reasons)
    if status is GenerationStatus.NEEDS_REVIEW:
        warnings.append(
            GenerationWarning(
                code=GenerationWarningCode.NEEDS_REVIEW,
                message=(
                    "Часть публикуемых полей не прошла проверку. "
                    "Подтвержденные характеристики сохранены."
                ),
            )
        )
    return warnings


def _closed_title(data: GenerationInput) -> str:
    for candidate in (trusted_product_title(data.name), data.manufacturer):
        title = candidate.strip()[:SEO_TITLE_MAX_LENGTH].rstrip()
        if title:
            return title
    return _CLOSED_TITLE


def _fact_commercial(data: GenerationInput) -> str:
    """Строка из подтвержденных данных: название, первая характеристика, производитель."""

    item = next(attribute for attribute in data.attributes if attribute.value.strip())
    product = trusted_product_title(data.name) or _CLOSED_TITLE
    text = f"{product}: {item.name} {item.value}."
    manufacturer = data.manufacturer.strip()
    if manufacturer:
        text += f" Производитель: {manufacturer}."
    return text
