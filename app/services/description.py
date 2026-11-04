"""Проверка и безопасное сокращение SEO-описания.

Описание допустимо, если оно укладывается в 800–1200 символов, не содержит
неподтвержденных чисел, обещаний и служебных фраз и не набирает объем повторами.
Слишком длинный текст сокращается только по границе предложения.
"""

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

from app.schemas.product import (
    SEO_DESCRIPTION_MAX_LENGTH,
    SEO_DESCRIPTION_MIN_LENGTH,
    GenerationInput,
)
from app.services.grounding import (
    Attempt,
    FindingKind,
    GroundingFinding,
    grounding_findings,
    log_findings,
    mentions_of,
    unsupported_claim,
)

logger = logging.getLogger("uvicorn.error")
_LOGGED_TEXT_LIMIT = 2000

# Больше стольких упоминаний одного значения в тексте — набор объема повтором.
MAX_VALUE_MENTIONS = 3
_MIN_VALUE_LENGTH = 3
_MIN_SENTENCE_LENGTH = 12
_MIN_WORDS_FOR_SIMILARITY = 6
_SIMILARITY_LIMIT = 0.8
_QUOTE_LIMIT = 60

_SERVICE_PHRASES = re.compile(
    "|".join(
        (
            r"подтвержд",
            r"недостаточно\s+данных",
            r"данных\s+недостаточно",
            r"не\s+собран",
            r"генератор",
            r"контент[\s-]*менеджер",
            r"языков\w+\s+модел",
            r"\bseo\b",
            r"\bjson\b",
            r"в\s+(?:исходных|имеющихся|предоставленных|входных)\s+данных",
            r"(?:исходн|предоставленн|входн)\w+\s+данн",
        )
    ),
    re.IGNORECASE,
)
_ABBREVIATIONS = frozenset(
    {"т", "см", "мм", "кг", "мг", "им", "др", "пр", "рис", "ул", "стр", "напр", "гост"}
)


class DescriptionIssue(StrEnum):
    """Что именно не так с описанием. Влияет на код предупреждения и запрос исправления."""

    EMPTY = "empty"
    UNGROUNDED = "ungrounded"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    SERVICE_TEXT = "service_text"
    REPEATED = "repeated"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"


@dataclass(frozen=True)
class DescriptionCheck:
    """Результат проверки. text заполнен только у пригодного к публикации описания."""

    text: str | None
    issue: DescriptionIssue | None
    problems: tuple[str, ...]
    length: int

    @property
    def ok(self) -> bool:
        return self.issue is None


def check_description(
    data: GenerationInput,
    raw: str,
    *,
    attempt: Attempt = Attempt.INITIAL,
) -> DescriptionCheck:
    """Проверить описание и при необходимости сократить его по границе предложения.

    При отказе в DEBUG-лог пишется каждая причина: номер попытки, отклоненное
    значение, фрагмент текста и характеристика. Публичный результат остается прежним.
    """

    check, findings = _evaluate(data, raw)
    if check.ok:
        logger.debug("description accepted attempt=%s length=%d", attempt.value, check.length)
    else:
        _log_rejection(attempt, raw, check, findings)
    return check


def _log_rejection(
    attempt: Attempt,
    raw: str,
    check: DescriptionCheck,
    findings: list[GroundingFinding],
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    log_findings(findings, attempt)
    logger.debug(
        "description rejected attempt=%s field=seo_description issue=%s length=%d details=%s",
        attempt.value,
        check.issue.value if check.issue else None,
        check.length,
        " | ".join(check.problems),
    )
    logger.debug(
        "description rejected attempt=%s text=%r",
        attempt.value,
        raw.strip()[:_LOGGED_TEXT_LIMIT],
    )


def _evaluate(data: GenerationInput, raw: str) -> tuple[DescriptionCheck, list[GroundingFinding]]:
    text = raw.strip()
    if not text or _is_repeated_block(text):
        empty = _failed(DescriptionIssue.EMPTY, "описание пустое или состоит из одного символа", 0)
        return empty, []

    if len(text) > SEO_DESCRIPTION_MAX_LENGTH:
        shortened = shorten_at_sentence(text)
        if shortened is None:
            problem = (
                f"длина {len(text)} символов, допустимо "
                f"{SEO_DESCRIPTION_MIN_LENGTH}–{SEO_DESCRIPTION_MAX_LENGTH}; "
                "безопасного места для сокращения по границе предложения нет"
            )
            return _failed(DescriptionIssue.TOO_LONG, problem, len(text)), []
        text = shortened

    problems, findings = _content_problems(data, text)
    if len(text) < SEO_DESCRIPTION_MIN_LENGTH:
        problems.append(
            (
                DescriptionIssue.TOO_SHORT,
                f"длина {len(text)} символов, нужно {SEO_DESCRIPTION_MIN_LENGTH}–"
                f"{SEO_DESCRIPTION_MAX_LENGTH}",
            )
        )
    if problems:
        issue = problems[0][0]
        details = tuple(detail for _, detail in problems)
        failed = DescriptionCheck(text=None, issue=issue, problems=details, length=len(text))
        return failed, findings
    return DescriptionCheck(text=text, issue=None, problems=(), length=len(text)), findings


def shorten_at_sentence(text: str) -> str | None:
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


def _failed(issue: DescriptionIssue, problem: str, length: int) -> DescriptionCheck:
    return DescriptionCheck(text=None, issue=issue, problems=(problem,), length=length)


def _content_problems(
    data: GenerationInput, text: str
) -> tuple[list[tuple[DescriptionIssue, str]], list[GroundingFinding]]:
    problems: list[tuple[DescriptionIssue, str]] = []
    findings = grounding_findings(data, text, field_name="seo_description")
    number = next((item.value for item in findings if item.kind is FindingKind.NUMBER), None)
    if number is not None:
        problems.append(
            (
                DescriptionIssue.UNGROUNDED,
                f"число «{number}» не подтверждено характеристиками",
            )
        )
    elif findings:
        problems.append(
            (
                DescriptionIssue.UNGROUNDED,
                "значение приписано не той характеристике или взято из конфликтующего источника",
            )
        )
    claim = unsupported_claim(data, text)
    if claim is not None:
        problems.append(
            (
                DescriptionIssue.UNSUPPORTED_CLAIM,
                f"фраза «{claim}» — обещание или применение, которого нет в данных",
            )
        )
    service = _SERVICE_PHRASES.search(text)
    if service is not None:
        problems.append(
            (
                DescriptionIssue.SERVICE_TEXT,
                f"служебная фраза «{service.group()}» не нужна покупателю",
            )
        )
    repeated = _repetition(data, text)
    if repeated is not None:
        problems.append((DescriptionIssue.REPEATED, repeated))
    return problems, findings


def _repetition(data: GenerationInput, text: str) -> str | None:
    """Повтор предложений или характеристик ради объема."""

    seen: list[tuple[str, frozenset[str]]] = []
    for sentence in _sentences(text):
        normalized = " ".join(re.findall(r"\w+", sentence.casefold()))
        if len(normalized) < _MIN_SENTENCE_LENGTH:
            continue
        words = frozenset(normalized.split())
        for earlier, earlier_words in seen:
            if normalized == earlier or _similar(words, earlier_words):
                return f"повторяется предложение «{sentence[:_QUOTE_LIMIT]}»"
        seen.append((normalized, words))
    for item in data.attributes:
        value = item.value.strip()
        if len(value) < _MIN_VALUE_LENGTH:
            continue
        count = mentions_of(value, text)
        if count > MAX_VALUE_MENTIONS:
            return f"значение «{value}» повторено {count} раз"
    return None


def _similar(left: frozenset[str], right: frozenset[str]) -> bool:
    if min(len(left), len(right)) < _MIN_WORDS_FOR_SIMILARITY:
        return False
    return len(left & right) / len(left | right) >= _SIMILARITY_LIMIT


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


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for end in _sentence_ends(text):
        parts.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


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
