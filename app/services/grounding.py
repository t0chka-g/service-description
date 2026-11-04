"""Детерминированная проверка текста карточки по подтвержденным фактам.

Проверка смотрит на числа и пары «характеристика — значение», на конфликтные
значения и на неподтвержденные обещания. Смысл произвольного текста остается
зоной LLM Evaluation и ревью.

Число допустимо, если оно целиком лежит внутри копии подтвержденного значения
или названия/производителя. Безопасные переформулировки тоже допустимы:

- компонент диапазона с той же единицей: «от 0 до 1,6 МПа» для «0...1,6 МПа»;
- начало или конец кода из названия или значения: «M20» для «M20x1,5»,
  «Демо-01» для «МП-Демо-01», если код не обрывает число посередине.

Смена единицы, чужое число и склейка частей разных значений по-прежнему отклоняются.
"""

import logging
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from app.schemas.product import GenerationInput, trusted_product_title

# Целый числовой токен: цифры и разделители диапазона. Части разных значений не склеиваются.
NUMBER = re.compile(
    r"(?:[+\-−±])?\d+(?:[,.]\d+)?(?:(?:\.{2,3}|…|[/\-–−])(?:[+\-−])?\d+(?:[,.]\d+)?)*"
)
_RANGE_SEPARATOR = "..."
# Слова между компонентами диапазона: «от 0 до 1,6 МПа».
_RANGE_CONNECTORS = frozenset({"до", "и", "по", "включительно"})
_UNIT_AFTER_NUMBER = re.compile(r"\s*([°%]?[^\W\d_]*)")
_MAX_UNIT_LENGTH = 12
_CODE_CHARS = re.compile(r"[\w\-]")

logger = logging.getLogger("uvicorn.error")

_SNIPPET_RADIUS = 40


class Attempt(StrEnum):
    """Какой ответ модели проверяется: основной или исправление описания."""

    INITIAL = "initial"
    REPAIR = "repair"


class FindingKind(StrEnum):
    NUMBER = "number"
    SWAPPED_VALUE = "swapped_value"
    CONFLICT_VALUE = "conflict_value"


@dataclass(frozen=True)
class GroundingFinding:
    """Одно отклоненное значение. Только для внутренних логов, в API не попадает."""

    kind: FindingKind
    field: str
    value: str
    snippet: str
    attribute: str | None
    reason: str


# Неподтвержденные обещания и области применения. Основа слова допустима,
# только если она есть в подтвержденных данных (например, «ГОСТ» в названии).
_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (stem, re.compile(pattern))
    for stem, pattern in (
        ("гаранти", r"гаранти\w*"),
        ("сертифи", r"сертифи\w*"),
        ("декларац", r"декларац\w*\s+соответстви\w*"),
        ("наличи", r"в\s+наличии"),
        ("склад", r"(?:на|со)\s+склад\w*"),
        ("доставк", r"(?:бесплатн|быстр|оперативн|срочн)\w*\s+доставк\w*"),
        ("доставк", r"доставк\w*\s+(?:за|по|в\s+течение|от)\b"),
        ("скидк", r"скидк\w*"),
        ("выгодн цен", r"(?:низк|выгодн|доступн|лучш)\w*\s+цен\w*"),
        ("лучш", r"(?:наи)?лучш\w*"),
        ("уникальн", r"уникальн\w*"),
        ("идеальн", r"идеальн\w*"),
        ("лидер", r"лидер\w*"),
        ("самы", r"\bсам(?:ый|ая|ое|ые|ого|ых|ую|ым|ыми)\s+\w+"),
        ("безупречн", r"безупречн\w*"),
        ("непревзойд", r"непревзойд\w*"),
        ("первоклассн", r"первоклассн\w*"),
        ("высококачествен", r"высококачествен\w*"),
        ("надежн", r"надежн\w*|надёжн\w*"),
        ("премиум", r"премиум\w*"),
        ("№1", r"№\s*1\b"),
        ("гост", r"\bгост\b"),
        ("iso", r"\biso\b"),
        ("din", r"\bdin\b"),
        ("тр тс", r"\bтр\s*тс\b"),
        ("atex", r"\batex\b|\bатекс\b"),
        ("нефт", r"нефт\w*"),
        ("газ", r"\bгаз(?:а|у|ом|е|ов|ы|ами|овых|овой|овом|овые)?\b|газопровод\w*"),
        ("химическ", r"химическ\w*"),
        ("пищев", r"пищев\w*"),
        ("энергетик", r"энергетик\w*"),
        ("теплоснабж", r"теплоснабж\w*"),
        ("водоснабж", r"водоснабж\w*"),
        ("фармацевт", r"фармацевт\w*"),
        ("судостро", r"судостро\w*"),
        ("машиностро", r"машиностро\w*"),
        ("металлург", r"металлург\w*"),
        ("горнодобыв", r"горнодобыв\w*"),
        ("котельн", r"котельн\w*"),
        ("отоплен", r"отоплен\w*"),
        ("жкх", r"\bжкх\b"),
        ("насос", r"насос\w*"),
        ("компрессор", r"компрессор\w*"),
        ("трубопровод", r"трубопровод\w*"),
    )
)


def number_issue(
    data: GenerationInput,
    text: str,
    *,
    free_tokens: set[str] | None = None,
    field_name: str = "field",
) -> str | None:
    """Первое число, которого нет в подтвержденных данных, либо None.

    Сверка идет по нормализованной записи: запятая и точка в десятичной дроби,
    разделители диапазона, знаки умножения в резьбе и пробел в IP54 эквивалентны.
    Единицы и чужие числа из-за этого не становятся допустимыми.
    """

    findings = _number_findings(data, text, free_tokens=free_tokens, field_name=field_name)
    return findings[0].value if findings else None


def grounding_findings(
    data: GenerationInput,
    text: str,
    *,
    field_name: str,
    free_tokens: set[str] | None = None,
) -> list[GroundingFinding]:
    """Все отклоненные значения текста с причиной: числа, перепутанные и конфликтные."""

    findings = _number_findings(data, text, free_tokens=free_tokens, field_name=field_name)
    findings.extend(_swapped_findings(data, text, field_name))
    findings.extend(_conflict_findings(data, text, field_name))
    return findings


def log_findings(findings: list[GroundingFinding], attempt: Attempt) -> None:
    """Подробная диагностика отклонений. Только DEBUG: в логе есть фрагменты текста."""

    if not logger.isEnabledFor(logging.DEBUG):
        return
    for finding in findings:
        logger.debug(
            "grounding rejected attempt=%s field=%s kind=%s value=%r value_ascii=%s "
            "attribute=%s reason=%s snippet=%r",
            attempt.value,
            finding.field,
            finding.kind.value,
            finding.value,
            ascii(finding.value),
            finding.attribute or "не определена",
            finding.reason,
            finding.snippet,
        )


def text_grounded(data: GenerationInput, text: str) -> bool:
    """Числа, привязка значений к характеристикам и конфликтные значения в порядке."""

    return not grounding_findings(data, text, field_name="field")


def unsupported_claim(data: GenerationInput, text: str) -> str | None:
    """Фраза с гарантией, сертификатом, превосходной степенью или применением.

    Допустима только если та же основа есть в названии, производителе или
    подтвержденных характеристиках.
    """

    known = _known_text(data)
    folded = text.casefold()
    for stem, pattern in _CLAIM_PATTERNS:
        if re.search(rf"(?<!\w){re.escape(stem)}", known):
            continue
        match = pattern.search(folded)
        if match:
            return match.group().strip()
    return None


def query_grounded(query: str, data: GenerationInput) -> bool:
    """Запрос назван по производителю, первому слову названия или коду модели."""

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
    return any(token.casefold() in folded for token in NUMBER.findall(title))


def query_numbers_ok(data: GenerationInput, text: str) -> bool:
    """Числа запроса подтверждены, числа из названия разрешены как код модели."""

    return not query_findings(data, text)


def query_findings(data: GenerationInput, text: str) -> list[GroundingFinding]:
    """Отклонения в поисковом запросе: числа из названия модели разрешены как код."""

    title = trusted_product_title(data.name)
    free_tokens = {fold_technical(token) for token in NUMBER.findall(title)}
    return grounding_findings(data, text, field_name="search_query", free_tokens=free_tokens)


def is_cut_title(field: str, catalog_title: str) -> bool:
    """Поле — начало названия, обрезанное лимитом заголовка, а не новое число."""

    compact = field.strip()
    return bool(compact) and compact != catalog_title and catalog_title.startswith(compact)


def mentions_of(value: str, text: str) -> int:
    """Сколько раз значение встречается в тексте как отдельное, не часть другого числа."""

    folded_value = fold_technical(value.strip())
    if not folded_value:
        return 0
    pattern = re.compile(rf"(?<![\w,.]){re.escape(folded_value)}(?!\w|[,.]\d)", re.IGNORECASE)
    return len(pattern.findall(fold_technical(text)))


def fold_technical(text: str) -> str:
    """Свести эквивалентную техническую запись к одному виду, не меняя цифры."""

    folded = text.translate(str.maketrans({"×": "x", "х": "x", "Х": "x", "−": "-"}))
    folded = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "x", folded)
    folded = re.sub(r"(?<=\d)\s*(?:\.{2,3}|…)\s*(?=\d)", "...", folded)
    folded = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", "...", folded)
    folded = re.sub(r"(?<=\d)\.(?=\d)", ",", folded)
    return re.sub(r"IP[\s-]*(\d{2})\b", r"IP\1", folded, flags=re.IGNORECASE)


def fact_blobs(data: GenerationInput) -> list[str]:
    """Подтвержденные значения характеристик целиком, с единицами."""

    return [item.value.strip() for item in data.attributes if item.value.strip()]


def identity_blobs(data: GenerationInput) -> list[str]:
    """Название и производитель как идентификаторы, не как склад технических чисел."""

    blobs: list[str] = []
    title = trusted_product_title(data.name)
    if title:
        blobs.append(title)
    manufacturer = data.manufacturer.strip()
    if manufacturer:
        blobs.append(manufacturer)
    return blobs


def _known_text(data: GenerationInput) -> str:
    parts = [data.name, data.manufacturer]
    for item in data.attributes:
        parts.extend([item.name, item.value, item.evidence])
    return " ".join(parts).casefold()


def _number_spans(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group()) for match in NUMBER.finditer(text)]


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


def _range_components(blob: str) -> tuple[set[str], str]:
    """Компоненты диапазонов значения и единица после последнего числа."""

    spans = _number_spans(blob)
    if not spans:
        return set(), ""
    components: set[str] = set()
    for _start, _end, token in spans:
        if _RANGE_SEPARATOR in token:
            components.update(token.split(_RANGE_SEPARATOR))
    unit = blob[spans[-1][1] :].strip()
    if len(unit) > _MAX_UNIT_LENGTH or any(char.isdigit() for char in unit):
        unit = ""
    return components, unit


def _is_range_component(text: str, end: int, token: str, facts: list[str]) -> bool:
    """«от 0 до 1,6 МПа»: компонент диапазона и та же единица или связка «до»."""

    word = _UNIT_AFTER_NUMBER.match(text, end)
    following = (word.group(1) if word else "").casefold()
    for blob in facts:
        components, unit = _range_components(blob)
        if token not in components:
            continue
        if not following or following in _RANGE_CONNECTORS or following == unit.casefold():
            return True
    return False


def _code_run(text: str, start: int, end: int) -> tuple[int, int]:
    left = start
    while left > 0 and _CODE_CHARS.match(text[left - 1]):
        left -= 1
    right = end
    while right < len(text):
        if _CODE_CHARS.match(text[right]):
            right += 1
        elif (
            text[right] in ",."
            and right + 1 < len(text)
            and text[right + 1].isdigit()
            and text[right - 1].isdigit()
        ):
            right += 1
        else:
            break
    return left, right


def _is_code_fragment(text: str, start: int, end: int, blobs: list[str]) -> bool:
    """Число внутри кода, который является началом или концом подтвержденного кода."""

    left, right = _code_run(text, start, end)
    run = text[left:right].strip("-")
    if not any(char.isalpha() for char in run) or run == text[start:end]:
        return False
    needle = run.casefold()
    for blob in blobs:
        haystack = blob.casefold()
        index = haystack.find(needle)
        while index >= 0:
            before = haystack[index - 1] if index else ""
            after = haystack[index + len(needle) :]
            breaks_number = bool(re.match(r"\d|[,.]\d", after))
            if not before.isalnum() and not breaks_number:
                return True
            index = haystack.find(needle, index + 1)
    return False


def _exclusive_numbers(data: GenerationInput) -> dict[str, set[str]]:
    per_name = {
        item.name: set(NUMBER.findall(item.value))
        for item in data.attributes
        if item.name.strip() and item.value.strip()
    }
    counts = Counter(token for tokens in per_name.values() for token in tokens)
    return {
        name: {token for token in tokens if counts[token] == 1}
        for name, tokens in per_name.items()
    }


def _swapped_hits(data: GenerationInput, text: str) -> Iterator[tuple[str, str, str, str]]:
    """(имя в тексте, число, владелец числа, окно текста) для числа не у своей характеристики."""

    exclusive = _exclusive_numbers(data)
    names = [item.name for item in data.attributes if item.name.strip()]
    for name, window in _attribute_windows(text, names):
        for token in NUMBER.findall(window):
            owner = next(
                (item for item, tokens in exclusive.items() if token in tokens),
                None,
            )
            if owner is not None and owner.casefold() != name.casefold():
                yield name, token, owner, window


def _swapped_findings(data: GenerationInput, text: str, field_name: str) -> list[GroundingFinding]:
    values = {item.name: item.value for item in data.attributes}
    findings: list[GroundingFinding] = []
    for name, token, owner, window in _swapped_hits(data, text):
        context = f"{name}{window}"
        findings.append(
            GroundingFinding(
                kind=FindingKind.SWAPPED_VALUE,
                field=field_name,
                value=token,
                snippet=_around(context, context.find(token), len(token)),
                attribute=f"{name} = {values.get(name, '?')}",
                reason=(
                    f"число подтверждено только для «{owner}» "
                    f"(значение «{values.get(owner, '?')}»), "
                    f"но в тексте стоит после названия «{name}»"
                ),
            )
        )
    return findings


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


def _conflict_findings(data: GenerationInput, text: str, field_name: str) -> list[GroundingFinding]:
    findings: list[GroundingFinding] = []
    for conflict in data.conflicts:
        foreign = conflict.description_value.strip()
        confirmed = conflict.specs_value.strip()
        if not foreign or foreign.casefold() == confirmed.casefold():
            continue
        match = re.search(rf"(?<!\w){re.escape(foreign)}(?!\w)", text, re.IGNORECASE)
        if match:
            findings.append(
                GroundingFinding(
                    kind=FindingKind.CONFLICT_VALUE,
                    field=field_name,
                    value=foreign,
                    snippet=_around(text, match.start(), len(match.group())),
                    attribute=f"{conflict.name} = {confirmed}",
                    reason=(
                        "значение из технического описания противоречит подтвержденному "
                        "значению характеристики и не должно попадать в текст"
                    ),
                )
            )
    return findings


def _raw_snippet(text: str, token: str, occurrence: int) -> str | None:
    """Фрагмент исходного текста вокруг n-го дословного вхождения числа."""

    pattern = re.compile(rf"(?<![\d,.]){re.escape(token)}(?!\d|[,.]\d)")
    matches = list(pattern.finditer(text))
    if occurrence >= len(matches):
        return None
    match = matches[occurrence]
    return _around(text, match.start(), len(token))


def _around(text: str, start: int, length: int) -> str:
    start = max(start, 0)
    return text[max(0, start - _SNIPPET_RADIUS) : start + length + _SNIPPET_RADIUS]


def _number_findings(
    data: GenerationInput,
    text: str,
    *,
    free_tokens: set[str] | None,
    field_name: str,
) -> list[GroundingFinding]:
    """Числа текста, которых нет в подтвержденных данных, с причиной и характеристикой.

    Фрагмент берется из исходного текста, если число записано в нем дословно.
    Иначе это нормализованный текст («х»→«x», единый вид диапазонов и IP).
    """

    folded = fold_technical(text)
    facts = [fold_technical(blob) for blob in fact_blobs(data)]
    identity = [fold_technical(blob) for blob in identity_blobs(data)]
    allowed_free = free_tokens or set()
    findings: list[GroundingFinding] = []
    seen: Counter[str] = Counter()
    for start, end, token in _number_spans(folded):
        occurrence = seen[token]
        seen[token] += 1
        if _span_covered(folded, start, end, facts):
            continue
        if _span_covered(folded, start, end, identity):
            continue
        if token in allowed_free:
            continue
        if _is_range_component(folded, end, token, facts):
            continue
        if _is_code_fragment(folded, start, end, [*facts, *identity]):
            continue
        findings.append(
            GroundingFinding(
                kind=FindingKind.NUMBER,
                field=field_name,
                value=token,
                snippet=_raw_snippet(text, token, occurrence)
                or _around(folded, start, end - start),
                attribute=_related_attribute(data, folded, start, token),
                reason=_number_reason(folded, start, end, token, facts),
            )
        )
    return findings


def _related_attribute(data: GenerationInput, folded: str, start: int, token: str) -> str | None:
    """Характеристика, к которой текст относит число, и та, чье значение содержит число."""

    nearest: tuple[int, str] | None = None
    for item in data.attributes:
        name = fold_technical(item.name.strip())
        if not name:
            continue
        for match in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", folded, re.IGNORECASE):
            if match.end() <= start and (nearest is None or match.end() > nearest[0]):
                nearest = (match.end(), f"{item.name} = {item.value}")
    containing = next(
        (f"{item.name} = {item.value}" for item in data.attributes if token in item.value),
        None,
    )
    parts: list[str] = []
    if nearest is not None:
        parts.append(f"ближайшая перед числом: {nearest[1]}")
    if containing is not None and (nearest is None or containing != nearest[1]):
        parts.append(f"содержит это число: {containing}")
    return "; ".join(parts) or None


def _number_reason(folded: str, start: int, end: int, token: str, facts: list[str]) -> str:
    reasons: list[str] = []
    word = _UNIT_AFTER_NUMBER.match(folded, end)
    following = word.group(1) if word else ""
    for blob in facts:
        components, unit = _range_components(blob)
        if token in components:
            reasons.append(
                f"число — компонент диапазона «{blob}», но после него стоит «{following}», "
                f"а нужна единица «{unit}» или связка «до»/«и»"
            )
            break
    else:
        holder = next((blob for blob in facts if token in blob), None)
        if holder is not None:
            reasons.append(
                f"число входит в значение «{holder}», но запись в тексте не совпадает с ним "
                "целиком (другая единица, склейка или перестановка частей)"
            )
    left, right = _code_run(folded, start, end)
    run = folded[left:right].strip("-")
    if any(char.isalpha() for char in run) and run != token:
        reasons.append(
            f"число стоит внутри кода «{run}», который не является началом или концом "
            "подтвержденного кода"
        )
    return "; ".join(reasons) or (
        "числа нет ни в одном подтвержденном значении, названии и производителе"
    )
