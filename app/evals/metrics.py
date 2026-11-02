"""Метрики качества извлечения и минимальная нормализация значений."""

import re

_NOTATION = str.maketrans(
    {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u202f": " ",
    }
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalize_name(name: str) -> str:
    """Регистр и пробелы в имени характеристики не различают поле."""

    return " ".join(name.strip().casefold().split())


def normalize_value(value: str) -> str:
    """Убрать лишние пробелы, сохранив числа, разделители, единицы и регистр."""

    return " ".join(value.strip().split())


def notation_key(value: str) -> str:
    """Ключ для ручной проверки близких записей. Не используется как равенство."""

    folded = normalize_value(value).casefold().translate(_NOTATION)
    return "".join(folded.replace(",", ".").split())


def precision(true_positive: int, predicted: int) -> float:
    """Доля верных пар среди извлеченных. Нет извлеченных пар — нет и ложных."""

    if predicted == 0:
        return 1.0
    return true_positive / predicted


def recall(true_positive: int, expected: int) -> float:
    """Доля найденных пар среди ожидаемых. Пустой эталон — пропусков нет."""

    if expected == 0:
        return 1.0
    return true_positive / expected


def f1_score(precision_score: float, recall_score: float) -> float:
    """Гармоническое среднее. Ноль в обеих долях дает ноль."""

    total = precision_score + recall_score
    if total == 0:
        return 0.0
    return 2 * precision_score * recall_score / total


def hallucination_rate(unsupported: int, extracted: int) -> float:
    """Доля неподтвержденных характеристик. Пустой ответ — ноль галлюцинаций."""

    if extracted == 0:
        return 0.0
    return unsupported / extracted


def pair_key(name: str, value: str) -> tuple[str, str]:
    return (normalize_name(name), normalize_value(value))


def near_match_reason(left: str, right: str) -> str | None:
    """Почему значения похожи, если автоматическое равенство их не принимает."""

    if normalize_value(left) == normalize_value(right):
        return None
    if notation_key(left) == notation_key(right):
        return "notation"
    if _same_numbers(left, right):
        return "same_numbers"
    if _token_extension(left, right):
        return "token_extension"
    return None


def has_number(value: str) -> bool:
    return bool(re.search(r"\d", value))


def _same_numbers(left: str, right: str) -> bool:
    left_numbers = _numeric_tokens(left)
    right_numbers = _numeric_tokens(right)
    return bool(left_numbers) and left_numbers == right_numbers


def _numeric_tokens(value: str) -> tuple[str, ...]:
    return tuple(_NUMBER.findall(notation_key(value)))


def _token_extension(left: str, right: str) -> bool:
    left_tokens = normalize_value(left).casefold().split()
    right_tokens = normalize_value(right).casefold().split()
    if not left_tokens or not right_tokens or left_tokens == right_tokens:
        return False
    if len(left_tokens) == len(right_tokens):
        return False
    shorter, longer = (
        (left_tokens, right_tokens)
        if len(left_tokens) < len(right_tokens)
        else (right_tokens, left_tokens)
    )
    extras = _unmatched_tokens(shorter, longer)
    if extras is None:
        return False
    return all(not re.search(r"\d", token) for token in extras)


def _unmatched_tokens(shorter: list[str], longer: list[str]) -> list[str] | None:
    extras: list[str] = []
    index = 0
    for token in longer:
        if index < len(shorter) and token == shorter[index]:
            index += 1
            continue
        extras.append(token)
    if index != len(shorter):
        return None
    return extras
