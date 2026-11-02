"""Проверка характеристик, извлеченных из описания."""


def supported_quote(evidence: str, value: str, description: str) -> str | None:
    """Вернуть обрезанную цитату, только если она содержит value и лежит внутри описания.

    Совпадение фрагмента не доказывает, что характеристика извлечена верно:
    модель может процитировать соседнюю фразу. Поэтому value обязан входить в цитату.
    """
    quote = evidence.strip()
    source = description.strip()
    token = value.strip()
    if not quote or not source or not token:
        return None
    if quote == source or quote not in description or token not in quote:
        return None
    return quote


def same_technical_value(left: str, right: str) -> bool:
    """Сравнить записи без конвертации единиц и без смены десятичного разделителя."""
    return left.strip() == right.strip()
