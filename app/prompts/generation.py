"""Промпты для генерации SEO-заголовка, описания и commercial-блока."""

from app.schemas.product import GenerationInput, trusted_product_title


def build_generation_prompt(data: GenerationInput) -> str:
    """Собрать промпт только из проверенных полей, без сырого описания товара."""

    lines = [
        "Собери SEO-карточку строго по данным ниже.",
        "Название — только идентификатор товара, не источник технических фактов.",
        "Копируй числа, диапазоны и единицы целиком из подтвержденных характеристик.",
        "Не собирай новое число из частей разных значений и не меняй единицы измерения.",
        "Альтернативное значение конфликта не публикуй.",
        "Фактом остается только подтвержденное значение характеристики.",
        "seo_title не пустой и не длиннее 70 символов.",
        "seo_description содержит от 800 до 1200 символов.",
        "search_queries: от 3 до 5 разных запросов.",
        "Каждый запрос содержит производителя, первое слово публикуемого названия или его код.",
        "commercial_block не пустой и опирается только на подтвержденные данные.",
        "",
        f"Название: {data.name}",
        f"Публикуемое название: {trusted_product_title(data.name)}",
    ]
    if data.manufacturer.strip():
        lines.append(f"Производитель: {data.manufacturer}")

    lines.append("Подтвержденные характеристики:")
    if data.attributes:
        for attribute in data.attributes:
            lines.append(
                f"- {attribute.name} = {attribute.value}; цитата: {attribute.evidence}"
            )
    else:
        lines.append("- нет")

    lines.append("Отсутствующие поля:")
    if data.missing_fields:
        lines.extend(f"- {field}" for field in data.missing_fields)
    else:
        lines.append("- нет")

    lines.append("Конфликты:")
    if data.conflicts:
        for conflict in data.conflicts:
            lines.append(
                f"- {conflict.name}: подтверждено {conflict.specs_value}; "
                f"в другом источнике {conflict.description_value}"
            )
    else:
        lines.append("- нет")
    return "\n".join(lines)
