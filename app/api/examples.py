"""Синтетические примеры для OpenAPI. Это не параметры реальных приборов."""

from app.schemas.product import (
    AttributeSource,
    ExtractedAttribute,
    GenerationResult,
    GenerationStatus,
    GenerationWarning,
    GenerationWarningCode,
)

MANOMETER_REQUEST: dict[str, object] = {
    "name": "Манометр демонстрационный МП-Демо-01",
    "manufacturer": "Учебный стенд «Сигнал»",
    "description": (
        "Учебный пример, не паспорт реального прибора. "
        "Диапазон показаний 0...1,6 МПа. Класс точности 1,5. "
        "Присоединение M20x1,5. Степень защиты IP54."
    ),
    "specs": {
        "Диапазон показаний": "0...1,6 МПа",
        "Класс точности": "1,5",
        "Присоединение": "M20x1,5",
        "Степень защиты": "IP54",
    },
}

TEMPERATURE_SENSOR_REQUEST: dict[str, object] = {
    "name": "Датчик температуры демонстрационный ДТ-Демо-03",
    "manufacturer": "Учебный стенд «Сигнал»",
    "description": (
        "Учебный пример, не паспорт реального прибора. "
        "Диапазон измерений -50...+150 °C. Тип чувствительного элемента Pt100. "
        "Класс допуска B. Выходной сигнал 4...20 мА. Степень защиты IP65."
    ),
    "specs": {
        "Диапазон измерений": "-50...+150 °C",
        "Тип чувствительного элемента": "Pt100",
        "Класс допуска": "B",
        "Выходной сигнал": "4...20 мА",
        "Степень защиты": "IP65",
    },
}

THERMOMETER_REQUEST: dict[str, object] = {
    "name": "Термометр демонстрационный ТС-Демо-02",
    "manufacturer": "Учебный стенд «Сигнал»",
    "description": (
        "Учебный пример с одной подтвержденной характеристикой. "
        "Диапазон измерений 0...100 °C."
    ),
    "specs": {"Диапазон измерений": "0...100 °C"},
}

REQUEST_EXAMPLES: dict[str, dict[str, object]] = {
    "manometer": {
        "summary": "Сценарий A: демонстрационный манометр, 4 характеристики",
        "description": (
            "Синтетические параметры учебного примера, не характеристики реального прибора. "
            "Ожидается полная карточка."
        ),
        "value": MANOMETER_REQUEST,
    },
    "temperature_sensor": {
        "summary": "Сценарий B: демонстрационный датчик температуры, 5 характеристик",
        "description": (
            "Синтетические параметры учебного примера, не характеристики реального прибора. "
            "Ожидается полная карточка."
        ),
        "value": TEMPERATURE_SENSOR_REQUEST,
    },
    "thermometer": {
        "summary": "Демонстрационный термометр, данных мало",
        "description": (
            "Синтетический пример ТС-Демо-02. "
            "Одной характеристики недостаточно для полной карточки."
        ),
        "value": THERMOMETER_REQUEST,
    },
}

_COMPLETE_DESCRIPTION = (
    "Манометр демонстрационный МП-Демо-01 производства Учебный стенд «Сигнал» — "
    "прибор для измерения давления, который удобно сравнивать с другими моделями "
    "по основным параметрам. "
    "Диапазон показаний 0...1,6 МПа определяет границы шкалы, поэтому при выборе "
    "важно сопоставить его с рабочим давлением вашей системы. "
    "Класс точности 1,5 показывает допустимую погрешность показаний: чем меньше "
    "это число, тем точнее прибор, и это стоит учитывать там, где важна "
    "повторяемость измерений. "
    "Присоединение M20x1,5 описывает резьбу штуцера, поэтому перед заказом "
    "убедитесь, что ответная часть на вашем оборудовании имеет такой же размер "
    "и шаг резьбы. "
    "Степень защиты IP54 говорит о защите корпуса от пыли и брызг, что полезно "
    "учитывать при выборе места установки. "
    "Специалисту по закупке и подбору оборудования рекомендуется сверить все "
    "перечисленные параметры с проектной документацией и условиями эксплуатации. "
    "Перед оформлением заказа проверьте совместимость прибора с вашей системой "
    "и при необходимости уточните детали у поставщика."
)

MANOMETER_SPECIFICATIONS = [
    ExtractedAttribute(
        name=name,
        value=value,
        source=AttributeSource.SPECS,
        evidence=value,
    )
    for name, value in (MANOMETER_REQUEST["specs"]).items()  # type: ignore[attr-defined]
]

COMPLETE_EXAMPLE: dict[str, object] = GenerationResult(
    status=GenerationStatus.COMPLETE,
    seo_title="Манометр демонстрационный МП-Демо-01",
    seo_description=_COMPLETE_DESCRIPTION,
    specifications=MANOMETER_SPECIFICATIONS,
    search_queries=[
        "Манометр демонстрационный МП-Демо-01 купить",
        "Манометр демонстрационный МП-Демо-01 характеристики",
        "Учебный стенд «Сигнал» манометр",
    ],
    commercial_block=(
        "Манометр демонстрационный МП-Демо-01: Диапазон показаний 0...1,6 МПа. "
        "Производитель: Учебный стенд «Сигнал»."
    ),
).model_dump(mode="json")

# Частичный результат: описание не прошло проверку и не опубликовано.
# Причина лежит в warnings, а не в публикуемом тексте.
NEEDS_REVIEW_EXAMPLE: dict[str, object] = GenerationResult(
    status=GenerationStatus.NEEDS_REVIEW,
    seo_title="Манометр демонстрационный МП-Демо-01",
    seo_description=None,
    specifications=MANOMETER_SPECIFICATIONS,
    search_queries=[
        "Манометр демонстрационный МП-Демо-01 купить",
        "Манометр демонстрационный МП-Демо-01 характеристики",
        "Учебный стенд «Сигнал» манометр",
    ],
    commercial_block=(
        "Манометр демонстрационный МП-Демо-01: Диапазон показаний 0...1,6 МПа. "
        "Производитель: Учебный стенд «Сигнал»."
    ),
    warnings=[
        GenerationWarning(
            code=GenerationWarningCode.DESCRIPTION_TOO_SHORT,
            field="seo_description",
            message=(
                "Описание не принято: длина 640 символов, нужно 800–1200. "
                "Автоматическое исправление не помогло."
            ),
        ),
        GenerationWarning(
            code=GenerationWarningCode.NEEDS_REVIEW,
            message=(
                "Часть публикуемых полей не прошла проверку. "
                "Подтвержденные характеристики сохранены."
            ),
        ),
    ],
).model_dump(mode="json")

INSUFFICIENT_EXAMPLE: dict[str, object] = GenerationResult(
    status=GenerationStatus.INSUFFICIENT_DATA,
    seo_title="Термометр демонстрационный ТС-Демо-02",
    seo_description=None,
    specifications=[
        ExtractedAttribute(
            name="Диапазон измерений",
            value="0...100 °C",
            source=AttributeSource.SPECS,
            evidence="0...100 °C",
        )
    ],
    search_queries=[],
    commercial_block=None,
    missing_fields=["Класс допуска"],
    warnings=[
        GenerationWarning(code=GenerationWarningCode.MISSING_FIELD, field="Класс допуска"),
        GenerationWarning(
            code=GenerationWarningCode.INSUFFICIENT_DATA,
            message=(
                "Подтверждено характеристик: 1, для полной карточки нужно не меньше 3 "
                "и название товара. Модель не вызывалась."
            ),
        ),
    ],
).model_dump(mode="json")
