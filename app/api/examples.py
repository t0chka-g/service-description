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
        "summary": "Демонстрационный манометр",
        "description": (
            "Синтетические параметры учебного примера, не характеристики реального прибора."
        ),
        "value": MANOMETER_REQUEST,
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


def _complete_description() -> str:
    core = (
        "Манометр демонстрационный МП-Демо-01 — учебный пример карточки. "
        "Производитель: Учебный стенд «Сигнал». "
        "Диапазон показаний: 0...1,6 МПа. Класс точности: 1,5. "
        "Присоединение: M20x1,5. Степень защиты: IP54. "
        "Текст показывает форму успешного ответа и не описывает серийный прибор."
    )
    note = " Параметры примера не следует переносить в каталог реального оборудования."
    text = core
    while len(text) < 860:
        text += note
    return text


COMPLETE_EXAMPLE: dict[str, object] = GenerationResult(
    status=GenerationStatus.COMPLETE,
    seo_title="Манометр демонстрационный МП-Демо-01",
    seo_description=_complete_description(),
    specifications=[
        ExtractedAttribute(
            name="Диапазон показаний",
            value="0...1,6 МПа",
            source=AttributeSource.SPECS,
            evidence="0...1,6 МПа",
        ),
        ExtractedAttribute(
            name="Класс точности",
            value="1,5",
            source=AttributeSource.SPECS,
            evidence="1,5",
        ),
        ExtractedAttribute(
            name="Присоединение",
            value="M20x1,5",
            source=AttributeSource.SPECS,
            evidence="M20x1,5",
        ),
        ExtractedAttribute(
            name="Степень защиты",
            value="IP54",
            source=AttributeSource.SPECS,
            evidence="IP54",
        ),
    ],
    search_queries=[
        "Манометр демонстрационный МП-Демо-01 купить",
        "Манометр демонстрационный МП-Демо-01 характеристики",
        "Манометр демонстрационный МП-Демо-01 цена",
    ],
    commercial_block=(
        "Манометр демонстрационный МП-Демо-01: Диапазон показаний 0...1,6 МПа. "
        "Производитель: Учебный стенд «Сигнал»."
    ),
).model_dump(mode="json")

INSUFFICIENT_EXAMPLE: dict[str, object] = GenerationResult(
    status=GenerationStatus.INSUFFICIENT_DATA,
    seo_title="Термометр демонстрационный ТС-Демо-02",
    seo_description=(
        "Термометр демонстрационный ТС-Демо-02. "
        "Производитель: Учебный стенд «Сигнал». "
        "Полное описание не собрано: подтвержденных данных недостаточно."
    ),
    specifications=[
        ExtractedAttribute(
            name="Диапазон измерений",
            value="0...100 °C",
            source=AttributeSource.SPECS,
            evidence="0...100 °C",
        )
    ],
    search_queries=[],
    commercial_block=(
        "Термометр демонстрационный ТС-Демо-02: Диапазон измерений 0...100 °C."
    ),
    missing_fields=["Класс допуска"],
    warnings=[
        GenerationWarning(code=GenerationWarningCode.MISSING_FIELD, field="Класс допуска"),
        GenerationWarning(code=GenerationWarningCode.INSUFFICIENT_DATA),
    ],
).model_dump(mode="json")
