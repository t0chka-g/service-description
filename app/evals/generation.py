"""Контракт будущей оценки SEO-текста.

Реального прогона и судьи здесь нет: модуль только описывает датасет и метки,
которые unit-тесты не могут подтвердить по свободному тексту.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.product import GenerationInput, GenerationStatus, GenerationWarning

GENERATION_EVAL_METHODOLOGY = (
    "Unit-тесты проверяют схему, длины, типы полей, перенос характеристик, "
    "порядок предупреждений и ошибки разбора. "
    "Они отклоняют как ungrounded чужие числа, forbidden_claims с цифрами, конфликтные значения "
    "и запросы без названия или производителя. "
    "Филлер и текст короче 800 символов не получают статус complete: "
    "такие ответы становятся insufficient_data. "
    "Они не доказывают отсутствие смысловых галлюцинаций в свободном тексте. "
    "Судья нужен для перефразированных качественных утверждений без новых чисел, "
    "а также для назначения и целевой аудитории commercial-блока. "
    "Будущий судья сравнивает seo_description и commercial_block только с названием, "
    "производителем и подтвержденными характеристиками. "
    "Числа, диапазоны и единицы в тексте должны совпадать с подтвержденными значениями. "
    "Значение из конфликта нельзя подавать как факт. "
    "Заголовок не длиннее 70 символов, полное описание — 800–1200, запросов от 3 до 5. "
    "Три подтвержденные характеристики — только шлюз и не гарантируют честные 800 символов. "
    "При недостатке данных статус insufficient_data, и длину нельзя добирать вымыслом. "
    "Этот модуль не вызывает модель и не выставляет оценку."
)


class GenerationConstraints(BaseModel):
    """Проверяемые границы результата.

    Для insufficient_data поле seo_description_min_length — порог, который нельзя
    достигать дописыванием, а не требуемая длина текста.
    """

    model_config = ConfigDict(extra="forbid")

    status: GenerationStatus
    seo_title_max_length: int
    seo_description_min_length: int
    seo_description_max_length: int
    search_queries_min: int
    search_queries_max: int


class ForbiddenSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: str


class GenerationEvalCase(BaseModel):
    """Сценарий генерации без эталонного SEO-текста."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    input: GenerationInput
    expected_constraints: GenerationConstraints
    expected_missing_fields: list[str]
    expected_warnings: list[GenerationWarning]
    forbidden_claims: list[str] = Field(default_factory=list)
    forbidden_specifications: list[ForbiddenSpecification] = Field(default_factory=list)


class GenerationJudgeScore(BaseModel):
    """Оценка будущего LLM-судьи. Поле faithfulness заполняется только живым прогоном."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    faithfulness: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    conflict_stated_as_fact: bool | None = None
    purpose_covered: bool | None = None
    audience_covered: bool | None = None
    notes: str = ""


def dataset_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "generation_cases.json"


def load_generation_cases(path: Path | None = None) -> list[GenerationEvalCase]:
    """Загрузить сценарии. Судья по ним не запускается."""

    payload = json.loads((path or dataset_path()).read_text(encoding="utf-8"))
    return [GenerationEvalCase.model_validate(item) for item in payload]
