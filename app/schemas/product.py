"""Pydantic-схемы входных и выходных данных карточки товара."""

from enum import StrEnum
from typing import Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AttributeSource(StrEnum):
    SPECS = "specs"
    DESCRIPTION = "description"


class ProductInput(BaseModel):
    """Входные данные товара для извлечения характеристик."""

    name: str = ""
    manufacturer: str = ""
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""


class ExtractedAttribute(BaseModel):
    """Характеристика с источником и подтверждающим фрагментом."""

    name: str
    value: str
    source: AttributeSource
    evidence: str  # подтверждающий фрагмент


class AttributeConflict(BaseModel):
    """Противоречие между specs и description по одной характеристике."""

    name: str
    specs_value: str
    description_value: str
    evidence: str


class ExtractionResult(BaseModel):
    """Результат извлечения характеристик товара."""

    attributes: list[ExtractedAttribute] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    conflicts: list[AttributeConflict] = Field(default_factory=list)


class DescriptionAttribute(BaseModel):
    """Характеристика, которую модель нашла в тексте описания."""

    name: str
    value: str
    evidence: str


class DescriptionExtraction(BaseModel):
    """Структурированный ответ LLM по текстовому описанию."""

    attributes: list[DescriptionAttribute] = Field(default_factory=list)


SEO_TITLE_MAX_LENGTH = 70
SEO_DESCRIPTION_MIN_LENGTH = 800
SEO_DESCRIPTION_MAX_LENGTH = 1200
SEARCH_QUERIES_MIN = 3
SEARCH_QUERIES_MAX = 5
# Шлюз полной карточки: непустое название и минимум три непустых значения.
# Порог не гарантирует, что описание на 800 символов можно написать без домыслов.
# Чужие числа и forbidden_claims отсекаются отдельно.
COMPLETE_MIN_ATTRIBUTES = 3


class GenerationStatus(StrEnum):
    """Полная карточка либо явный отказ добирать текст неподтвержденными фактами."""

    COMPLETE = "complete"
    INSUFFICIENT_DATA = "insufficient_data"


class GenerationWarningCode(StrEnum):
    CONFLICT = "conflict"
    MISSING_FIELD = "missing_field"
    INSUFFICIENT_DATA = "insufficient_data"


class GenerationWarning(BaseModel):
    """Структурное предупреждение. Текст message не является эталоном для тестов."""

    model_config = ConfigDict(extra="forbid")

    code: GenerationWarningCode
    field: str = ""
    message: str = ""


class GenerationInput(BaseModel):
    """Проверенные данные для SEO-карточки.

    Исходное техническое описание сюда не входит: генератор не извлекает
    характеристики повторно. Источник и цитата лежат в каждом атрибуте.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    manufacturer: str = ""
    attributes: list[ExtractedAttribute] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    conflicts: list[AttributeConflict] = Field(default_factory=list)


def assess_generation_status(data: GenerationInput) -> GenerationStatus:
    """Решить, открыт ли шлюз длинной карточки.

    Три подтвержденные характеристики не гарантируют честные 800 символов.
    Чужие числа и forbidden_claims отсекаются отдельно.
    """

    confirmed = [item for item in data.attributes if item.value.strip()]
    if data.name.strip() and len(confirmed) >= COMPLETE_MIN_ATTRIBUTES:
        return GenerationStatus.COMPLETE
    return GenerationStatus.INSUFFICIENT_DATA


def _validate_search_queries(status: GenerationStatus, queries: list[str]) -> None:
    normalized = [query.strip().casefold() for query in queries]
    if any(not query for query in normalized):
        raise ValueError("search query is empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate search query")
    if len(queries) > SEARCH_QUERIES_MAX:
        raise ValueError(f"search_queries exceed {SEARCH_QUERIES_MAX} items")
    if status is GenerationStatus.COMPLETE:
        if len(queries) < SEARCH_QUERIES_MIN:
            raise ValueError(
                f"search_queries must contain {SEARCH_QUERIES_MIN}..{SEARCH_QUERIES_MAX} items"
            )
        return
    if status is GenerationStatus.INSUFFICIENT_DATA:
        return
    assert_never(status)


class GenerationResult(BaseModel):
    """Черновик SEO-карточки. Характеристики копируются из GenerationInput."""

    model_config = ConfigDict(extra="forbid")

    status: GenerationStatus
    seo_title: str
    seo_description: str
    specifications: list[ExtractedAttribute] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    commercial_block: str
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[GenerationWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_contract(self) -> Self:
        if not self.seo_title.strip():
            raise ValueError("seo_title is empty")
        if len(self.seo_title) > SEO_TITLE_MAX_LENGTH:
            raise ValueError(f"seo_title exceeds {SEO_TITLE_MAX_LENGTH} characters")

        if not self.seo_description.strip():
            raise ValueError("seo_description is empty")
        description_length = len(self.seo_description)
        if self.status is GenerationStatus.COMPLETE:
            fits_complete_length = (
                SEO_DESCRIPTION_MIN_LENGTH <= description_length <= SEO_DESCRIPTION_MAX_LENGTH
            )
            if not fits_complete_length:
                raise ValueError(
                    "seo_description length must be "
                    f"{SEO_DESCRIPTION_MIN_LENGTH}..{SEO_DESCRIPTION_MAX_LENGTH}"
                )
        elif self.status is GenerationStatus.INSUFFICIENT_DATA:
            if description_length >= SEO_DESCRIPTION_MIN_LENGTH:
                raise ValueError(
                    "seo_description must stay below "
                    f"{SEO_DESCRIPTION_MIN_LENGTH} characters when data is insufficient"
                )
        else:
            assert_never(self.status)

        _validate_search_queries(self.status, self.search_queries)
        if not self.commercial_block.strip():
            raise ValueError("commercial_block is empty")
        return self
