"""Pydantic-схемы входных и выходных данных карточки товара."""

from enum import StrEnum
from typing import Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AttributeSource(StrEnum):
    """Откуда взято подтвержденное значение."""

    SPECS = "specs"
    DESCRIPTION = "description"


class ProductInput(BaseModel):
    """Входные данные товара для извлечения характеристик."""

    name: str = Field(
        default="",
        description="Название прибора так, как его нужно показать покупателю.",
    )
    manufacturer: str = Field(
        default="",
        description="Производитель. Пустая строка допустима.",
    )
    specs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Характеристики: имя параметра и значение. "
            "Пустое значение не считается подтвержденным. "
            "Ключи этого словаря задают, какие характеристики искать в описании."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "Техническое описание. Из него принимаются только характеристики, "
            "чьи имена есть среди ключей specs, и только с дословным подтверждением."
        ),
    )


class ExtractedAttribute(BaseModel):
    """Характеристика с источником и подтверждающим фрагментом."""

    name: str = Field(description="Имя характеристики.")
    value: str = Field(description="Подтвержденное значение без перефразирования.")
    source: AttributeSource = Field(description="Источник значения: specs или description.")
    evidence: str = Field(description="Фрагмент, которым подтверждено значение.")


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
# Структурный шлюз, не оценка полноты. Три значения не доказывают,
# что честные 800 символов можно написать без домыслов.
# Альтернатива из конфликта в этот счет не входит как отдельный факт:
# учитывается только непустое value атрибута, уже выбранное Extractor.
COMPLETE_MIN_ATTRIBUTES = 3


class GenerationStatus(StrEnum):
    """Полная карточка, нехватка входных данных или частичный результат без текста модели."""

    COMPLETE = "complete"
    INSUFFICIENT_DATA = "insufficient_data"
    NEEDS_REVIEW = "needs_review"


class GenerationWarningCode(StrEnum):
    """Почему карточку нельзя считать полной или бесспорной."""

    CONFLICT = "conflict"
    MISSING_FIELD = "missing_field"
    INSUFFICIENT_DATA = "insufficient_data"
    NEEDS_REVIEW = "needs_review"


class GenerationWarning(BaseModel):
    """Структурное предупреждение. Текст message не является эталоном для тестов."""

    model_config = ConfigDict(extra="forbid")

    code: GenerationWarningCode = Field(description="Код предупреждения.")
    field: str = Field(
        default="",
        description="Имя характеристики, если предупреждение относится к полю.",
    )
    message: str = Field(default="", description="Пояснение. Может быть пустым.")


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


def trusted_product_title(name: str) -> str:
    """Публикуемый идентификатор товара.

    Название не очищается эвристиками вроде разбиения по ``. ``: сокращения
    ``исп. 2`` и коды с точками законны. Числа из названия не подтверждают
    технические характеристики — это делает только Generator по specs.
    """

    return name.strip()


def assess_generation_status(data: GenerationInput) -> GenerationStatus:
    """Решить, есть ли минимум данных, чтобы вызывать модель.

    Порог в три непустых значения не измеряет достаточность информации и не
    обещает честное описание на 800 символов. Другую числовую «полноту»
    функция не вводит. Значение из чужого источника конфликта подтвержденным
    фактом не является: в шлюз попадает только value атрибута.
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
    if status is GenerationStatus.INSUFFICIENT_DATA or status is GenerationStatus.NEEDS_REVIEW:
        return
    assert_never(status)


class GenerationResult(BaseModel):
    """Черновик SEO-карточки. Характеристики копируются из GenerationInput."""

    model_config = ConfigDict(extra="forbid")

    status: GenerationStatus = Field(
        description=(
            "complete — полная карточка. "
            "insufficient_data — данных мало, текст частичный и не является готовым описанием. "
            "needs_review — данные есть, но текст модели нельзя опубликовать целиком."
        )
    )
    seo_title: str = Field(description="SEO-заголовок не длиннее 70 символов.")
    seo_description: str = Field(
        description=(
            "SEO-описание. У полной карточки от 800 до 1200 символов. "
            "При нехватке данных текст короче 800 символов."
        )
    )
    specifications: list[ExtractedAttribute] = Field(
        default_factory=list,
        description="Подтвержденные характеристики. Ответ модели их не подменяет.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Поисковые запросы. При нехватке данных список может быть пустым.",
    )
    commercial_block: str = Field(description="Коммерческий блок по подтвержденным данным.")
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Имена характеристик, которые не удалось подтвердить.",
    )
    warnings: list[GenerationWarning] = Field(
        default_factory=list,
        description="Предупреждения о конфликтах, пропусках и нехватке данных.",
    )

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
        elif self.status is GenerationStatus.NEEDS_REVIEW:
            if description_length > SEO_DESCRIPTION_MAX_LENGTH:
                raise ValueError(
                    "seo_description length must be at most "
                    f"{SEO_DESCRIPTION_MAX_LENGTH} characters"
                )
        else:
            assert_never(self.status)

        _validate_search_queries(self.status, self.search_queries)
        if not self.commercial_block.strip():
            raise ValueError("commercial_block is empty")
        return self
