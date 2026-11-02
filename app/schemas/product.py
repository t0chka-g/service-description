"""Pydantic-схемы входных и выходных данных карточки товара."""

from enum import StrEnum

from pydantic import BaseModel, Field


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
