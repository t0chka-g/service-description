"""Извлечение характеристик товара из текстового описания через LLM."""

import json

from pydantic import ValidationError

from app.clients.llm import LLMClient, LLMError
from app.prompts.extraction import build_extraction_prompt
from app.schemas.product import (
    AttributeConflict,
    AttributeSource,
    DescriptionExtraction,
    ExtractedAttribute,
    ExtractionResult,
    ProductInput,
)
from app.services.validators import same_technical_value, supported_quote


class CharacteristicExtractor:
    """Собирает характеристики из specs и подтвержденных цитат описания."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def extract(
        self,
        product: ProductInput,
        required_fields: list[str],
    ) -> ExtractionResult:
        """
        Вернуть характеристики из specs и description.

        Значения specs сохраняются дословно. Ответ модели принимается только
        если evidence — фрагмент description. Расхождение источников пишется
        в conflicts, значение specs при этом не заменяется.
        """
        fields = _unique(required_fields)
        attributes = _attributes_from_specs(product, fields)
        conflicts: list[AttributeConflict] = []

        if product.description.strip() and fields:
            candidates = _parse_description_response(
                await self._llm_client.complete(build_extraction_prompt(product, fields))
            )
            conflicts = _apply_description_candidates(product, fields, attributes, candidates)

        found = set(attributes)
        missing = [name for name in fields if name not in found]
        return ExtractionResult(
            attributes=list(attributes.values()),
            missing_fields=missing,
            conflicts=conflicts,
        )


def _attributes_from_specs(
    product: ProductInput,
    fields: list[str],
) -> dict[str, ExtractedAttribute]:
    attributes: dict[str, ExtractedAttribute] = {}
    for name in fields:
        value = _spec_value(product.specs, name)
        if value is None:
            continue
        attributes[name] = ExtractedAttribute(
            name=name,
            value=value,
            source=AttributeSource.SPECS,
            evidence=value,
        )
    return attributes


def _spec_value(specs: dict[str, str], field: str) -> str | None:
    """Найти значение spec по точному имени, иначе по первому совпадению без регистра."""
    if field in specs:
        value = specs[field]
        return value if value.strip() else None
    folded = field.strip().casefold()
    for key, value in specs.items():
        if key.strip().casefold() == folded and value.strip():
            return value
    return None


def _apply_description_candidates(
    product: ProductInput,
    fields: list[str],
    attributes: dict[str, ExtractedAttribute],
    candidates: DescriptionExtraction,
) -> list[AttributeConflict]:
    allowed: dict[str, str] = {}
    for field in fields:
        allowed.setdefault(_field_key(field), field)
    conflicts: list[AttributeConflict] = []
    conflict_names: set[str] = set()

    for candidate in candidates.attributes:
        name = allowed.get(_field_key(candidate.name))
        if name is None or not candidate.value.strip():
            continue
        quote = supported_quote(candidate.evidence, candidate.value, product.description)
        if quote is None:
            continue

        spec_value = _spec_value(product.specs, name)
        if spec_value is not None:
            if name not in conflict_names and not same_technical_value(spec_value, candidate.value):
                conflicts.append(
                    AttributeConflict(
                        name=name,
                        specs_value=spec_value,
                        description_value=candidate.value,
                        evidence=quote,
                    )
                )
                conflict_names.add(name)
            continue

        if name in attributes:
            continue
        attributes[name] = ExtractedAttribute(
            name=name,
            value=candidate.value,
            source=AttributeSource.DESCRIPTION,
            evidence=quote,
        )

    return conflicts


def _parse_description_response(raw: str) -> DescriptionExtraction:
    try:
        return DescriptionExtraction.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMError("LLM response does not match the extraction schema") from exc


def _field_key(name: str) -> str:
    return name.strip().casefold()


def _unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique
