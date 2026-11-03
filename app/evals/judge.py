"""Структурированный вердикт LLM-судьи.

Оценка судьи сохраняется как есть. Она не заменяет ручную проверку.
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClaimKind(StrEnum):
    CONFIRMED_FACT = "confirmed_fact"
    ACCEPTABLE_GENERALIZATION = "acceptable_generalization"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    DIRECT_CONTRADICTION = "direct_contradiction"


class JudgeVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"


class UnsupportedClaim(BaseModel):
    """Цитата из карточки, которую входные данные не подтверждают."""

    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    kind: ClaimKind

    @model_validator(mode="after")
    def only_unconfirmed(self) -> Self:
        if self.kind not in {
            ClaimKind.UNSUPPORTED_CLAIM,
            ClaimKind.DIRECT_CONTRADICTION,
        }:
            raise ValueError("unsupported_claims accepts only unconfirmed kinds")
        return self


class EvidenceItem(BaseModel):
    """Фрагмент карточки и его отношение к входным данным."""

    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1)
    kind: ClaimKind
    explanation: str = Field(min_length=1)


class JudgeAssessment(BaseModel):
    """Контракт Structured Outputs. Неполный ответ не получает оценку."""

    model_config = ConfigDict(extra="forbid")

    factual_consistency_score: int = Field(ge=0, le=5)
    seo_quality_score: int = Field(ge=0, le=5)
    language_quality_score: int = Field(ge=0, le=5)
    unsupported_claims: list[UnsupportedClaim]
    critical_errors: list[str]
    evidence: list[EvidenceItem]
    reasoning_summary: str = Field(min_length=1)
    verdict: JudgeVerdict
