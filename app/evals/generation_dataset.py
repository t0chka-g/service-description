"""Датасет оценки генерации. Разметка не обновляется по ответам моделей."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.product import GenerationInput


class ConfirmedFact(BaseModel):
    """Факт, который карточка может утверждать."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: str


class ConflictingValue(BaseModel):
    """Значение, которое нельзя подавать как подтвержденный факт."""

    model_config = ConfigDict(extra="forbid")

    name: str
    confirmed_value: str
    rejected_value: str
    note: str = ""


class GenerationDatasetCase(BaseModel):
    """Сценарий генерации с явной разметкой фактов, пропусков и запретов."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    input: GenerationInput
    confirmed_facts: list[ConfirmedFact]
    missing_data: list[str]
    conflicting_values: list[ConflictingValue] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    allowed_generalizations: str = ""


def dataset_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "generation_eval_cases.json"
    )


def load_generation_dataset(path: Path | None = None) -> list[GenerationDatasetCase]:
    """Загрузить сценарии оценки. Ответы моделей в файл не записываются."""

    payload = json.loads((path or dataset_path()).read_text(encoding="utf-8"))
    return [GenerationDatasetCase.model_validate(item) for item in payload]
