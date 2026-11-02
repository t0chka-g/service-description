"""Схемы датасета оценки и JSON-отчета."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.product import AttributeConflict, ExtractedAttribute, ProductInput


class ScenarioStatus(StrEnum):
    SCORED = "scored"
    API_ERROR = "api_error"
    MODEL_ERROR = "model_error"


class PairMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class QualityMetrics(BaseModel):
    """Метрики одного сценария или пула сценариев с ответом модели."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    expected_pairs: int
    predicted_pairs: int
    hallucination_rate: float
    model_claims: int
    unsupported_claims: int
    pipeline_hallucination_rate: float
    pipeline_description_attributes: int
    pipeline_unsupported: int
    specs_preservation: float
    specs_expected: int
    specs_preserved: int
    numeric_preservation: float
    numeric_expected: int
    numeric_exact: int
    conflicts: PairMetrics


class AggregateMetrics(QualityMetrics):
    macro_f1: float
    description_cases_in_macro: int


class ValueMismatch(BaseModel):
    name: str
    expected: str | None = None
    actual: str | None = None


class ClaimRecord(BaseModel):
    name: str
    value: str
    evidence: str = ""


class ConflictRecord(BaseModel):
    name: str
    specs_value: str
    description_value: str


class ManualReviewItem(BaseModel):
    name: str
    expected: str
    actual: str
    reason: str


class ScenarioErrors(BaseModel):
    false_positives: list[ValueMismatch] = Field(default_factory=list)
    false_negatives: list[ValueMismatch] = Field(default_factory=list)
    missed_conflicts: list[ConflictRecord] = Field(default_factory=list)
    spurious_conflicts: list[ConflictRecord] = Field(default_factory=list)
    numeric_altered: list[ValueMismatch] = Field(default_factory=list)
    unsupported_claims: list[ClaimRecord] = Field(default_factory=list)
    forbidden_hits: list[ClaimRecord] = Field(default_factory=list)


class ScenarioReport(BaseModel):
    case_id: str
    description: str
    status: ScenarioStatus
    error_message: str | None = None
    model_called: bool = False
    product: ProductInput
    required_fields: list[str]
    metrics: QualityMetrics | None = None
    errors: ScenarioErrors = Field(default_factory=ScenarioErrors)
    manual_review: list[ManualReviewItem] = Field(default_factory=list)
    model_response: Any = None
    expected: dict[str, Any]
    actual: dict[str, Any] | None = None


class LabelingNote(BaseModel):
    case_id: str
    note: str


class EvaluationReport(BaseModel):
    created_at: datetime
    model: str
    base_url: str | None = None
    prompt_hash: str
    scenario_count: int
    scored_count: int
    api_error_count: int
    model_error_count: int
    methodology: str
    labeling_notes: list[LabelingNote]
    metrics: AggregateMetrics | None
    scenarios: list[ScenarioReport]


class ExtractionCase(BaseModel):
    """Один сценарий ground truth. Поле llm_response датасета сюда не входит."""

    model_config = ConfigDict(extra="ignore")

    id: str
    description: str = ""
    input: ProductInput
    required_fields: list[str]
    expected: dict[str, ExtractedAttribute]
    missing_fields: list[str] = Field(default_factory=list)
    conflicts: list[AttributeConflict] = Field(default_factory=list)
    forbidden_values: dict[str, str] = Field(default_factory=dict)


METHODOLOGY = (
    "Precision, Recall и F1 считаются только по парам «характеристика — значение», "
    "извлеченным из description. Копирование specs в эти метрики не входит. "
    "Имя сравнивается без регистра и лишних пробелов. Значение сравнивается "
    "без лишних пробелов; числа, разделители и единицы не приводятся. "
    "Неверное значение — это одновременно FP и FN. "
    "Характеристика из missing_fields не является FN. "
    "Пустые expected и predicted: P=1, R=1, F1=1. "
    "Нет predicted при непустом expected: P=1, R=0, F1=0. "
    "Нет expected при непустом predicted: P=0, R=1, F1=0. "
    "Hallucination rate — доля атрибутов сырого ответа модели, цитата которых "
    "не подтверждает value в тексте description. Ноль атрибутов дает 0. "
    "pipeline_hallucination_rate считает description-атрибуты, прошедшие в результат "
    "extractor. Сохранение specs, конфликты и числовые значения считаются отдельно. "
    "macro_f1 усредняет сценарии, где есть пары description; сценарии только со specs "
    "его не поднимают. Близкие по записи значения не засчитываются как TP и попадают "
    "в manual_review. Ошибки API и сбои разбора ответа в метрики качества не входят."
)

STATIC_LABELING_NOTES: tuple[LabelingNote, ...] = (
    LabelingNote(
        case_id="synonym_names",
        note=(
            "В описании есть формулировка «тип K», в specs значение «K». "
            "Разметка не содержит конфликта и ожидает значение из specs. "
            "Если модель вернет «тип K», extractor зафиксирует конфликт, "
            "и относительно этой разметки он будет ложным. Ground truth не изменялся."
        ),
    ),
)
