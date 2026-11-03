"""Калибровка LLM-судьи на заранее размеченных карточках.

Карточки передаются судье напрямую, без Generator.
Ожидаемый verdict и класс нарушения в промпт не входят.
"""

import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.clients.llm import LLMClient, LLMError
from app.evals.generation_dataset import (
    ConfirmedFact,
    ConflictingValue,
    GenerationDatasetCase,
)
from app.evals.judge import JudgeAssessment, JudgeVerdict
from app.evals.judge_checks import check_judge_assessment
from app.prompts.judge import build_judge_prompt
from app.schemas.product import GenerationInput, GenerationResult


class ViolationClass(StrEnum):
    NONE = "none"
    UNIT_SUBSTITUTION = "unit_substitution"
    FABRICATED_CERTIFICATE = "fabricated_certificate"
    UNSUPPORTED_PURPOSE = "unsupported_purpose"
    UNSUPPORTED_COMMERCIAL_ADVANTAGE = "unsupported_commercial_advantage"
    ACCEPTABLE_GENERALIZATION = "acceptable_generalization"
    SPEC_CONTRADICTION = "spec_contradiction"


class JudgeCalibrationCase(BaseModel):
    """Сценарий калибровки: вход, готовая карточка и эталон вне промпта."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    input: GenerationInput
    confirmed_facts: list[ConfirmedFact]
    missing_data: list[str] = Field(default_factory=list)
    conflicting_values: list[ConflictingValue] = Field(default_factory=list)
    card: GenerationResult
    expected_violation_class: ViolationClass
    allowed_verdicts: list[JudgeVerdict]


class CalibrationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    expected_violation_class: ViolationClass
    allowed_verdicts: list[JudgeVerdict]
    actual_verdict: JudgeVerdict | None
    verdict_match: bool
    issue_presence_match: bool
    matched: bool
    needs_manual_review: bool
    judge_error: bool
    review_reason: str = ""
    error_message: str | None = None


class CalibrationRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparisons: list[CalibrationComparison]
    matched_count: int
    case_count: int


def calibration_dataset_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "judge_calibration_cases.json"
    )


def load_judge_calibration_dataset(
    path: Path | None = None,
) -> list[JudgeCalibrationCase]:
    payload = json.loads((path or calibration_dataset_path()).read_text(encoding="utf-8"))
    return [JudgeCalibrationCase.model_validate(item) for item in payload]


def as_prompt_case(case: JudgeCalibrationCase) -> GenerationDatasetCase:
    """Собрать вход судьи без эталонных меток калибровки."""

    return GenerationDatasetCase(
        id=case.id,
        description=case.description,
        input=case.input,
        confirmed_facts=case.confirmed_facts,
        missing_data=case.missing_data,
        conflicting_values=case.conflicting_values,
        forbidden_claims=[],
        allowed_generalizations="",
    )


def expects_semantic_issue(violation: ViolationClass) -> bool:
    return violation not in {
        ViolationClass.NONE,
        ViolationClass.ACCEPTABLE_GENERALIZATION,
    }


def compare_calibration(
    case: JudgeCalibrationCase,
    assessment: JudgeAssessment | None,
    *,
    error_message: str | None = None,
) -> CalibrationComparison:
    if assessment is None:
        return CalibrationComparison(
            case_id=case.id,
            expected_violation_class=case.expected_violation_class,
            allowed_verdicts=case.allowed_verdicts,
            actual_verdict=None,
            verdict_match=False,
            issue_presence_match=False,
            matched=False,
            needs_manual_review=True,
            judge_error=True,
            review_reason="Ответ судьи отсутствует.",
            error_message=error_message,
        )
    checked = check_judge_assessment(assessment, case.card)
    blob = _card_blob(case.card)
    verified_issue = bool(assessment.critical_errors) or any(
        claim.quote.strip() in blob for claim in assessment.unsupported_claims
    )
    expected_issue = expects_semantic_issue(case.expected_violation_class)
    verdict_match = assessment.verdict in case.allowed_verdicts
    issue_match = verified_issue == expected_issue
    matched = (
        verdict_match
        and issue_match
        and not checked.judge_error
        and not checked.inconsistent
    )
    return CalibrationComparison(
        case_id=case.id,
        expected_violation_class=case.expected_violation_class,
        allowed_verdicts=case.allowed_verdicts,
        actual_verdict=assessment.verdict,
        verdict_match=verdict_match,
        issue_presence_match=issue_match,
        matched=matched,
        needs_manual_review=checked.needs_manual_review or not matched,
        judge_error=checked.judge_error,
        review_reason=checked.review_reason,
        error_message=error_message,
    )


async def run_judge_calibration(
    cases: Sequence[JudgeCalibrationCase],
    judge_client: LLMClient,
) -> CalibrationRunResult:
    """Прогнать готовые карточки через судью. Generator не вызывается."""

    comparisons: list[CalibrationComparison] = []
    for case in cases:
        prompt = build_judge_prompt(as_prompt_case(case), case.card)
        try:
            raw = await judge_client.complete(prompt, response_model=JudgeAssessment)
            assessment = JudgeAssessment.model_validate(json.loads(raw))
            comparisons.append(compare_calibration(case, assessment))
        except (LLMError, json.JSONDecodeError, TypeError, ValueError) as exc:
            comparisons.append(
                compare_calibration(case, None, error_message=str(exc))
            )
    matched_count = sum(item.matched for item in comparisons)
    return CalibrationRunResult(
        comparisons=comparisons,
        matched_count=matched_count,
        case_count=len(comparisons),
    )


def _card_blob(card: GenerationResult) -> str:
    return "\n".join(
        [card.seo_title, card.seo_description, card.commercial_block, *card.search_queries]
    )
