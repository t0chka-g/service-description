"""Калибровка LLM-судьи на заранее размеченных карточках.

Карточки передаются судье напрямую, без Generator.
Ожидаемый verdict и класс нарушения в промпт не входят.
Ground truth сравнивается только после ответа модели.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clients.llm import LLMCallMeta, LLMClient, LLMError, LLMErrorKind, LLMUsage
from app.evals.generation_dataset import (
    ConfirmedFact,
    ConflictingValue,
    GenerationDatasetCase,
)
from app.evals.generation_metrics import (
    ModelPrice,
    UsageCost,
    aggregate_costs,
    count_rate,
    mean,
    usage_cost,
)
from app.evals.generation_runner import judge_prompt_hash, load_prices
from app.evals.judge import ClaimKind, JudgeAssessment, JudgeVerdict
from app.evals.judge_checks import JudgeCheckResult, check_judge_assessment
from app.prompts.judge import build_judge_prompt
from app.schemas.product import GenerationInput, GenerationResult

_SAFE_MODEL = re.compile(r"[^\w.\-]+", re.UNICODE)

CALIBRATION_METHOD = (
    "Карточки эталонного набора передаются LLM-судье напрямую; Generator "
    "не вызывается. В промпт входят только идентификационные данные товара, "
    "подтвержденные характеристики, сведения об отсутствии данных, "
    "расхождения исходников и текст карточки. Ожидаемый verdict, "
    "expected_violation_class, forbidden_claims и идентификаторы сценариев, "
    "раскрывающие нарушение, в промпт не входят. "
    "Verdict Accuracy и Claim Classification Accuracy считаются только по "
    "сценариям со структурированным ответом без API-ошибки и без "
    "judge_error (выдуманные цитаты / конфликт evidence). "
    "Внутренне противоречивые ответы судьи исключаются из знаменателей "
    "классификации и попадают в manual_review. "
    "False Positive Rate — доля ложных обвинений среди сценариев без "
    "семантического нарушения (none, acceptable_generalization, "
    "keyword_stuffing). "
    "False Negative Rate — доля пропусков среди сценариев с ожидаемым "
    "нарушением. Invalid Evidence Rate — доля цитат, которых нет в "
    "карточке, среди всех цитат unsupported_claims и evidence. "
    "Judge Error Rate — структурные сбои разбора и невалидные цитаты. "
    "API Error Rate считается отдельно и не смешивается с ошибками "
    "классификации. Средние оценки судьи — описательная сводка одного "
    "прогона, а не объективная характеристика модели. Пустой знаменатель "
    "дает null."
)


class ViolationClass(StrEnum):
    NONE = "none"
    UNIT_SUBSTITUTION = "unit_substitution"
    FABRICATED_CERTIFICATE = "fabricated_certificate"
    UNSUPPORTED_PURPOSE = "unsupported_purpose"
    UNSUPPORTED_COMMERCIAL_ADVANTAGE = "unsupported_commercial_advantage"
    ACCEPTABLE_GENERALIZATION = "acceptable_generalization"
    SPEC_CONTRADICTION = "spec_contradiction"
    KEYWORD_STUFFING = "keyword_stuffing"


class CalibrationRunStatus(StrEnum):
    SCORED = "scored"
    API_ERROR = "api_error"
    JUDGE_ERROR = "judge_error"


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
    expected_claim_kind: ClaimKind | None
    allowed_verdicts: list[JudgeVerdict]
    actual_verdict: JudgeVerdict | None
    verdict_match: bool
    claim_classification_match: bool | None
    issue_presence_match: bool
    matched: bool
    needs_manual_review: bool
    judge_error: bool
    review_reason: str = ""
    discrepancies: list[str] = Field(default_factory=list)
    error_message: str | None = None


class CountRateProxy(BaseModel):
    """Совместимый с generation_metrics.CountRate контракт отчета."""

    model_config = ConfigDict(extra="forbid")

    numerator: int
    denominator: int
    value: float | None


class CalibrationRates(BaseModel):
    """Доли с явными числителями и знаменателями. value is None при пустом знаменателе."""

    model_config = ConfigDict(extra="forbid")

    verdict_accuracy: CountRateProxy
    claim_classification_accuracy: CountRateProxy
    false_positive_rate: CountRateProxy
    false_negative_rate: CountRateProxy
    invalid_evidence_rate: CountRateProxy
    judge_error_rate: CountRateProxy
    api_error_rate: CountRateProxy


class CalibrationScenarioReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    description: str
    status: CalibrationRunStatus
    expected_violation_class: ViolationClass
    expected_claim_kind: ClaimKind | None
    allowed_verdicts: list[JudgeVerdict]
    actual_verdict: JudgeVerdict | None = None
    verdict_match: bool = False
    claim_classification_match: bool | None = None
    issue_presence_match: bool = False
    matched: bool = False
    needs_manual_review: bool = False
    manual_review_reason: str = ""
    discrepancies: list[str] = Field(default_factory=list)
    judge: JudgeAssessment | None = None
    check: dict[str, Any] | None = None
    latency_seconds: float | None = None
    response_model: str | None = None
    usage: LLMUsage | None = None
    cost: UsageCost | None = None
    error_message: str | None = None
    error_kind: str | None = None


class CalibrationRunResult(BaseModel):
    """Краткий итог для unit-тестов и обратной совместимости."""

    model_config = ConfigDict(extra="forbid")

    comparisons: list[CalibrationComparison]
    matched_count: int
    case_count: int


class JudgeCalibrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    model: str
    base_url: str | None = None
    temperature: float | None = None
    determinism_note: str
    prompt_hash: str
    schema_hash: str
    dataset_hash: str
    selected_case_ids: list[str]
    scenario_count: int
    methodology: str
    rates: CalibrationRates
    mean_factual_consistency: float | None
    mean_seo_quality: float | None
    mean_language_quality: float | None
    score_means_note: str
    mean_latency_seconds: float | None
    cost: UsageCost
    scenarios: list[CalibrationScenarioReport]
    manual_review_cases: list[str]
    api_errors: list[str]
    judge_errors: list[str]


def calibration_dataset_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "judge_calibration_cases.json"
    )


def default_calibration_report_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "reports" / "evals" / "judge_calibration"


def load_judge_calibration_dataset(
    path: Path | None = None,
) -> list[JudgeCalibrationCase]:
    payload = json.loads((path or calibration_dataset_path()).read_text(encoding="utf-8"))
    return [JudgeCalibrationCase.model_validate(item) for item in payload]


def select_calibration_cases(
    cases: Sequence[JudgeCalibrationCase],
    case_ids: Sequence[str] | None = None,
) -> list[JudgeCalibrationCase]:
    if not case_ids:
        return list(cases)
    wanted = [item.strip() for item in case_ids if item.strip()]
    by_id = {case.id: case for case in cases}
    missing = [case_id for case_id in wanted if case_id not in by_id]
    if missing:
        raise ValueError(f"Неизвестные case_id: {', '.join(missing)}")
    return [by_id[case_id] for case_id in wanted]


def as_prompt_case(case: JudgeCalibrationCase) -> GenerationDatasetCase:
    """Собрать вход судьи без эталонных меток калибровки."""

    return GenerationDatasetCase(
        id="calibration_case",
        description="calibration",
        input=case.input,
        confirmed_facts=case.confirmed_facts,
        missing_data=case.missing_data,
        conflicting_values=case.conflicting_values,
        forbidden_claims=[],
        allowed_generalizations="",
    )


def expects_semantic_issue(violation: ViolationClass) -> bool:
    """SEO-дефект без ложного факта не считается семантическим нарушением."""

    return violation not in {
        ViolationClass.NONE,
        ViolationClass.ACCEPTABLE_GENERALIZATION,
        ViolationClass.KEYWORD_STUFFING,
    }


def expected_claim_kind(violation: ViolationClass) -> ClaimKind | None:
    """Ожидаемый ClaimKind для ground-truth класса. Не передается в промпт."""

    mapping: dict[ViolationClass, ClaimKind | None] = {
        ViolationClass.NONE: None,
        ViolationClass.ACCEPTABLE_GENERALIZATION: None,
        ViolationClass.UNIT_SUBSTITUTION: ClaimKind.DIRECT_CONTRADICTION,
        ViolationClass.SPEC_CONTRADICTION: ClaimKind.DIRECT_CONTRADICTION,
        ViolationClass.FABRICATED_CERTIFICATE: ClaimKind.UNSUPPORTED_CLAIM,
        ViolationClass.UNSUPPORTED_PURPOSE: ClaimKind.UNSUPPORTED_CLAIM,
        ViolationClass.UNSUPPORTED_COMMERCIAL_ADVANTAGE: ClaimKind.UNSUPPORTED_CLAIM,
        ViolationClass.KEYWORD_STUFFING: None,
    }
    return mapping[violation]


def judge_schema_hash() -> str:
    payload = json.dumps(
        JudgeAssessment.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calibration_dataset_hash(cases: Sequence[JudgeCalibrationCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_calibration(
    case: JudgeCalibrationCase,
    assessment: JudgeAssessment | None,
    *,
    error_message: str | None = None,
    check: JudgeCheckResult | None = None,
) -> CalibrationComparison:
    expected_kind = expected_claim_kind(case.expected_violation_class)
    if assessment is None:
        return CalibrationComparison(
            case_id=case.id,
            expected_violation_class=case.expected_violation_class,
            expected_claim_kind=expected_kind,
            allowed_verdicts=case.allowed_verdicts,
            actual_verdict=None,
            verdict_match=False,
            claim_classification_match=None,
            issue_presence_match=False,
            matched=False,
            needs_manual_review=True,
            judge_error=True,
            review_reason="Ответ судьи отсутствует.",
            discrepancies=["Ответ судьи отсутствует."],
            error_message=error_message,
        )
    checked = check or check_judge_assessment(assessment, case.card)
    expected_issue = expects_semantic_issue(case.expected_violation_class)
    verified_issue = checked.confirmed_semantic_issue
    verdict_match = assessment.verdict in case.allowed_verdicts
    classification_match = _claim_classification_match(
        case,
        assessment,
        checked,
        expected_kind=expected_kind,
    )
    issue_match = verified_issue == expected_issue
    discrepancies = _discrepancies(
        case,
        assessment,
        checked,
        verdict_match=verdict_match,
        issue_match=issue_match,
        classification_match=classification_match,
    )
    matched = (
        verdict_match
        and issue_match
        and classification_match is True
        and not checked.judge_error
        and not checked.inconsistent
    )
    return CalibrationComparison(
        case_id=case.id,
        expected_violation_class=case.expected_violation_class,
        expected_claim_kind=expected_kind,
        allowed_verdicts=case.allowed_verdicts,
        actual_verdict=assessment.verdict,
        verdict_match=verdict_match,
        claim_classification_match=classification_match,
        issue_presence_match=issue_match,
        matched=matched,
        needs_manual_review=checked.needs_manual_review or not matched,
        judge_error=checked.judge_error,
        review_reason=checked.review_reason,
        discrepancies=discrepancies,
        error_message=error_message,
    )


async def run_judge_calibration(
    cases: Sequence[JudgeCalibrationCase],
    judge_client: LLMClient,
    *,
    model: str = "judge",
    temperature: float | None = None,
    prices: dict[str, ModelPrice] | None = None,
    base_url: str | None = None,
    report_dir: Path | None = None,
    created_at: datetime | None = None,
) -> CalibrationRunResult:
    """Прогнать готовые карточки через судью. Generator не вызывается."""

    report = await run_judge_calibration_report(
        cases,
        judge_client,
        model=model,
        temperature=temperature,
        prices=prices,
        base_url=base_url,
        report_dir=report_dir,
        created_at=created_at,
    )
    comparisons = [
        CalibrationComparison(
            case_id=item.case_id,
            expected_violation_class=item.expected_violation_class,
            expected_claim_kind=item.expected_claim_kind,
            allowed_verdicts=item.allowed_verdicts,
            actual_verdict=item.actual_verdict,
            verdict_match=item.verdict_match,
            claim_classification_match=item.claim_classification_match,
            issue_presence_match=item.issue_presence_match,
            matched=item.matched,
            needs_manual_review=item.needs_manual_review,
            judge_error=item.status is CalibrationRunStatus.JUDGE_ERROR,
            review_reason=item.manual_review_reason,
            discrepancies=item.discrepancies,
            error_message=item.error_message,
        )
        for item in report.scenarios
    ]
    return CalibrationRunResult(
        comparisons=comparisons,
        matched_count=sum(item.matched for item in comparisons),
        case_count=len(comparisons),
    )


async def run_judge_calibration_report(
    cases: Sequence[JudgeCalibrationCase],
    judge_client: LLMClient,
    *,
    model: str,
    temperature: float | None = None,
    prices: dict[str, ModelPrice] | None = None,
    base_url: str | None = None,
    report_dir: Path | None = None,
    created_at: datetime | None = None,
) -> JudgeCalibrationReport:
    """Полный прогон калибровки с метриками и записью отчета."""

    tariff = (prices or {}).get(model)
    scenarios: list[CalibrationScenarioReport] = []
    for case in cases:
        scenarios.append(
            await _run_calibration_case(case, judge_client, judge_price=tariff)
        )
    report = build_calibration_report(
        scenarios,
        cases,
        model=model,
        temperature=temperature,
        base_url=base_url,
        created_at=created_at,
    )
    if report_dir is not None:
        write_calibration_report(report, report_dir)
    return report


def build_calibration_report(
    scenarios: Sequence[CalibrationScenarioReport],
    cases: Sequence[JudgeCalibrationCase],
    *,
    model: str,
    temperature: float | None = None,
    base_url: str | None = None,
    created_at: datetime | None = None,
) -> JudgeCalibrationReport:
    created = created_at or datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)

    api_errors = [
        item.case_id for item in scenarios if item.status is CalibrationRunStatus.API_ERROR
    ]
    judge_errors = [
        item.case_id
        for item in scenarios
        if item.status is CalibrationRunStatus.JUDGE_ERROR
    ]
    eligible = [
        item
        for item in scenarios
        if item.status is CalibrationRunStatus.SCORED
        and item.check is not None
        and not item.check.get("inconsistent", False)
    ]
    clean = [
        item
        for item in eligible
        if not expects_semantic_issue(item.expected_violation_class)
    ]
    issues = [
        item
        for item in eligible
        if expects_semantic_issue(item.expected_violation_class)
    ]
    quotes_total, quotes_invalid = _quote_counts(scenarios)
    scored_with_judge = [item for item in eligible if item.judge is not None]
    latencies = [
        item.latency_seconds
        for item in scenarios
        if item.latency_seconds is not None
    ]
    manual = [item.case_id for item in scenarios if item.needs_manual_review]

    return JudgeCalibrationReport(
        created_at=created,
        model=model,
        base_url=base_url or None,
        temperature=temperature,
        determinism_note=(
            "Температура передается в API, если задана. "
            "Повторный прогон может отличаться."
        ),
        prompt_hash=judge_prompt_hash(),
        schema_hash=judge_schema_hash(),
        dataset_hash=calibration_dataset_hash(cases),
        selected_case_ids=[case.id for case in cases],
        scenario_count=len(scenarios),
        methodology=CALIBRATION_METHOD,
        rates=CalibrationRates(
            verdict_accuracy=_rate(
                sum(item.verdict_match for item in eligible),
                len(eligible),
            ),
            claim_classification_accuracy=_rate(
                sum(item.claim_classification_match is True for item in eligible),
                len(eligible),
            ),
            false_positive_rate=_rate(
                sum(_false_positive(item) for item in clean),
                len(clean),
            ),
            false_negative_rate=_rate(
                sum(_false_negative(item) for item in issues),
                len(issues),
            ),
            invalid_evidence_rate=_rate(quotes_invalid, quotes_total),
            judge_error_rate=_rate(len(judge_errors), len(scenarios)),
            api_error_rate=_rate(len(api_errors), len(scenarios)),
        ),
        mean_factual_consistency=mean(
            [
                float(item.judge.factual_consistency_score)
                for item in scored_with_judge
                if item.judge is not None
            ]
        ),
        mean_seo_quality=mean(
            [
                float(item.judge.seo_quality_score)
                for item in scored_with_judge
                if item.judge is not None
            ]
        ),
        mean_language_quality=mean(
            [
                float(item.judge.language_quality_score)
                for item in scored_with_judge
                if item.judge is not None
            ]
        ),
        score_means_note=(
            "Средние оценки судьи — описательная сводка одного прогона "
            "и не являются объективной характеристикой модели."
        ),
        mean_latency_seconds=mean(latencies),
        cost=aggregate_costs([(True, item.cost) for item in scenarios]),
        scenarios=list(scenarios),
        manual_review_cases=manual,
        api_errors=api_errors,
        judge_errors=judge_errors,
    )


def _report_suffix(report: JudgeCalibrationReport) -> str:
    selected = list(report.selected_case_ids)
    if len(selected) == 1:
        return _SAFE_MODEL.sub("_", selected[0]).strip("_") or "case"
    full_ids = [case.id for case in load_judge_calibration_dataset()]
    if selected == full_ids:
        return "all"
    return f"{len(selected)}cases"


def write_calibration_report(report: JudgeCalibrationReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.strftime("%Y%m%dT%H%M%S%fZ")
    safe_model = _SAFE_MODEL.sub("_", report.model).strip("_") or "model"
    suffix = _report_suffix(report)
    path = directory / f"{stamp}_{safe_model}_{suffix}.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


class _RecordingClient:
    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.metas: list[LLMCallMeta | None] = []
        self.attempted = False

    async def complete(self, prompt: str, *, response_model: object = None) -> str:
        self.attempted = True
        try:
            if response_model is None:
                response = await self._inner.complete(prompt)
            else:
                response = await self._inner.complete(
                    prompt,
                    response_model=response_model,
                )
        except BaseException:
            meta = getattr(self._inner, "last_call", None)
            self.metas.append(meta if isinstance(meta, LLMCallMeta) else None)
            raise
        meta = getattr(self._inner, "last_call", None)
        self.metas.append(meta if isinstance(meta, LLMCallMeta) else None)
        return response


async def _run_calibration_case(
    case: JudgeCalibrationCase,
    judge_client: LLMClient,
    *,
    judge_price: ModelPrice | None,
) -> CalibrationScenarioReport:
    recorder = _RecordingClient(judge_client)
    prompt = build_judge_prompt(as_prompt_case(case), case.card)
    started = time.perf_counter()
    try:
        raw = await recorder.complete(prompt, response_model=JudgeAssessment)
        assessment = JudgeAssessment.model_validate(json.loads(raw))
    except LLMError as exc:
        latency = time.perf_counter() - started
        meta = recorder.metas[-1] if recorder.metas else None
        status = (
            CalibrationRunStatus.API_ERROR
            if exc.kind is LLMErrorKind.API
            else CalibrationRunStatus.JUDGE_ERROR
        )
        comparison = compare_calibration(case, None, error_message=str(exc))
        return _scenario_from_comparison(
            case,
            comparison,
            status=status,
            latency=latency if recorder.attempted else None,
            meta=meta,
            price=judge_price,
            error_kind=exc.kind.value,
        )
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        latency = time.perf_counter() - started
        meta = recorder.metas[-1] if recorder.metas else None
        comparison = compare_calibration(case, None, error_message=str(exc))
        return _scenario_from_comparison(
            case,
            comparison,
            status=CalibrationRunStatus.JUDGE_ERROR,
            latency=latency if recorder.attempted else None,
            meta=meta,
            price=judge_price,
            error_kind="model",
        )

    latency = time.perf_counter() - started
    meta = recorder.metas[-1] if recorder.metas else None
    checked = check_judge_assessment(assessment, case.card)
    comparison = compare_calibration(case, assessment, check=checked)
    status = (
        CalibrationRunStatus.JUDGE_ERROR
        if checked.judge_error
        else CalibrationRunStatus.SCORED
    )
    return _scenario_from_comparison(
        case,
        comparison,
        status=status,
        latency=latency,
        meta=meta,
        price=judge_price,
        assessment=assessment,
        check=checked,
    )


def _scenario_from_comparison(
    case: JudgeCalibrationCase,
    comparison: CalibrationComparison,
    *,
    status: CalibrationRunStatus,
    latency: float | None,
    meta: LLMCallMeta | None,
    price: ModelPrice | None,
    assessment: JudgeAssessment | None = None,
    check: JudgeCheckResult | None = None,
    error_kind: str | None = None,
) -> CalibrationScenarioReport:
    usage = meta.usage if meta is not None else None
    return CalibrationScenarioReport(
        case_id=case.id,
        description=case.description,
        status=status,
        expected_violation_class=case.expected_violation_class,
        expected_claim_kind=comparison.expected_claim_kind,
        allowed_verdicts=case.allowed_verdicts,
        actual_verdict=comparison.actual_verdict,
        verdict_match=comparison.verdict_match,
        claim_classification_match=comparison.claim_classification_match,
        issue_presence_match=comparison.issue_presence_match,
        matched=comparison.matched and status is CalibrationRunStatus.SCORED,
        needs_manual_review=comparison.needs_manual_review,
        manual_review_reason=comparison.review_reason,
        discrepancies=comparison.discrepancies,
        judge=assessment,
        check=_check_payload(check) if check is not None else None,
        latency_seconds=latency,
        response_model=meta.response_model if meta is not None else None,
        usage=usage,
        cost=usage_cost([usage], price) if usage is not None else None,
        error_message=comparison.error_message,
        error_kind=error_kind,
    )


def _check_payload(check: JudgeCheckResult) -> dict[str, Any]:
    return {
        "fabricated_quotes": list(check.fabricated_quotes),
        "inconsistent": check.inconsistent,
        "confirmed_semantic_issue": check.confirmed_semantic_issue,
        "needs_manual_review": check.needs_manual_review,
        "review_reason": check.review_reason,
        "judge_error": check.judge_error,
    }


def _claim_classification_match(
    case: JudgeCalibrationCase,
    assessment: JudgeAssessment,
    checked: JudgeCheckResult,
    *,
    expected_kind: ClaimKind | None,
) -> bool | None:
    if checked.judge_error or checked.inconsistent:
        return None
    blob = _card_blob(case.card)
    verified_claims = [
        claim
        for claim in assessment.unsupported_claims
        if claim.quote.strip() in blob
    ]
    if expected_kind is None:
        return not verified_claims and not assessment.critical_errors
    if any(claim.kind is expected_kind for claim in verified_claims):
        return True
    # Прямое противоречие иногда попадает только в critical_errors без kind.
    if expected_kind is ClaimKind.DIRECT_CONTRADICTION and assessment.critical_errors:
        return True
    return False


def _discrepancies(
    case: JudgeCalibrationCase,
    assessment: JudgeAssessment,
    checked: JudgeCheckResult,
    *,
    verdict_match: bool,
    issue_match: bool,
    classification_match: bool | None,
) -> list[str]:
    items: list[str] = []
    if checked.fabricated_quotes:
        items.append("Цитаты судьи отсутствуют в карточке.")
    if checked.inconsistent:
        items.append("Вердикт судьи не согласован с претензиями.")
    if not verdict_match:
        items.append(
            f"verdict={assessment.verdict.value}; допустимо "
            f"{[item.value for item in case.allowed_verdicts]}."
        )
    if not issue_match:
        expected = "ожидалось нарушение" if expects_semantic_issue(
            case.expected_violation_class
        ) else "нарушение не ожидалось"
        items.append(f"Наличие нарушения: {expected}.")
    if classification_match is False:
        items.append(
            f"Класс нарушения: ожидался "
            f"{expected_claim_kind(case.expected_violation_class) or 'отсутствие претензий'}."
        )
    return items


def _false_positive(item: CalibrationScenarioReport) -> bool:
    if item.judge is None or item.check is None:
        return False
    return bool(item.check.get("confirmed_semantic_issue"))


def _false_negative(item: CalibrationScenarioReport) -> bool:
    if item.judge is None or item.check is None:
        return True
    return not bool(item.check.get("confirmed_semantic_issue"))


def _quote_counts(scenarios: Sequence[CalibrationScenarioReport]) -> tuple[int, int]:
    total = 0
    invalid = 0
    for item in scenarios:
        if item.judge is None:
            continue
        quotes = [claim.quote for claim in item.judge.unsupported_claims]
        quotes.extend(evidence.quote for evidence in item.judge.evidence)
        fabricated = list(item.check.get("fabricated_quotes") or []) if item.check else []
        # fabricated_quotes сохраняет порядок появлений; считаем по нему invalid.
        invalid += len(fabricated)
        total += len(quotes)
    return total, invalid


def _rate(numerator: int, denominator: int) -> CountRateProxy:
    result = count_rate(numerator, denominator)
    return CountRateProxy(
        numerator=result.numerator,
        denominator=result.denominator,
        value=result.value,
    )


def _card_blob(card: GenerationResult) -> str:
    return "\n".join(
        [card.seo_title, card.seo_description, card.commercial_block, *card.search_queries]
    )


# Re-export for CLI convenience without forcing callers to know generation_runner.
__all__ = [
    "CALIBRATION_METHOD",
    "CalibrationComparison",
    "CalibrationRates",
    "CalibrationRunResult",
    "CalibrationRunStatus",
    "CalibrationScenarioReport",
    "JudgeCalibrationCase",
    "JudgeCalibrationReport",
    "ViolationClass",
    "as_prompt_case",
    "build_calibration_report",
    "calibration_dataset_hash",
    "calibration_dataset_path",
    "compare_calibration",
    "default_calibration_report_dir",
    "expected_claim_kind",
    "expects_semantic_issue",
    "judge_schema_hash",
    "load_judge_calibration_dataset",
    "load_prices",
    "run_judge_calibration",
    "run_judge_calibration_report",
    "select_calibration_cases",
    "write_calibration_report",
]
