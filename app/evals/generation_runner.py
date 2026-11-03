"""Прогон SEO-генератора и независимого судьи.

Ошибки API и разбора генератора не смешиваются с отказом судьи.
Судья не вызывается, если карточка не собрана.
"""

import hashlib
import inspect
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
from app.evals.generation_dataset import GenerationDatasetCase
from app.evals.generation_metrics import (
    CountRate,
    GenerationRates,
    ModelPrice,
    UsageCost,
    aggregate_costs,
    count_rate,
    mean,
    usage_cost,
)
from app.evals.judge import JudgeAssessment
from app.evals.judge_checks import JudgeCheckResult, check_judge_assessment
from app.prompts.generation import build_generation_prompt
from app.prompts.judge import build_judge_prompt
from app.schemas.product import (
    SEARCH_QUERIES_MAX,
    SEARCH_QUERIES_MIN,
    SEO_DESCRIPTION_MAX_LENGTH,
    SEO_DESCRIPTION_MIN_LENGTH,
    SEO_TITLE_MAX_LENGTH,
    GenerationResult,
    assess_generation_status,
)
from app.schemas.product import GenerationStatus as GateStatus
from app.services.generator import GenerationDraft, SEOGenerator

GENERATION_EVAL_METHOD = (
    "Валидная генерация — ответ SEOGenerator со статусом complete. "
    "Знаменатель valid_generation_rate, schema_compliance_rate, "
    "generation_error_rate, api_error_rate и model_error_rate — сценарии, "
    "у которых входной шлюз открыт. Сценарии insufficient_data в этот "
    "знаменатель не входят: они считаются только в insufficient_data_rate "
    "от всех сценариев. "
    "Schema compliance считается по открытым сценариям без ошибки API: "
    "сетевой сбой не является ошибкой схемы. Длины и число запросов "
    "считаются по сырому JSON, разобранному как GenerationDraft. "
    "Пустой знаменатель дает null. "
    "Ошибка схемы и детерминированный отказ генератора — model_error. "
    "Ошибка сети — api_error. Оба входят в generation_error_rate и не "
    "входят в оценки судьи. "
    "Судья вызывается только после успешной генерации и другой моделью, "
    "если конфигурация ее задает. Его отказ — judge_error без фиктивной "
    "оценки; генерация при этом остается валидной. "
    "semantic_issue_rate считает только согласованные разборы судьи с "
    "подтвержденными цитатами из карточки и critical_errors. "
    "Выдуманные цитаты и внутренние противоречия verdict/evidence не "
    "повышают semantic_issue_rate: такой ответ идет в needs_manual_review "
    "или judge_error. "
    "Оценка судьи не является истиной. "
    "Температура фиксируется в отчете и не означает детерминизм. "
    "Стоимость считается по usage ответа и тарифу модели. Без usage или "
    "тарифа у любого учтенного вызова агрегированный cost равен null, "
    "а не частичной сумме."
)

_SAFE_MODEL = re.compile(r"[^\w.\-]+", re.UNICODE)


class GenerationRunStatus(StrEnum):
    GENERATED = "generated"
    INSUFFICIENT_DATA = "insufficient_data"
    API_ERROR = "api_error"
    MODEL_ERROR = "model_error"
    JUDGE_ERROR = "judge_error"


class JudgeRunStatus(StrEnum):
    SCORED = "scored"
    SKIPPED = "skipped"
    ERROR = "error"


class DraftChecks(BaseModel):
    schema_valid: bool
    title_length_ok: bool | None = None
    description_length_ok: bool | None = None
    search_queries_ok: bool | None = None


class GenerationScenarioReport(BaseModel):
    case_id: str
    description: str
    status: GenerationRunStatus
    eligible: bool
    error_message: str | None = None
    judge_status: JudgeRunStatus
    judge_error_message: str | None = None
    model_called: bool = False
    latency_seconds: float | None = None
    response_model: str | None = None
    usage: LLMUsage | None = None
    cost: UsageCost | None = None
    draft_checks: DraftChecks
    generation: dict[str, Any] | None = None
    raw_response: Any = None
    judge: JudgeAssessment | None = None
    judge_usage: LLMUsage | None = None
    judge_cost: UsageCost | None = None
    confirmed_semantic_issue: bool = False
    needs_manual_review: bool = False
    manual_review_reason: str = ""


class GenerationEvaluationReport(BaseModel):
    created_at: datetime
    model: str
    judge_model: str
    judge_independent: bool
    base_url: str | None = None
    temperature: float | None = None
    determinism_note: str
    prompt_hash: str
    judge_prompt_hash: str
    dataset_hash: str
    scenario_count: int
    eligible_count: int
    methodology: str
    rates: GenerationRates
    mean_factual_consistency: float | None
    mean_seo_quality: float | None
    mean_language_quality: float | None
    judged_count: int
    mean_latency_seconds: float | None
    generation_cost: UsageCost
    judge_cost: UsageCost
    scenarios: list[GenerationScenarioReport]


class GenerationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    dataset_hash: str
    prompt_hash: str
    judge_prompt_hash: str
    judge_model: str
    temperature: float | None
    models: list[str] = Field(default_factory=list)
    reports: list[GenerationEvaluationReport]


def default_report_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "reports" / "evals" / "generation"


def generation_prompt_hash() -> str:
    source = inspect.getsource(build_generation_prompt)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def judge_prompt_hash() -> str:
    source = inspect.getsource(build_judge_prompt)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def dataset_hash(cases: Sequence[GenerationDatasetCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def run_generation_evaluation(
    cases: Sequence[GenerationDatasetCase],
    generator_client: LLMClient,
    judge_client: LLMClient,
    *,
    model: str,
    judge_model: str,
    temperature: float | None = None,
    prices: dict[str, ModelPrice] | None = None,
    base_url: str | None = None,
    report_dir: Path | None = None,
    created_at: datetime | None = None,
) -> GenerationEvaluationReport:
    """Прогнать один датасет одной моделью. Судья получает только успешные карточки."""

    tariff = (prices or {}).get(model)
    judge_tariff = (prices or {}).get(judge_model)
    scenarios: list[GenerationScenarioReport] = []
    for case in cases:
        scenarios.append(
            await _run_case(
                case,
                generator_client,
                judge_client,
                generator_price=tariff,
                judge_price=judge_tariff,
            )
        )
    report = build_generation_report(
        scenarios,
        cases,
        model=model,
        judge_model=judge_model,
        temperature=temperature,
        base_url=base_url,
        created_at=created_at,
    )
    if report_dir is not None:
        write_generation_report(report, report_dir)
    return report


def build_generation_report(
    scenarios: Sequence[GenerationScenarioReport],
    cases: Sequence[GenerationDatasetCase],
    *,
    model: str,
    judge_model: str,
    temperature: float | None = None,
    base_url: str | None = None,
    created_at: datetime | None = None,
) -> GenerationEvaluationReport:
    created = created_at or datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    eligible = [item for item in scenarios if item.eligible]
    schema_pool = [
        item for item in eligible if item.status is not GenerationRunStatus.API_ERROR
    ]
    parsed = [item for item in schema_pool if item.draft_checks.schema_valid]
    generated = [
        item
        for item in scenarios
        if item.status in {GenerationRunStatus.GENERATED, GenerationRunStatus.JUDGE_ERROR}
    ]
    judged = [
        item
        for item in scenarios
        if item.judge is not None and item.judge_status is JudgeRunStatus.SCORED
    ]
    semantic = [item for item in judged if item.confirmed_semantic_issue]
    latencies = [
        item.latency_seconds
        for item in scenarios
        if item.latency_seconds is not None
        and (item.model_called or item.status is GenerationRunStatus.API_ERROR)
    ]
    return GenerationEvaluationReport(
        created_at=created,
        model=model,
        judge_model=judge_model,
        judge_independent=judge_model.strip() != model.strip(),
        base_url=base_url or None,
        temperature=temperature,
        determinism_note=(
            "Температура передается в API, если задана. "
            "Повторный прогон может отличаться."
        ),
        prompt_hash=generation_prompt_hash(),
        judge_prompt_hash=judge_prompt_hash(),
        dataset_hash=dataset_hash(cases),
        scenario_count=len(scenarios),
        eligible_count=len(eligible),
        methodology=GENERATION_EVAL_METHOD,
        rates=GenerationRates(
            valid_generation_rate=count_rate(len(generated), len(eligible)),
            schema_compliance_rate=count_rate(
                sum(item.draft_checks.schema_valid for item in schema_pool),
                len(schema_pool),
            ),
            seo_title_length_compliance=_flag_rate(
                parsed,
                lambda item: item.draft_checks.title_length_ok,
            ),
            seo_description_length_compliance=_flag_rate(
                parsed,
                lambda item: item.draft_checks.description_length_ok,
            ),
            search_queries_count_compliance=_flag_rate(
                parsed,
                lambda item: item.draft_checks.search_queries_ok,
            ),
            generation_error_rate=count_rate(
                sum(
                    item.status
                    in {GenerationRunStatus.API_ERROR, GenerationRunStatus.MODEL_ERROR}
                    for item in eligible
                ),
                len(eligible),
            ),
            api_error_rate=count_rate(
                sum(item.status is GenerationRunStatus.API_ERROR for item in eligible),
                len(eligible),
            ),
            model_error_rate=count_rate(
                sum(item.status is GenerationRunStatus.MODEL_ERROR for item in eligible),
                len(eligible),
            ),
            insufficient_data_rate=count_rate(
                sum(
                    item.status is GenerationRunStatus.INSUFFICIENT_DATA
                    for item in scenarios
                ),
                len(scenarios),
            ),
            judge_error_rate=count_rate(
                sum(item.status is GenerationRunStatus.JUDGE_ERROR for item in generated),
                len(generated),
            ),
            semantic_issue_rate=count_rate(len(semantic), len(judged)),
        ),
        mean_factual_consistency=mean(
            [float(item.judge.factual_consistency_score) for item in judged if item.judge]
        ),
        mean_seo_quality=mean(
            [float(item.judge.seo_quality_score) for item in judged if item.judge]
        ),
        mean_language_quality=mean(
            [float(item.judge.language_quality_score) for item in judged if item.judge]
        ),
        judged_count=len(judged),
        mean_latency_seconds=mean(latencies),
        generation_cost=aggregate_costs(
            [(item.model_called, item.cost) for item in scenarios]
        ),
        judge_cost=aggregate_costs(
            [
                (item.judge_status is not JudgeRunStatus.SKIPPED, item.judge_cost)
                for item in scenarios
            ]
        ),
        scenarios=list(scenarios),
    )


def write_generation_report(report: GenerationEvaluationReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.strftime("%Y%m%dT%H%M%S%fZ")
    safe_model = _SAFE_MODEL.sub("_", report.model).strip("_") or "model"
    path = directory / f"{stamp}_{safe_model}.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_comparison(comparison: GenerationComparison, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = comparison.created_at.strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stamp}_comparison.json"
    path.write_text(
        json.dumps(comparison.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_prices(path: Path) -> dict[str, ModelPrice]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {model: ModelPrice.model_validate(price) for model, price in payload.items()}


class _RecordingClient:
    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.responses: list[str] = []
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
        self.responses.append(response)
        meta = getattr(self._inner, "last_call", None)
        self.metas.append(meta if isinstance(meta, LLMCallMeta) else None)
        return response


async def _run_case(
    case: GenerationDatasetCase,
    generator_client: LLMClient,
    judge_client: LLMClient,
    *,
    generator_price: ModelPrice | None,
    judge_price: ModelPrice | None,
) -> GenerationScenarioReport:
    eligible = assess_generation_status(case.input) is GateStatus.COMPLETE
    recorder = _RecordingClient(generator_client)
    generator = SEOGenerator(recorder)
    started = time.perf_counter()
    try:
        result = await generator.generate(case.input)
    except LLMError as exc:
        latency = time.perf_counter() - started
        raw = recorder.responses[-1] if recorder.responses else None
        meta = recorder.metas[-1] if recorder.metas else None
        status = (
            GenerationRunStatus.API_ERROR
            if exc.kind is LLMErrorKind.API
            else GenerationRunStatus.MODEL_ERROR
        )
        return _scenario(
            case,
            status=status,
            eligible=eligible,
            error_message=str(exc),
            judge_status=JudgeRunStatus.SKIPPED,
            model_called=recorder.attempted or bool(recorder.responses),
            latency=latency if recorder.attempted or recorder.responses else None,
            meta=meta,
            price=generator_price,
            checks=assess_draft(raw),
            raw_response=_payload(raw),
        )
    latency = time.perf_counter() - started
    raw = recorder.responses[-1] if recorder.responses else None
    meta = recorder.metas[-1] if recorder.metas else None
    if result.status is GateStatus.INSUFFICIENT_DATA:
        return _scenario(
            case,
            status=GenerationRunStatus.INSUFFICIENT_DATA,
            eligible=False,
            error_message=None,
            judge_status=JudgeRunStatus.SKIPPED,
            model_called=False,
            latency=None,
            meta=None,
            price=None,
            checks=DraftChecks(schema_valid=False),
            generation=result.model_dump(mode="json"),
        )
    judgment = await _judge(case, result, judge_client)
    status = (
        GenerationRunStatus.JUDGE_ERROR
        if judgment.status is JudgeRunStatus.ERROR
        else GenerationRunStatus.GENERATED
    )
    return _scenario(
        case,
        status=status,
        eligible=eligible,
        error_message=None,
        judge_status=judgment.status,
        judge_error_message=judgment.error_message,
        model_called=recorder.attempted or bool(recorder.responses),
        latency=latency,
        meta=meta,
        price=generator_price,
        checks=assess_draft(raw),
        generation=result.model_dump(mode="json"),
        raw_response=_payload(raw),
        judge=judgment.assessment,
        judge_meta=judgment.meta,
        judge_price=judge_price,
        judge_check=judgment.check,
    )


class _Judgment:
    def __init__(
        self,
        status: JudgeRunStatus,
        assessment: JudgeAssessment | None = None,
        error_message: str | None = None,
        meta: LLMCallMeta | None = None,
        check: JudgeCheckResult | None = None,
    ) -> None:
        self.status = status
        self.assessment = assessment
        self.error_message = error_message
        self.meta = meta
        self.check = check


async def _judge(
    case: GenerationDatasetCase,
    card: GenerationResult,
    judge_client: LLMClient,
) -> _Judgment:
    recorder = _RecordingClient(judge_client)
    try:
        raw = await recorder.complete(
            build_judge_prompt(case, card),
            response_model=JudgeAssessment,
        )
        assessment = JudgeAssessment.model_validate(json.loads(raw))
    except (LLMError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        meta = recorder.metas[-1] if recorder.metas else None
        return _Judgment(JudgeRunStatus.ERROR, error_message=str(exc), meta=meta)
    meta = recorder.metas[-1] if recorder.metas else None
    checked = check_judge_assessment(assessment, card)
    if checked.judge_error:
        return _Judgment(
            JudgeRunStatus.ERROR,
            assessment=assessment,
            error_message=checked.review_reason or "Некорректный ответ судьи",
            meta=meta,
            check=checked,
        )
    return _Judgment(
        JudgeRunStatus.SCORED,
        assessment=assessment,
        meta=meta,
        check=checked,
    )


def assess_draft(raw: str | None) -> DraftChecks:
    if not raw:
        return DraftChecks(schema_valid=False)
    try:
        draft = GenerationDraft.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return DraftChecks(schema_valid=False)
    title = draft.seo_title.strip()
    queries = [query.strip() for query in draft.search_queries]
    unique = {query.casefold() for query in queries}
    return DraftChecks(
        schema_valid=True,
        title_length_ok=bool(title) and len(draft.seo_title) <= SEO_TITLE_MAX_LENGTH,
        description_length_ok=(
            SEO_DESCRIPTION_MIN_LENGTH
            <= len(draft.seo_description)
            <= SEO_DESCRIPTION_MAX_LENGTH
        ),
        search_queries_ok=(
            SEARCH_QUERIES_MIN <= len(queries) <= SEARCH_QUERIES_MAX
            and all(queries)
            and len(unique) == len(queries)
        ),
    )


def _scenario(
    case: GenerationDatasetCase,
    *,
    status: GenerationRunStatus,
    eligible: bool,
    error_message: str | None,
    judge_status: JudgeRunStatus,
    model_called: bool,
    latency: float | None,
    meta: LLMCallMeta | None,
    price: ModelPrice | None,
    checks: DraftChecks,
    judge_error_message: str | None = None,
    generation: dict[str, Any] | None = None,
    raw_response: Any = None,
    judge: JudgeAssessment | None = None,
    judge_meta: LLMCallMeta | None = None,
    judge_price: ModelPrice | None = None,
    judge_check: JudgeCheckResult | None = None,
) -> GenerationScenarioReport:
    usage = meta.usage if meta is not None else None
    judge_usage = judge_meta.usage if judge_meta is not None else None
    review = bool(judge_check.needs_manual_review) if judge_check is not None else False
    reason = judge_check.review_reason if judge_check is not None else ""
    confirmed = (
        bool(judge_check.confirmed_semantic_issue) if judge_check is not None else False
    )
    return GenerationScenarioReport(
        case_id=case.id,
        description=case.description,
        status=status,
        eligible=eligible,
        error_message=error_message,
        judge_status=judge_status,
        judge_error_message=judge_error_message,
        model_called=model_called,
        latency_seconds=latency,
        response_model=meta.response_model if meta is not None else None,
        usage=usage,
        cost=usage_cost([usage], price) if usage is not None else None,
        draft_checks=checks,
        generation=generation,
        raw_response=raw_response,
        judge=judge,
        judge_usage=judge_usage,
        judge_cost=(
            usage_cost([judge_usage], judge_price) if judge_usage is not None else None
        ),
        confirmed_semantic_issue=confirmed,
        needs_manual_review=review,
        manual_review_reason=reason,
    )


def _payload(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _flag_rate(
    scenarios: Sequence[GenerationScenarioReport],
    flag: Any,
) -> CountRate:
    known = [item for item in scenarios if flag(item) is not None]
    passed = sum(bool(flag(item)) for item in known)
    return count_rate(passed, len(known))
