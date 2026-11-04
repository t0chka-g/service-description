"""Оценка генерации на fake-клиентах. Сеть не используется."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.clients.llm import LLMCallMeta, LLMError, LLMErrorKind, LLMUsage
from app.evals.generation_cli import build_parser, run_cli, selected_models
from app.evals.generation_dataset import load_generation_dataset
from app.evals.generation_metrics import ModelPrice, aggregate_costs, count_rate, usage_cost
from app.evals.generation_runner import (
    GenerationRunStatus,
    JudgeRunStatus,
    assess_draft,
    run_generation_evaluation,
    write_generation_report,
)
from app.evals.judge import ClaimKind, JudgeAssessment, JudgeVerdict
from app.evals.judge_calibration import (
    CalibrationRunStatus,
    ViolationClass,
    as_prompt_case,
    compare_calibration,
    expected_claim_kind,
    expects_semantic_issue,
    judge_schema_hash,
    load_judge_calibration_dataset,
    run_judge_calibration,
    run_judge_calibration_report,
    select_calibration_cases,
    write_calibration_report,
)
from app.evals.judge_calibration_cli import (
    build_parser as build_calibration_parser,
)
from app.evals.judge_calibration_cli import (
    run_cli as run_calibration_cli,
)
from app.evals.judge_calibration_cli import (
    selected_case_ids,
    selected_judge,
)
from app.evals.judge_checks import check_judge_assessment, quote_in_card
from app.prompts.judge import build_judge_prompt
from app.schemas.product import GenerationResult, GenerationStatus
from tests.texts import NEUTRAL_SENTENCES, with_length

EVIDENCE_QUOTE = NEUTRAL_SENTENCES[0].rstrip(".")
# Похвала без числа и без слов из детерминированного списка: ее ловит судья, не генератор.
CLAIM_QUOTE = "оптимальное соотношение"
CLAIM_SENTENCE = "Здесь оптимальное соотношение возможностей для задач покупателя."

DATASET_IDS = [
    "full_specs",
    "insufficient_inputs",
    "contradictory_specs",
    "complex_notation",
    "numeric_ranges",
    "similar_parameters",
    "missing_purpose",
    "commercial_claims",
    "prompt_injection",
    "non_numeric_specs",
]

CALIBRATION_IDS = [
    "cal_fully_supported",
    "cal_unit_substitution",
    "cal_fabricated_certificate",
    "cal_unsupported_purpose",
    "cal_unsupported_commercial",
    "cal_acceptable_generalization",
    "cal_spec_contradiction",
    "cal_keyword_stuffing",
]


class ScriptedClient:
    def __init__(
        self,
        response: str | Exception | None = None,
        *,
        error: Exception | None = None,
        usage: LLMUsage | None = None,
        steps: list[str | Exception] | None = None,
    ) -> None:
        self._response = "" if isinstance(response, Exception) else (response or "")
        self._error = error or (response if isinstance(response, Exception) else None)
        self._usage = usage
        self._steps = list(steps or [])
        self.calls: list[str] = []
        self.last_call: LLMCallMeta | None = None

    def _set_meta(self) -> None:
        if self._usage is not None:
            self.last_call = LLMCallMeta(response_model="fake-model", usage=self._usage)

    async def complete(self, prompt: str, *, response_model: object = None) -> str:
        del response_model
        self.calls.append(prompt)
        if self._steps:
            step = self._steps.pop(0)
            self._set_meta()
            if isinstance(step, Exception):
                raise step
            return step
        if self._error is not None:
            self._set_meta()
            raise self._error
        self._set_meta()
        return self._response


def _case(case_id: str):
    return next(case for case in load_generation_dataset() if case.id == case_id)


def _pad(seed: str) -> str:
    return with_length(seed, 900)


def _valid_draft(case_id: str = "full_specs", *, extra: str = "") -> str:
    case = _case(case_id)
    head = case.input.name.split()[0]
    manufacturer = case.input.manufacturer
    facts = "; ".join(f"{fact.name} {fact.value}" for fact in case.confirmed_facts)
    description = _pad(
        f"{case.input.name}. Производитель {manufacturer}. {facts}. {extra}".strip()
    )
    return json.dumps(
        {
            "status": "complete",
            "seo_title": f"{head} {manufacturer}"[:70],
            "seo_description": description,
            "search_queries": [
                f"{head} {manufacturer}",
                f"{manufacturer} {head}",
                f"{head} характеристики {manufacturer}",
            ],
            "commercial_block": "Подробности уточняйте у поставщика.",
        },
        ensure_ascii=False,
    )


def _judge_body(
    *,
    verdict: str = "pass",
    score: int = 5,
    claims: list[dict] | None = None,
    critical_errors: list[str] | None = None,
    evidence: list[dict] | None = None,
    seo_score: int | None = None,
    quote: str = "подтвержденные сведения",
) -> str:
    return json.dumps(
        {
            "factual_consistency_score": score,
            "seo_quality_score": score if seo_score is None else seo_score,
            "language_quality_score": score,
            "unsupported_claims": claims or [],
            "critical_errors": critical_errors or [],
            "evidence": evidence
            or [
                {
                    "quote": quote,
                    "kind": "confirmed_fact",
                    "explanation": "Фраза не добавляет новых чисел.",
                }
            ],
            "reasoning_summary": "Проверка fake-судьи.",
            "verdict": verdict,
        },
        ensure_ascii=False,
    )


def _generation_judge_body(**kwargs: object) -> str:
    return _judge_body(quote=EVIDENCE_QUOTE, **kwargs)  # type: ignore[arg-type]


def test_dataset_covers_required_scenarios() -> None:
    cases = load_generation_dataset()
    assert [case.id for case in cases] == DATASET_IDS
    for case in cases:
        assert case.confirmed_facts
        assert case.allowed_generalizations
        if case.id != "full_specs":
            assert case.missing_data or case.conflicting_values or case.forbidden_claims


def test_count_rate_has_no_zero_division() -> None:
    empty = count_rate(0, 0)
    assert empty.value is None
    assert empty.denominator == 0


async def test_metrics_split_statuses(tmp_path: Path) -> None:
    price = ModelPrice(prompt_per_million=1.0, completion_per_million=2.0)
    usage = LLMUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
    cases = [
        _case("full_specs"),
        _case("insufficient_inputs"),
        _case("contradictory_specs"),
    ]
    generator = ScriptedClient(
        usage=usage,
        steps=[_valid_draft(), LLMError("schema", kind=LLMErrorKind.MODEL)],
    )
    judge = ScriptedClient(
        _generation_judge_body(score=4),
        usage=LLMUsage(prompt_tokens=10, completion_tokens=10),
    )

    report = await run_generation_evaluation(
        cases,
        generator,
        judge,
        model="generator-a",
        judge_model="judge-b",
        temperature=0.0,
        prices={"generator-a": price, "judge-b": price},
        report_dir=tmp_path,
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
    )

    by_id = {item.case_id: item for item in report.scenarios}
    assert by_id["full_specs"].status is GenerationRunStatus.GENERATED
    assert by_id["full_specs"].judge_status is JudgeRunStatus.SCORED
    assert by_id["insufficient_inputs"].status is GenerationRunStatus.INSUFFICIENT_DATA
    assert by_id["insufficient_inputs"].judge_status is JudgeRunStatus.SKIPPED
    assert by_id["contradictory_specs"].status is GenerationRunStatus.MODEL_ERROR
    assert by_id["contradictory_specs"].judge_status is JudgeRunStatus.SKIPPED
    assert by_id["contradictory_specs"].usage is not None
    assert len(judge.calls) == 1
    assert report.eligible_count == 2
    assert report.rates.valid_generation_rate.value == 0.5
    assert report.rates.insufficient_data_rate.value == pytest.approx(1 / 3)
    assert report.rates.model_error_rate.value == 0.5
    assert report.rates.generation_error_rate.value == 0.5
    assert report.rates.api_error_rate.value == 0.0
    assert report.mean_factual_consistency == 4.0
    assert report.judged_count == 1
    assert report.generation_cost.priced is True
    assert report.generation_cost.cost == pytest.approx(4.0)
    assert report.judge_independent is True
    assert report.judge_cost.cost == pytest.approx(30 / 1_000_000)
    written = list(tmp_path.glob("*.json"))
    assert written
    restored = json.loads(written[0].read_text(encoding="utf-8"))
    assert restored["scenarios"][0]["status"] == "generated"
    assert "judge" in restored["scenarios"][0]


async def test_api_error_keeps_latency() -> None:
    case = _case("full_specs")
    generator = ScriptedClient(error=LLMError("down", kind=LLMErrorKind.API))
    judge = ScriptedClient(_generation_judge_body())
    report = await run_generation_evaluation(
        [case],
        generator,
        judge,
        model="generator-a",
        judge_model="judge-b",
    )
    scenario = report.scenarios[0]
    assert scenario.status is GenerationRunStatus.API_ERROR
    assert scenario.model_called is True
    assert scenario.latency_seconds is not None
    assert scenario.latency_seconds >= 0
    assert report.mean_latency_seconds is not None
    assert scenario.judge_status is JudgeRunStatus.SKIPPED
    assert judge.calls == []
    assert report.rates.api_error_rate.value == 1.0
    assert report.rates.schema_compliance_rate.value is None
    assert report.rates.valid_generation_rate.value == 0.0


async def test_refusal_preserves_available_usage() -> None:
    usage = LLMUsage(prompt_tokens=12, completion_tokens=3)
    generator = ScriptedClient(
        error=LLMError("model refused the extraction request"),
        usage=usage,
    )
    report = await run_generation_evaluation(
        [_case("full_specs")],
        generator,
        ScriptedClient(_generation_judge_body()),
        model="generator-a",
        judge_model="judge-b",
        prices={"generator-a": ModelPrice(prompt_per_million=1.0, completion_per_million=2.0)},
    )
    scenario = report.scenarios[0]
    assert scenario.status is GenerationRunStatus.MODEL_ERROR
    assert scenario.usage == usage
    assert scenario.cost is not None
    assert scenario.cost.priced is True


async def test_judge_error_keeps_generation_valid() -> None:
    generator = ScriptedClient(_valid_draft("non_numeric_specs"))
    judge = ScriptedClient('{"factual_consistency_score": 5}')
    report = await run_generation_evaluation(
        [_case("non_numeric_specs")],
        generator,
        judge,
        model="generator-a",
        judge_model="judge-b",
    )
    scenario = report.scenarios[0]
    assert scenario.status is GenerationRunStatus.JUDGE_ERROR
    assert scenario.judge is None
    assert scenario.judge_error_message
    assert report.rates.valid_generation_rate.value == 1.0
    assert report.rates.judge_error_rate.value == 1.0
    assert report.mean_factual_consistency is None
    assert report.rates.semantic_issue_rate.value is None


async def test_only_insufficient_cases_do_not_divide_by_zero() -> None:
    report = await run_generation_evaluation(
        [_case("insufficient_inputs")],
        ScriptedClient("unused"),
        ScriptedClient("unused"),
        model="generator-a",
        judge_model="judge-b",
    )
    assert report.eligible_count == 0
    assert report.rates.valid_generation_rate.value is None
    assert report.rates.generation_error_rate.value is None
    assert report.rates.insufficient_data_rate.value == 1.0
    assert report.mean_latency_seconds is None
    assert report.generation_cost.cost is None


async def test_cost_without_price_stays_empty() -> None:
    usage = LLMUsage(prompt_tokens=100, completion_tokens=40)
    report = await run_generation_evaluation(
        [_case("non_numeric_specs")],
        ScriptedClient(_valid_draft("non_numeric_specs"), usage=usage),
        ScriptedClient(_generation_judge_body()),
        model="generator-a",
        judge_model="judge-b",
    )
    assert report.scenarios[0].usage is not None
    assert report.scenarios[0].cost is not None
    assert report.scenarios[0].cost.priced is False
    assert report.generation_cost.cost is None
    assert report.generation_cost.prompt_tokens == 100


def test_usage_cost_requires_complete_usage() -> None:
    price = ModelPrice(prompt_per_million=1.0, completion_per_million=2.0)
    partial = usage_cost([None], price)
    assert partial.priced is False
    assert partial.cost is None
    missing_price = usage_cost([LLMUsage(prompt_tokens=1, completion_tokens=1)], None)
    assert missing_price.priced is False
    assert missing_price.cost is None


def test_aggregate_cost_is_null_if_any_call_unpriced() -> None:
    priced = usage_cost(
        [LLMUsage(prompt_tokens=1_000_000, completion_tokens=0)],
        ModelPrice(prompt_per_million=1.0, completion_per_million=2.0),
    )
    total = aggregate_costs([(True, priced), (True, None)])
    assert total.priced is False
    assert total.cost is None
    assert total.prompt_tokens == 1_000_000


def test_draft_checks_separate_length_from_schema() -> None:
    raw = json.dumps(
        {
            "status": "complete",
            "seo_title": "А" * 71,
            "seo_description": "Б" * 800,
            "search_queries": ["один", "два"],
            "commercial_block": "блок",
        },
        ensure_ascii=False,
    )
    checks = assess_draft(raw)
    assert checks.schema_valid is True
    assert checks.title_length_ok is False
    assert checks.description_length_ok is True
    assert checks.search_queries_ok is False
    assert assess_draft("not-json").schema_valid is False


async def test_fabricated_quote_is_judge_error_not_semantic_issue() -> None:
    claim = {
        "quote": CLAIM_QUOTE,
        "explanation": "Цены во входе нет.",
        "kind": "unsupported_claim",
    }
    report = await run_generation_evaluation(
        [_case("commercial_claims")],
        ScriptedClient(_valid_draft("commercial_claims")),
        ScriptedClient(_generation_judge_body(verdict="fail", score=1, claims=[claim])),
        model="generator-a",
        judge_model="judge-b",
    )
    scenario = report.scenarios[0]
    assert scenario.status is GenerationRunStatus.JUDGE_ERROR
    assert scenario.needs_manual_review is True
    assert scenario.confirmed_semantic_issue is False
    assert scenario.judge is not None
    assert scenario.judge.unsupported_claims[0].kind is ClaimKind.UNSUPPORTED_CLAIM
    assert report.rates.semantic_issue_rate.value is None
    assert report.rates.judge_error_rate.value == 1.0


async def test_verdict_pass_with_real_claim_needs_review_without_semantic_inflation() -> None:
    draft = _valid_draft("commercial_claims", extra=CLAIM_SENTENCE)
    claim = {
        "quote": CLAIM_QUOTE,
        "explanation": "Цены во входе нет.",
        "kind": "unsupported_claim",
    }
    report = await run_generation_evaluation(
        [_case("commercial_claims")],
        ScriptedClient(draft),
        ScriptedClient(_generation_judge_body(verdict="pass", score=4, claims=[claim])),
        model="generator-a",
        judge_model="judge-b",
    )
    scenario = report.scenarios[0]
    assert scenario.judge_status is JudgeRunStatus.SCORED
    assert scenario.needs_manual_review is True
    assert scenario.confirmed_semantic_issue is False
    assert report.rates.semantic_issue_rate.value == 0.0


async def test_consistent_fail_with_real_quote_counts_semantic_issue() -> None:
    draft = _valid_draft("commercial_claims", extra=CLAIM_SENTENCE)
    claim = {
        "quote": CLAIM_QUOTE,
        "explanation": "Цены во входе нет.",
        "kind": "unsupported_claim",
    }
    report = await run_generation_evaluation(
        [_case("commercial_claims")],
        ScriptedClient(draft),
        ScriptedClient(_generation_judge_body(verdict="fail", score=1, claims=[claim])),
        model="generator-a",
        judge_model="judge-b",
    )
    scenario = report.scenarios[0]
    assert scenario.judge_status is JudgeRunStatus.SCORED
    assert scenario.confirmed_semantic_issue is True
    assert report.rates.semantic_issue_rate.value == 1.0


async def test_fail_without_confirmed_violation_is_inconsistent() -> None:
    report = await run_generation_evaluation(
        [_case("full_specs")],
        ScriptedClient(_valid_draft("full_specs")),
        ScriptedClient(_generation_judge_body(verdict="fail", score=1)),
        model="generator-a",
        judge_model="judge-b",
    )
    scenario = report.scenarios[0]
    assert scenario.needs_manual_review is True
    assert scenario.confirmed_semantic_issue is False
    assert report.rates.semantic_issue_rate.value == 0.0


def test_judge_prompt_excludes_labeling_keys() -> None:
    case = _case("contradictory_specs")
    card = GenerationResult(
        status=GenerationStatus.INSUFFICIENT_DATA,
        seo_title="Манометр",
        seo_description="Короткое описание без новых чисел.",
        commercial_block="Условия не указаны.",
    )
    prompt = build_judge_prompt(case, card)
    assert "0...1,6 МПа" in prompt
    assert "0...2,5 МПа" in prompt
    assert case.id not in prompt
    assert "Запрещенные утверждения" not in prompt
    assert "Допустимые обобщения" not in prompt
    assert "рабочее давление 2,5 МПа" not in prompt
    assert case.allowed_generalizations not in prompt
    assert "нельзя утверждать" not in prompt.casefold()
    assert "warnings" not in prompt
    assert "Полное описание не собрано" not in prompt
    assert "verdict" in prompt


def test_quote_check_rejects_missing_fragment() -> None:
    card = GenerationResult(
        status=GenerationStatus.INSUFFICIENT_DATA,
        seo_title="Манометр",
        seo_description="Короткое описание без новых чисел.",
        commercial_block="Условия не указаны.",
    )
    assessment = JudgeAssessment.model_validate(
        json.loads(
            _judge_body(
                verdict="fail",
                score=1,
                claims=[
                    {
                        "quote": "сертификат ГОСТ",
                        "explanation": "Сертификата нет.",
                        "kind": "unsupported_claim",
                    }
                ],
            )
        )
    )
    assert quote_in_card("Короткое описание", card) is True
    assert quote_in_card("сертификат ГОСТ", card) is False
    checked = check_judge_assessment(assessment, card)
    assert checked.fabricated_quotes
    assert checked.judge_error is True
    assert checked.confirmed_semantic_issue is False


def test_calibration_dataset_covers_required_scenarios() -> None:
    cases = load_judge_calibration_dataset()
    assert [case.id for case in cases] == CALIBRATION_IDS
    classes = {case.expected_violation_class for case in cases}
    assert classes == {
        ViolationClass.NONE,
        ViolationClass.UNIT_SUBSTITUTION,
        ViolationClass.FABRICATED_CERTIFICATE,
        ViolationClass.UNSUPPORTED_PURPOSE,
        ViolationClass.UNSUPPORTED_COMMERCIAL_ADVANTAGE,
        ViolationClass.ACCEPTABLE_GENERALIZATION,
        ViolationClass.SPEC_CONTRADICTION,
        ViolationClass.KEYWORD_STUFFING,
    }
    for case in cases:
        assert case.allowed_verdicts
        prompt_case = as_prompt_case(case)
        assert prompt_case.id == "calibration_case"
        assert prompt_case.forbidden_claims == []
        assert prompt_case.allowed_generalizations == ""
        prompt = build_judge_prompt(prompt_case, case.card)
        assert case.id not in prompt
        assert case.description not in prompt
        assert "expected_violation_class" not in prompt
        assert "allowed_verdicts" not in prompt
        assert f"класс нарушения: {case.expected_violation_class.value}" not in prompt
        assert f"expected_violation_class={case.expected_violation_class.value}" not in prompt
        for verdict in case.allowed_verdicts:
            assert f"ожидаемый verdict: {verdict}" not in prompt
            assert f"ожидаемый verdict: {verdict.value}" not in prompt


def test_calibration_compare_matches_ground_truth_offline() -> None:
    case = next(
        item
        for item in load_judge_calibration_dataset()
        if item.id == "cal_unit_substitution"
    )
    assessment = JudgeAssessment.model_validate(
        json.loads(
            _judge_body(
                verdict="fail",
                score=1,
                claims=[
                    {
                        "quote": "0...1,0 bar",
                        "explanation": "Единица bar не подтверждена.",
                        "kind": "direct_contradiction",
                    }
                ],
                evidence=[
                    {
                        "quote": "0...1,0 bar",
                        "kind": "direct_contradiction",
                        "explanation": "Подмена единицы.",
                    }
                ],
            )
        )
    )
    result = compare_calibration(case, assessment)
    assert result.expected_claim_kind is ClaimKind.DIRECT_CONTRADICTION
    assert result.verdict_match is True
    assert result.claim_classification_match is True
    assert result.issue_presence_match is True
    assert result.matched is True


def test_expected_claim_kind_mapping() -> None:
    assert expected_claim_kind(ViolationClass.NONE) is None
    assert expected_claim_kind(ViolationClass.ACCEPTABLE_GENERALIZATION) is None
    assert expected_claim_kind(ViolationClass.KEYWORD_STUFFING) is None
    assert expects_semantic_issue(ViolationClass.KEYWORD_STUFFING) is False
    assert expects_semantic_issue(ViolationClass.UNIT_SUBSTITUTION) is True
    assert (
        expected_claim_kind(ViolationClass.UNIT_SUBSTITUTION)
        is ClaimKind.DIRECT_CONTRADICTION
    )
    assert (
        expected_claim_kind(ViolationClass.FABRICATED_CERTIFICATE)
        is ClaimKind.UNSUPPORTED_CLAIM
    )
    for violation in ViolationClass:
        expected_claim_kind(violation)


def test_judge_prompt_separates_seo_from_verdict() -> None:
    case = next(
        item
        for item in load_judge_calibration_dataset()
        if item.id == "cal_keyword_stuffing"
    )
    prompt = build_judge_prompt(as_prompt_case(case), case.card)
    assert "(×12)" in prompt
    assert "(повторено N раз)" in prompt
    assert "непрерывная подстрока текста карточки" in prompt
    assert "слабое SEO этому не мешает" in prompt
    assert "не переводят verdict в review или fail" in prompt
    assert "замечание только про SEO или стиль" in prompt
    assert "keyword stuffing" in prompt.casefold()
    assert case.id not in prompt
    assert "keyword_stuffing" not in prompt


def test_abbreviated_repetition_quote_stays_invalid() -> None:
    description = "Манометр ИзмерТех МП-У-01. " * 4
    card = GenerationResult(
        status=GenerationStatus.INSUFFICIENT_DATA,
        seo_title="Манометр",
        seo_description=description.strip(),
        commercial_block="Условия не указаны.",
    )
    exact = "Манометр ИзмерТех МП-У-01."
    assert quote_in_card(exact, card) is True
    abbreviated = (
        "Манометр ИзмерТех МП-У-01. (×12)",
        "Манометр ИзмерТех МП-У-01. (x4)",
        "Манометр ИзмерТех МП-У-01. (повторено 4 раз)",
    )
    for quote in abbreviated:
        assert quote_in_card(quote, card) is False
        assessment = JudgeAssessment.model_validate(
            json.loads(
                _judge_body(
                    verdict="review",
                    score=5,
                    claims=[
                        {
                            "quote": quote,
                            "explanation": "Повтор не является новым фактом.",
                            "kind": "unsupported_claim",
                        }
                    ],
                    evidence=[
                        {
                            "quote": quote,
                            "kind": "unsupported_claim",
                            "explanation": "Сокращенная ссылка на повтор.",
                        }
                    ],
                )
            )
        )
        checked = check_judge_assessment(assessment, card)
        assert checked.judge_error is True
        assert checked.confirmed_semantic_issue is False
        assert quote in checked.fabricated_quotes

    exact_claim = JudgeAssessment.model_validate(
        json.loads(
            _judge_body(
                verdict="fail",
                score=2,
                claims=[
                    {
                        "quote": exact,
                        "explanation": "Фраза не следует из входа.",
                        "kind": "unsupported_claim",
                    }
                ],
                evidence=[
                    {
                        "quote": exact,
                        "kind": "unsupported_claim",
                        "explanation": "Точная цитата сохраняется.",
                    }
                ],
            )
        )
    )
    exact_check = check_judge_assessment(exact_claim, card)
    assert exact_check.judge_error is False
    assert exact_check.fabricated_quotes == ()
    assert exact_check.confirmed_semantic_issue is True


def test_calibration_cards_keep_facts_without_filler() -> None:
    filler = "Текст карточки повторяет только подтвержденные сведения"
    verdicts = {
        "cal_fully_supported": [JudgeVerdict.PASS],
        "cal_unit_substitution": [JudgeVerdict.FAIL],
        "cal_fabricated_certificate": [JudgeVerdict.FAIL, JudgeVerdict.REVIEW],
        "cal_unsupported_purpose": [JudgeVerdict.FAIL, JudgeVerdict.REVIEW],
        "cal_unsupported_commercial": [JudgeVerdict.FAIL, JudgeVerdict.REVIEW],
        "cal_acceptable_generalization": [JudgeVerdict.PASS],
        "cal_spec_contradiction": [JudgeVerdict.FAIL],
        "cal_keyword_stuffing": [JudgeVerdict.PASS],
    }
    anchors = {
        "cal_fabricated_certificate": "Изделие имеет сертификат ГОСТ Р.",
        "cal_unsupported_purpose": "Предназначен для автоматизации пищевой промышленности.",
        "cal_unsupported_commercial": "Лучшая цена на рынке и гарантия 5 лет.",
        "cal_acceptable_generalization": (
            "подходит для контроля избыточного давления в указанных пределах"
        ),
    }
    cases = load_judge_calibration_dataset()
    assert [case.id for case in cases] == list(verdicts)
    for case in cases:
        description = case.card.seo_description
        assert filler not in description
        assert "(×" not in description
        assert "повторено" not in description
        assert case.allowed_verdicts == verdicts[case.id]
        for fact in case.confirmed_facts:
            if case.id == "cal_unit_substitution" and fact.name == "диапазон измерений":
                assert fact.value not in description
                assert "0...1,0 bar" in description
                continue
            if case.id == "cal_spec_contradiction" and fact.name == "диапазон измерений":
                assert fact.value not in description
                assert "0...2,5 МПа" in description
                continue
            assert fact.value in description
        if case.id in anchors:
            assert anchors[case.id] in description

    stuffing = cases[-1]
    assert stuffing.card.seo_description.count("Манометр ИзмерТех МП-У-01.") >= 20
    assert expects_semantic_issue(stuffing.expected_violation_class) is False


def test_keyword_stuffing_does_not_excuse_a_factual_accusation() -> None:
    case = next(
        item
        for item in load_judge_calibration_dataset()
        if item.id == "cal_keyword_stuffing"
    )
    exact = "Манометр ИзмерТех МП-У-01."
    evidence = [
        {
            "quote": exact,
            "kind": "confirmed_fact",
            "explanation": "Название и производитель подтверждены. Повтор снижает SEO.",
        }
    ]
    passed = JudgeAssessment.model_validate(
        json.loads(
            _judge_body(
                verdict="pass",
                score=5,
                seo_score=1,
                evidence=evidence,
            )
        )
    )
    passed_result = compare_calibration(case, passed)
    assert passed.seo_quality_score == 1
    assert passed_result.verdict_match is True
    assert passed_result.claim_classification_match is True
    assert passed_result.issue_presence_match is True
    assert passed_result.matched is True
    assert passed_result.judge_error is False

    reviewed = JudgeAssessment.model_validate(
        json.loads(_judge_body(verdict="review", score=5, seo_score=1, evidence=evidence))
    )
    reviewed_result = compare_calibration(case, reviewed)
    assert reviewed_result.verdict_match is False
    assert reviewed_result.judge_error is False
    assert reviewed_result.claim_classification_match is True
    assert reviewed_result.issue_presence_match is True
    assert reviewed_result.matched is False

    accused = JudgeAssessment.model_validate(
        json.loads(
            _judge_body(
                verdict="fail",
                score=2,
                seo_score=1,
                claims=[
                    {
                        "quote": exact,
                        "explanation": "Повтор объявлен неподтвержденным фактом.",
                        "kind": "unsupported_claim",
                    }
                ],
                evidence=[
                    {
                        "quote": exact,
                        "kind": "unsupported_claim",
                        "explanation": "Ложное фактическое обвинение.",
                    }
                ],
            )
        )
    )
    accused_result = compare_calibration(case, accused)
    assert accused_result.judge_error is False
    assert accused_result.claim_classification_match is False
    assert accused_result.verdict_match is False

    abbreviated = f"{exact} (×28)"
    assert quote_in_card(abbreviated, case.card) is False
    broken = JudgeAssessment.model_validate(
        json.loads(
            _judge_body(
                verdict="pass",
                score=5,
                seo_score=1,
                evidence=[
                    {
                        "quote": abbreviated,
                        "kind": "confirmed_fact",
                        "explanation": "Сокращенный повтор.",
                    }
                ],
            )
        )
    )
    broken_result = compare_calibration(case, broken)
    assert broken_result.judge_error is True
    assert broken_result.claim_classification_match is None
    assert broken_result.matched is False


async def test_calibration_runner_bypasses_generator(tmp_path: Path) -> None:
    cases = select_calibration_cases(
        load_judge_calibration_dataset(),
        ["cal_fully_supported"],
    )
    judge = ScriptedClient(
        _judge_body(verdict="pass", score=5),
        usage=LLMUsage(prompt_tokens=10, completion_tokens=5),
    )
    report = await run_judge_calibration_report(
        cases,
        judge,
        model="judge/test",
        temperature=0.0,
        prices={"judge/test": ModelPrice(prompt_per_million=1.0, completion_per_million=2.0)},
        report_dir=tmp_path,
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
    )
    assert report.scenario_count == 1
    assert report.scenarios[0].status is CalibrationRunStatus.SCORED
    assert report.scenarios[0].matched is True
    assert report.rates.verdict_accuracy.value == 1.0
    assert report.rates.claim_classification_accuracy.value == 1.0
    assert report.rates.false_positive_rate.value == 0.0
    assert report.rates.false_negative_rate.value is None
    assert report.rates.api_error_rate.value == 0.0
    assert report.prompt_hash
    assert report.schema_hash == judge_schema_hash()
    assert len(judge.calls) == 1
    assert "Запрещенные утверждения" not in judge.calls[0]
    assert cases[0].expected_violation_class.value not in judge.calls[0].split("Карточка:")[0]
    path = write_calibration_report(report, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"] == "judge/test"
    assert "rates" in payload
    assert payload["score_means_note"]


async def test_calibration_runner_compat_wrapper() -> None:
    cases = select_calibration_cases(
        load_judge_calibration_dataset(),
        ["cal_fully_supported"],
    )
    judge = ScriptedClient(_judge_body(verdict="pass", score=5))
    result = await run_judge_calibration(cases, judge)
    assert result.case_count == 1
    assert result.matched_count == 1


async def test_calibration_separates_api_error_from_classification() -> None:
    cases = select_calibration_cases(
        load_judge_calibration_dataset(),
        ["cal_unit_substitution"],
    )
    judge = ScriptedClient(error=LLMError("network", kind=LLMErrorKind.API))
    report = await run_judge_calibration_report(
        cases,
        judge,
        model="judge/test",
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
    )
    assert report.scenarios[0].status is CalibrationRunStatus.API_ERROR
    assert report.rates.api_error_rate.value == 1.0
    assert report.rates.verdict_accuracy.value is None
    assert report.rates.claim_classification_accuracy.value is None
    assert report.api_errors == ["cal_unit_substitution"]


async def test_calibration_cli_refuses_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.evals.judge_calibration_cli.settings.openai_api_key", "")
    args = build_calibration_parser().parse_args(
        ["--judge-model", "provider/judge", "--case", "cal_unit_substitution"]
    )
    assert selected_judge(args) == "provider/judge"
    assert selected_case_ids(args) == ["cal_unit_substitution"]
    assert await run_calibration_cli(args) == 2


async def test_report_roundtrip(tmp_path: Path) -> None:
    assessment = JudgeAssessment.model_validate(json.loads(_judge_body(verdict="review", score=3)))
    assert assessment.verdict is JudgeVerdict.REVIEW
    report = await run_generation_evaluation(
        [_case("insufficient_inputs")],
        ScriptedClient("unused"),
        ScriptedClient("unused"),
        model="generator/a",
        judge_model="judge/b",
    )
    path = write_generation_report(report, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"] == "generator/a"
    assert payload["rates"]["valid_generation_rate"]["value"] is None


async def test_cli_refuses_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.evals.generation_cli.settings.openai_api_key", "")
    args = build_parser().parse_args(
        ["--model", "provider/model", "--judge-model", "provider/judge"]
    )
    assert selected_models(args) == ["provider/model"]
    assert await run_cli(args) == 2
