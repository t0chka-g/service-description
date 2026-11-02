"""Unit-тесты метрик оценки извлечения."""

import json
from datetime import UTC, datetime

import pytest

from app.clients.llm import LLMError, LLMErrorKind
from app.evals.metrics import (
    f1_score,
    hallucination_rate,
    near_match_reason,
    normalize_name,
    normalize_value,
    precision,
    recall,
)
from app.evals.models import ExtractionCase, ScenarioStatus
from app.evals.runner import (
    build_report,
    extraction_prompt_hash,
    labeling_notes_for,
    load_cases,
    run_evaluation,
)
from app.evals.scoring import score_scenario
from app.schemas.product import (
    AttributeConflict,
    AttributeSource,
    ExtractedAttribute,
    ExtractionResult,
    ProductInput,
)
from tests.fakes import FakeLLMClient


def test_precision_recall_f1_and_hallucination_rates() -> None:
    assert precision(true_positive=0, predicted=0) == 1.0
    assert precision(true_positive=1, predicted=2) == pytest.approx(0.5)
    assert precision(true_positive=0, predicted=3) == 0.0

    assert recall(true_positive=0, expected=0) == 1.0
    assert recall(true_positive=1, expected=4) == pytest.approx(0.25)
    assert recall(true_positive=0, expected=2) == 0.0

    assert f1_score(0.0, 0.0) == 0.0
    assert f1_score(1.0, 1.0) == 1.0
    assert f1_score(0.5, 1.0) == pytest.approx(2 / 3)
    assert f1_score(1.0, 0.0) == 0.0

    assert hallucination_rate(unsupported=0, extracted=0) == 0.0
    assert hallucination_rate(unsupported=1, extracted=4) == pytest.approx(0.25)
    assert hallucination_rate(unsupported=4, extracted=4) == 1.0


def test_normalization_keeps_numbers_units_and_value_case() -> None:
    assert normalize_name("  Класс   точности ") == "класс точности"
    assert normalize_value("  0...1,0   МПа ") == "0...1,0 МПа"
    assert normalize_value("1,5") == "1,5"
    assert normalize_value("IP65") == "IP65"
    assert normalize_value("4…20mA") == "4…20mA"

    assert near_match_reason("1,5", "1.5") == "notation"
    assert near_match_reason("4…20mA", "4...20 mA") == "notation"
    assert near_match_reason("0...10 бар", "0...10 bar") == "same_numbers"
    assert near_match_reason("K", "тип K") == "token_extension"
    assert near_match_reason("1,5", "1,6") is None
    assert near_match_reason("24 В DC", "24 В DC") is None


def test_wrong_value_is_false_positive_and_false_negative() -> None:
    case = _case(
        description="Питание 24 В DC. Корпус пластиковый.",
        expected={"питание": _description("питание", "24 В DC", "Питание 24 В DC")},
    )
    actual = _result([_description("питание", "220 В", "220 В")])

    score = score_scenario(case, actual, model_response=None, model_called=False)
    metrics = _metrics(score)

    assert metrics.tp == 0
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert score.manual_review == []
    assert score.errors.false_positives[0].actual == "220 В"
    assert score.errors.false_negatives[0].expected == "24 В DC"


def test_near_values_stay_unmatched_and_go_to_manual_review() -> None:
    case = _case(
        description="Класс точности 1.5.",
        expected={"класс точности": _description("класс точности", "1,5", "1,5")},
    )
    actual = _result([_description("Класс точности", "1.5", "1.5")])

    score = score_scenario(case, actual, model_response=None, model_called=False)
    metrics = _metrics(score)

    assert metrics.tp == 0
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert score.manual_review[0].reason == "notation"
    assert score.manual_review[0].expected == "1,5"
    assert score.manual_review[0].actual == "1.5"


def test_missing_source_field_is_not_a_false_negative() -> None:
    case = _case(
        description="Датчик без класса точности.",
        fields=["диапазон измерений", "класс точности"],
        expected={},
        missing_fields=["класс точности"],
    )
    actual = _result([])

    score = score_scenario(case, actual, model_response='{"attributes": []}', model_called=True)
    metrics = _metrics(score)

    assert metrics.fn == 0
    assert metrics.fp == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.hallucination_rate == 0.0


def test_empty_prediction_does_not_invent_precision() -> None:
    case = _case(
        description="Питание 24 В DC.",
        expected={"питание": _description("питание", "24 В DC", "Питание 24 В DC")},
    )

    score = score_scenario(
        case,
        _result([]),
        model_response='{"attributes": []}',
        model_called=True,
    )
    metrics = _metrics(score)

    assert metrics.predicted_pairs == 0
    assert metrics.expected_pairs == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.hallucination_rate == 0.0


def test_specs_copy_does_not_count_as_description_hit() -> None:
    case = _case(
        description="Питание 24 В DC. Корпус пластиковый.",
        specs={"диапазон измерений": "0...1,0 МПа"},
        fields=["диапазон измерений", "питание"],
        expected={
            "диапазон измерений": _specs("диапазон измерений", "0...1,0 МПа"),
            "питание": _description("питание", "24 В DC", "Питание 24 В DC"),
        },
    )
    actual = _result([_specs("диапазон измерений", "0...1,0 МПа")])

    score = score_scenario(case, actual, model_response='{"attributes": []}', model_called=True)
    metrics = _metrics(score)

    assert metrics.tp == 0
    assert metrics.fn == 1
    assert metrics.specs_expected == 1
    assert metrics.specs_preserved == 1
    assert metrics.f1 == 0.0


def test_name_case_and_extra_spaces_match_without_changing_numbers() -> None:
    case = _case(
        description="Питание 24 В DC.",
        expected={"Питание": _description("Питание", "24 В DC", "Питание 24 В DC")},
    )
    actual = _result([_description("питание", "24  В   DC", "Питание 24 В DC")])

    score = score_scenario(case, actual, model_response=None, model_called=False)

    assert _metrics(score).tp == 1
    assert _metrics(score).fp == 0
    assert _metrics(score).fn == 0


def test_value_case_difference_is_not_a_match() -> None:
    case = _case(
        description="Питание 24 В DC.",
        expected={"питание": _description("питание", "24 В DC", "Питание 24 В DC")},
    )
    actual = _result([_description("питание", "24 в dc", "24 в dc")])

    score = score_scenario(case, actual, model_response=None, model_called=False)

    assert _metrics(score).tp == 0
    assert score.manual_review[0].reason == "notation"


def test_unsupported_output_counts_as_pipeline_hallucination() -> None:
    description = "Манометр для воды."
    case = _case(description=description, expected={}, missing_fields=["класс точности"])
    actual = _result([_description("класс точности", "0,05", "класс точности 0,05")])
    raw = json.dumps(
        {
            "attributes": [
                {
                    "name": "класс точности",
                    "value": "0,05",
                    "evidence": "класс точности 0,05",
                }
            ]
        },
        ensure_ascii=False,
    )

    score = score_scenario(case, actual, model_response=raw, model_called=True)
    metrics = _metrics(score)

    assert metrics.fp == 1
    assert metrics.fn == 0
    assert metrics.hallucination_rate == 1.0
    assert metrics.pipeline_hallucination_rate == 1.0
    assert score.errors.unsupported_claims[0].value == "0,05"


def test_altered_number_is_not_preserved() -> None:
    case = _case(
        description="Диапазон измерений 0...10 бар.",
        expected={
            "диапазон измерений": _description(
                "диапазон измерений",
                "0...10 бар",
                "Диапазон измерений 0...10 бар",
            )
        },
    )
    actual = _result(
        [_description("диапазон измерений", "0...16 бар", "Диапазон измерений 0...16 бар")]
    )

    score = score_scenario(case, actual, model_response=None, model_called=False)
    metrics = _metrics(score)

    assert metrics.tp == 0
    assert metrics.numeric_expected == 1
    assert metrics.numeric_exact == 0
    assert metrics.numeric_preservation == 0.0
    assert score.errors.numeric_altered[0].expected == "0...10 бар"
    assert score.errors.numeric_altered[0].actual == "0...16 бар"
    assert score.manual_review == []


def test_missed_conflict_is_separate_from_description_f1() -> None:
    case = _case(
        description="Диапазон измерений 0...2,5 МПа.",
        specs={"диапазон измерений": "0...1,6 МПа"},
        expected={"диапазон измерений": _specs("диапазон измерений", "0...1,6 МПа")},
        conflicts=[
            AttributeConflict(
                name="диапазон измерений",
                specs_value="0...1,6 МПа",
                description_value="0...2,5 МПа",
                evidence="Диапазон измерений 0...2,5 МПа",
            )
        ],
    )
    actual = ExtractionResult(
        attributes=[_specs("диапазон измерений", "0...1,6 МПа")],
        conflicts=[],
    )

    score = score_scenario(case, actual, model_response=None, model_called=False)
    metrics = _metrics(score)

    assert metrics.expected_pairs == 0
    assert metrics.f1 == 1.0
    assert metrics.conflicts.tp == 0
    assert metrics.conflicts.fn == 1
    assert metrics.specs_preserved == 1
    assert score.errors.missed_conflicts[0].description_value == "0...2,5 МПа"


async def test_report_contains_scenario_payload_and_aggregate(tmp_path) -> None:
    description = "Питание 24 В DC. Корпус пластиковый."
    case = _case(
        case_id="report_case",
        description_text="Проверка отчета",
        description=description,
        expected={"питание": _description("питание", "24 В DC", "Питание 24 В DC")},
    )
    client = FakeLLMClient(
        {
            "attributes": [
                {
                    "name": "питание",
                    "value": "24 В DC",
                    "evidence": "Питание 24 В DC",
                }
            ]
        }
    )
    created = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)

    report = await run_evaluation(
        [case],
        client,
        model="fake model",
        report_dir=tmp_path,
        created_at=created,
    )

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    text = written[0].read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "Проверка отчета" in text
    assert "\\u" not in text
    assert payload["created_at"].startswith("2026-11-03T12:00:00")
    assert payload["model"] == "fake model"
    assert payload["prompt_hash"] == extraction_prompt_hash()
    assert len(payload["prompt_hash"]) == 64
    assert payload["scenario_count"] == 1
    assert payload["scored_count"] == 1
    assert payload["api_error_count"] == 0
    assert payload["model_error_count"] == 0
    assert payload["metrics"]["tp"] == 1
    assert payload["metrics"]["f1"] == 1.0
    assert payload["metrics"]["hallucination_rate"] == 0.0
    scenario = payload["scenarios"][0]
    assert scenario["status"] == "scored"
    assert scenario["model_response"]["attributes"][0]["value"] == "24 В DC"
    assert scenario["expected"]["attributes"]["питание"]["value"] == "24 В DC"
    assert scenario["actual"]["attributes"][0]["value"] == "24 В DC"
    assert scenario["errors"]["false_positives"] == []
    assert report.metrics is not None
    assert report.metrics.macro_f1 == 1.0


async def test_api_error_is_excluded_from_quality_metrics() -> None:
    good = _case(
        case_id="good",
        description="Питание 24 В DC. Корпус пластиковый.",
        expected={"питание": _description("питание", "24 В DC", "Питание 24 В DC")},
    )
    failed = _case(
        case_id="api_case",
        name="API_CASE",
        description="Текст без нужной характеристики.",
        expected={"класс точности": _description("класс точности", "1,5", "1,5")},
    )

    class _Client:
        async def complete(self, prompt: str) -> str:
            if "API_CASE" in prompt:
                raise LLMError("OpenAI request failed", kind=LLMErrorKind.API)
            return json.dumps(
                {
                    "attributes": [
                        {
                            "name": "питание",
                            "value": "24 В DC",
                            "evidence": "Питание 24 В DC",
                        }
                    ]
                },
                ensure_ascii=False,
            )

    report = await run_evaluation([good, failed], _Client(), model="fake")

    assert report.api_error_count == 1
    assert report.model_error_count == 0
    assert report.scored_count == 1
    assert report.metrics is not None
    assert report.metrics.tp == 1
    assert report.metrics.fn == 0
    failed_report = report.scenarios[1]
    assert failed_report.status is ScenarioStatus.API_ERROR
    assert failed_report.metrics is None
    assert failed_report.model_response is None
    assert failed_report.expected["attributes"]["класс точности"]["value"] == "1,5"


async def test_model_schema_error_is_not_scored_as_empty_extraction() -> None:
    case = _case(
        description="Питание 24 В DC.",
        expected={"питание": _description("питание", "24 В DC", "Питание 24 В DC")},
    )

    class _Client:
        async def complete(self, prompt: str) -> str:
            return "not-json"

    report = await run_evaluation([case], _Client(), model="fake")

    assert report.model_error_count == 1
    assert report.scored_count == 0
    assert report.metrics is None
    scenario = report.scenarios[0]
    assert scenario.status is ScenarioStatus.MODEL_ERROR
    assert scenario.metrics is None
    assert scenario.model_response == "not-json"
    assert "extraction schema" in (scenario.error_message or "")


async def test_empty_input_does_not_call_the_model() -> None:
    case = next(item for item in load_cases() if item.id == "empty_invalid_input")
    client = FakeLLMClient(
        {"attributes": [{"name": "класс точности", "value": "1,5", "evidence": "1,5"}]}
    )

    report = await run_evaluation([case], client, model="fake")
    metrics = _metrics(report.scenarios[0])

    assert client.calls == []
    assert report.scenarios[0].model_called is False
    assert metrics.fn == 0
    assert metrics.expected_pairs == 0
    assert metrics.hallucination_rate == 0.0
    assert metrics.f1 == 1.0


async def test_canned_fixture_responses_match_ground_truth_without_spec_inflation(
    extraction_cases: list[dict],
) -> None:
    cases = load_cases()
    client = _FixtureLLM(extraction_cases)

    report = await run_evaluation(cases, client, model="canned")
    metrics = report.metrics
    assert metrics is not None

    description_pairs = sum(
        attribute.source is AttributeSource.DESCRIPTION
        for case in cases
        for attribute in case.expected.values()
    )
    spec_pairs = sum(
        attribute.source is AttributeSource.SPECS
        for case in cases
        for attribute in case.expected.values()
    )
    conflict_pairs = sum(len(case.conflicts) for case in cases)

    assert metrics.tp == description_pairs
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.f1 == 1.0
    assert description_pairs > 0
    assert metrics.tp < description_pairs + spec_pairs
    assert metrics.specs_preserved == spec_pairs
    assert metrics.conflicts.tp == conflict_pairs
    assert metrics.conflicts.fp == 0
    assert metrics.conflicts.fn == 0
    assert metrics.pipeline_hallucination_rate == 0.0
    assert metrics.hallucination_rate > 0.0
    assert metrics.numeric_preservation == 1.0
    assert report.api_error_count == 0
    assert report.model_error_count == 0

    specs_only = next(item for item in report.scenarios if item.case_id == "specs_complete")
    assert _metrics(specs_only).expected_pairs == 0
    assert _metrics(specs_only).tp == 0
    injection = next(
        item for item in report.scenarios if item.case_id == "prompt_injection_no_hallucination"
    )
    assert _metrics(injection).hallucination_rate == pytest.approx(0.8)
    assert len(injection.errors.forbidden_hits) == 4
    assert _metrics(injection).pipeline_hallucination_rate == 0.0


def test_dataset_notes_keep_known_ambiguity_without_integrity_gaps() -> None:
    cases = load_cases()

    notes = labeling_notes_for(cases)

    assert len(cases) == 16
    assert [note.case_id for note in notes] == ["synonym_names"]
    assert extraction_prompt_hash() == extraction_prompt_hash()


def test_prompt_hash_changes_when_source_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    original = extraction_prompt_hash()

    def _other_prompt(*_args: object, **_kwargs: object) -> str:
        return "другой промпт"

    monkeypatch.setattr("app.evals.runner.build_extraction_prompt", _other_prompt)

    assert extraction_prompt_hash() != original


def test_macro_f1_ignores_specs_only_scenarios() -> None:
    specs_case = _case(
        case_id="specs",
        description="Текст без дополнительной характеристики.",
        specs={"класс точности": "1,5"},
        expected={"класс точности": _specs("класс точности", "1,5")},
    )
    specs_score = score_scenario(
        specs_case,
        _result([_specs("класс точности", "1,5")]),
        model_response=None,
        model_called=False,
    )
    missed_case = _case(
        case_id="missed",
        description="Питание 24 В DC.",
        expected={"питание": _description("питание", "24 В DC", "Питание 24 В DC")},
    )
    missed_score = score_scenario(
        missed_case,
        _result([]),
        model_response=None,
        model_called=False,
    )

    report = build_report(
        [specs_score, missed_score],
        [specs_case, missed_case],
        model="fake",
        created_at=datetime(2026, 11, 3, tzinfo=UTC),
    )

    assert report.metrics is not None
    assert _metrics(specs_score).f1 == 1.0
    assert _metrics(missed_score).f1 == 0.0
    assert report.metrics.macro_f1 == 0.0
    assert report.metrics.description_cases_in_macro == 1
    assert report.metrics.tp == 0
    assert report.metrics.specs_preservation == 1.0


def test_build_report_without_scored_scenarios_has_no_quality_metrics() -> None:
    case = _case(description="Питание 24 В DC.")
    scenario = score_scenario(
        case,
        _result([]),
        model_response=None,
        model_called=False,
    )
    failed = scenario.model_copy(
        update={"status": ScenarioStatus.API_ERROR, "metrics": None, "actual": None}
    )

    report = build_report(
        [failed],
        [case],
        model="fake",
        created_at=datetime(2026, 11, 3, tzinfo=UTC),
    )

    assert report.metrics is None
    assert report.scored_count == 0
    assert report.scenario_count == 1


class _FixtureLLM:
    def __init__(self, raw_cases: list[dict]) -> None:
        self._by_name = {
            case["input"]["name"]: case for case in raw_cases if case["input"]["name"]
        }

    async def complete(self, prompt: str) -> str:
        matches = [name for name in self._by_name if name in prompt]
        if not matches:
            raise AssertionError("prompt did not match a fixture product")
        name = max(matches, key=len)
        return json.dumps(self._by_name[name]["llm_response"], ensure_ascii=False)


def _case(
    *,
    case_id: str = "case",
    description_text: str = "сценарий",
    name: str = "Датчик",
    description: str,
    specs: dict[str, str] | None = None,
    fields: list[str] | None = None,
    expected: dict[str, ExtractedAttribute] | None = None,
    missing_fields: list[str] | None = None,
    conflicts: list[AttributeConflict] | None = None,
) -> ExtractionCase:
    attributes = expected or {}
    return ExtractionCase(
        id=case_id,
        description=description_text,
        input=ProductInput(
            name=name,
            manufacturer="Завод",
            specs=specs or {},
            description=description,
        ),
        required_fields=fields or list(attributes),
        expected=attributes,
        missing_fields=missing_fields or [],
        conflicts=conflicts or [],
    )


def _description(name: str, value: str, evidence: str) -> ExtractedAttribute:
    return ExtractedAttribute(
        name=name,
        value=value,
        source=AttributeSource.DESCRIPTION,
        evidence=evidence,
    )


def _specs(name: str, value: str) -> ExtractedAttribute:
    return ExtractedAttribute(
        name=name,
        value=value,
        source=AttributeSource.SPECS,
        evidence=value,
    )


def _result(attributes: list[ExtractedAttribute]) -> ExtractionResult:
    return ExtractionResult(attributes=attributes)


def _metrics(scenario):
    assert scenario.metrics is not None
    return scenario.metrics
