"""Прогон Extractor по ground truth и сохранение JSON-отчета."""

import hashlib
import inspect
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from app.clients.llm import LLMClient, LLMError, LLMErrorKind
from app.evals.metrics import normalize_name
from app.evals.models import (
    METHODOLOGY,
    STATIC_LABELING_NOTES,
    AggregateMetrics,
    EvaluationReport,
    ExtractionCase,
    LabelingNote,
    ScenarioReport,
    ScenarioStatus,
)
from app.evals.scoring import build_pair_metrics, expected_view, response_payload, score_scenario
from app.prompts.extraction import build_extraction_prompt
from app.schemas.product import AttributeSource
from app.services.extractor import CharacteristicExtractor

_SAFE_MODEL = re.compile(r"[^\w.\-]+", re.UNICODE)


def dataset_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "extraction_cases.json"


def default_report_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "reports" / "evals"


def extraction_prompt_hash() -> str:
    """SHA-256 исходника промпта: отчет привязан к версии инструкции, а не к ответу."""

    source = inspect.getsource(build_extraction_prompt)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def load_cases(path: Path | None = None) -> list[ExtractionCase]:
    """Загрузить ground truth. Кандидатский llm_response фикстуры игнорируется."""

    payload = json.loads((path or dataset_path()).read_text(encoding="utf-8"))
    cases: list[ExtractionCase] = []
    for item in payload:
        expected = {
            name: {**attributes, "name": name} for name, attributes in item["expected"].items()
        }
        cases.append(ExtractionCase.model_validate({**item, "expected": expected}))
    return cases


def labeling_notes_for(cases: Sequence[ExtractionCase]) -> list[LabelingNote]:
    """Замечания к разметке. Сам ground truth эта функция не меняет."""

    known = {case.id for case in cases}
    notes = [note for note in STATIC_LABELING_NOTES if note.case_id in known]
    for case in cases:
        notes.extend(_integrity_notes(case))
    return notes


async def run_evaluation(
    cases: Sequence[ExtractionCase],
    llm_client: LLMClient,
    *,
    model: str,
    base_url: str | None = None,
    report_dir: Path | None = None,
    created_at: datetime | None = None,
) -> EvaluationReport:
    """Последовательно прогнать сценарии и, если задан каталог, записать JSON."""

    scenarios: list[ScenarioReport] = []
    for case in cases:
        scenarios.append(await _run_case(case, llm_client))
    report = build_report(
        scenarios,
        cases,
        model=model,
        base_url=base_url,
        created_at=created_at,
    )
    if report_dir is not None:
        write_report(report, report_dir)
    return report


def build_report(
    scenarios: Sequence[ScenarioReport],
    cases: Sequence[ExtractionCase],
    *,
    model: str,
    base_url: str | None = None,
    created_at: datetime | None = None,
) -> EvaluationReport:
    scored = [scenario for scenario in scenarios if scenario.status is ScenarioStatus.SCORED]
    created = created_at or datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return EvaluationReport(
        created_at=created,
        model=model,
        base_url=base_url or None,
        prompt_hash=extraction_prompt_hash(),
        scenario_count=len(scenarios),
        scored_count=len(scored),
        api_error_count=sum(
            scenario.status is ScenarioStatus.API_ERROR for scenario in scenarios
        ),
        model_error_count=sum(
            scenario.status is ScenarioStatus.MODEL_ERROR for scenario in scenarios
        ),
        methodology=METHODOLOGY,
        labeling_notes=labeling_notes_for(cases),
        metrics=_aggregate(scored),
        scenarios=list(scenarios),
    )


def write_report(report: EvaluationReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.strftime("%Y%m%dT%H%M%S%fZ")
    safe_model = _SAFE_MODEL.sub("_", report.model).strip("_") or "model"
    path = directory / f"{stamp}_{safe_model}.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


class _RecordingLLMClient:
    """Запоминает ответ модели, не выполняя второй запрос."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.responses: list[str] = []

    async def complete(self, prompt: str) -> str:
        response = await self._inner.complete(prompt)
        self.responses.append(response)
        return response


async def _run_case(case: ExtractionCase, llm_client: LLMClient) -> ScenarioReport:
    recorder = _RecordingLLMClient(llm_client)
    extractor = CharacteristicExtractor(recorder)
    try:
        actual = await extractor.extract(case.input, case.required_fields)
    except LLMError as exc:
        status = (
            ScenarioStatus.API_ERROR
            if exc.kind is LLMErrorKind.API
            else ScenarioStatus.MODEL_ERROR
        )
        raw = recorder.responses[-1] if recorder.responses else None
        return ScenarioReport(
            case_id=case.id,
            description=case.description,
            status=status,
            error_message=str(exc),
            model_called=bool(recorder.responses),
            product=case.input,
            required_fields=list(case.required_fields),
            model_response=response_payload(raw),
            expected=expected_view(case),
        )
    raw = recorder.responses[-1] if recorder.responses else None
    return score_scenario(
        case,
        actual,
        model_response=raw,
        model_called=bool(recorder.responses),
    )


def _aggregate(scenarios: Sequence[ScenarioReport]) -> AggregateMetrics | None:
    rows = [scenario.metrics for scenario in scenarios if scenario.metrics is not None]
    if not rows:
        return None
    description = build_pair_metrics(
        sum(row.tp for row in rows),
        sum(row.fp for row in rows),
        sum(row.fn for row in rows),
    )
    conflicts = build_pair_metrics(
        sum(row.conflicts.tp for row in rows),
        sum(row.conflicts.fp for row in rows),
        sum(row.conflicts.fn for row in rows),
    )
    relevant = [row for row in rows if row.expected_pairs or row.predicted_pairs]
    macro_f1 = sum(row.f1 for row in relevant) / len(relevant) if relevant else 1.0
    specs_expected = sum(row.specs_expected for row in rows)
    specs_preserved = sum(row.specs_preserved for row in rows)
    numeric_expected = sum(row.numeric_expected for row in rows)
    numeric_exact = sum(row.numeric_exact for row in rows)
    model_claims = sum(row.model_claims for row in rows)
    unsupported_claims = sum(row.unsupported_claims for row in rows)
    pipeline_attributes = sum(row.pipeline_description_attributes for row in rows)
    pipeline_unsupported = sum(row.pipeline_unsupported for row in rows)
    return AggregateMetrics(
        precision=description.precision,
        recall=description.recall,
        f1=description.f1,
        tp=description.tp,
        fp=description.fp,
        fn=description.fn,
        expected_pairs=description.tp + description.fn,
        predicted_pairs=description.tp + description.fp,
        hallucination_rate=_ratio(unsupported_claims, model_claims, empty=0.0),
        model_claims=model_claims,
        unsupported_claims=unsupported_claims,
        pipeline_hallucination_rate=_ratio(pipeline_unsupported, pipeline_attributes, empty=0.0),
        pipeline_description_attributes=pipeline_attributes,
        pipeline_unsupported=pipeline_unsupported,
        specs_preservation=_ratio(specs_preserved, specs_expected, empty=1.0),
        specs_expected=specs_expected,
        specs_preserved=specs_preserved,
        numeric_preservation=_ratio(numeric_exact, numeric_expected, empty=1.0),
        numeric_expected=numeric_expected,
        numeric_exact=numeric_exact,
        conflicts=conflicts,
        macro_f1=macro_f1,
        description_cases_in_macro=len(relevant),
    )


def _ratio(found: int, total: int, *, empty: float) -> float:
    if total == 0:
        return empty
    return found / total


def _integrity_notes(case: ExtractionCase) -> list[LabelingNote]:
    notes: list[LabelingNote] = []
    expected_names = {normalize_name(name) for name in case.expected}
    missing_names = {normalize_name(name) for name in case.missing_fields}
    overlap = sorted(expected_names & missing_names)
    if overlap:
        joined = ", ".join(overlap)
        notes.append(
            LabelingNote(
                case_id=case.id,
                note=f"Поле одновременно ожидается и числится отсутствующим: {joined}.",
            )
        )
    covered = expected_names | missing_names
    uncovered = [field for field in case.required_fields if normalize_name(field) not in covered]
    if uncovered:
        joined = ", ".join(uncovered)
        notes.append(
            LabelingNote(
                case_id=case.id,
                note=f"Обязательное поле не входит ни в expected, ни в missing_fields: {joined}.",
            )
        )
    for name, attribute in case.expected.items():
        if attribute.source is AttributeSource.DESCRIPTION:
            if attribute.value.strip() not in case.input.description:
                notes.append(
                    LabelingNote(
                        case_id=case.id,
                        note=(
                            f"Значение «{attribute.value}» поля «{name}» "
                            "не найдено в description."
                        ),
                    )
                )
            continue
        if attribute.source is AttributeSource.SPECS:
            spec_value = _lookup_spec(case.input.specs, name)
            if spec_value != attribute.value:
                notes.append(
                    LabelingNote(
                        case_id=case.id,
                        note=(
                            f"Значение specs для «{name}» в разметке «{attribute.value}», "
                            f"во входе «{spec_value}»."
                        ),
                    )
                )
    return notes


def _lookup_spec(specs: dict[str, str], field: str) -> str | None:
    if field in specs and specs[field].strip():
        return specs[field]
    folded = field.strip().casefold()
    for key, value in specs.items():
        if key.strip().casefold() == folded and value.strip():
            return value
    return None
