"""Сравнение результата extractor и сырого ответа модели с ground truth."""

import json
from collections import Counter
from typing import Any

from pydantic import ValidationError

from app.evals.metrics import (
    f1_score,
    hallucination_rate,
    has_number,
    near_match_reason,
    normalize_name,
    normalize_value,
    pair_key,
    precision,
    recall,
)
from app.evals.models import (
    ClaimRecord,
    ConflictRecord,
    ExtractionCase,
    ManualReviewItem,
    PairMetrics,
    QualityMetrics,
    ScenarioErrors,
    ScenarioReport,
    ScenarioStatus,
    ValueMismatch,
)
from app.schemas.product import (
    AttributeConflict,
    AttributeSource,
    DescriptionAttribute,
    DescriptionExtraction,
    ExtractedAttribute,
    ExtractionResult,
)
from app.services.validators import supported_quote


def score_scenario(
    case: ExtractionCase,
    actual: ExtractionResult,
    *,
    model_response: str | None,
    model_called: bool,
) -> ScenarioReport:
    """Посчитать метрики сценария. Близкие значения остаются ошибками."""

    expected_description = [
        (name, attr)
        for name, attr in case.expected.items()
        if attr.source is AttributeSource.DESCRIPTION
    ]
    actual_description = [
        attr for attr in actual.attributes if attr.source is AttributeSource.DESCRIPTION
    ]
    pair_counts, false_positives, false_negatives = _compare_attributes(
        expected_description,
        actual_description,
    )
    spec_expected, spec_preserved = _spec_preservation(case, actual)
    conflict_counts, missed_conflicts, spurious_conflicts = _compare_conflicts(
        case.conflicts,
        actual.conflicts,
    )
    numeric_expected, numeric_exact, numeric_altered = _numeric_preservation(
        expected_description,
        actual_description,
    )
    claims = parse_claims(model_response)
    unsupported = [
        claim
        for claim in claims
        if supported_quote(claim.evidence, claim.value, case.input.description) is None
    ]
    pipeline_unsupported = [
        attr
        for attr in actual_description
        if supported_quote(attr.evidence, attr.value, case.input.description) is None
    ]
    description_metrics = build_pair_metrics(
        pair_counts["tp"],
        pair_counts["fp"],
        pair_counts["fn"],
    )
    errors = ScenarioErrors(
        false_positives=false_positives,
        false_negatives=false_negatives,
        missed_conflicts=missed_conflicts,
        spurious_conflicts=spurious_conflicts,
        numeric_altered=numeric_altered,
        unsupported_claims=[_claim_record(claim) for claim in unsupported],
        forbidden_hits=_forbidden_hits(case, claims, actual),
    )
    manual_review = _manual_review(expected_description, actual_description)
    manual_review.extend(_conflict_manual_review(missed_conflicts, spurious_conflicts))
    return ScenarioReport(
        case_id=case.id,
        description=case.description,
        status=ScenarioStatus.SCORED,
        model_called=model_called,
        product=case.input,
        required_fields=list(case.required_fields),
        metrics=QualityMetrics(
            precision=description_metrics.precision,
            recall=description_metrics.recall,
            f1=description_metrics.f1,
            tp=description_metrics.tp,
            fp=description_metrics.fp,
            fn=description_metrics.fn,
            expected_pairs=description_metrics.tp + description_metrics.fn,
            predicted_pairs=description_metrics.tp + description_metrics.fp,
            hallucination_rate=hallucination_rate(len(unsupported), len(claims)),
            model_claims=len(claims),
            unsupported_claims=len(unsupported),
            pipeline_hallucination_rate=hallucination_rate(
                len(pipeline_unsupported),
                len(actual_description),
            ),
            pipeline_description_attributes=len(actual_description),
            pipeline_unsupported=len(pipeline_unsupported),
            specs_preservation=_rate(spec_preserved, spec_expected),
            specs_expected=spec_expected,
            specs_preserved=spec_preserved,
            numeric_preservation=_rate(numeric_exact, numeric_expected),
            numeric_expected=numeric_expected,
            numeric_exact=numeric_exact,
            conflicts=build_pair_metrics(
                conflict_counts["tp"],
                conflict_counts["fp"],
                conflict_counts["fn"],
            ),
        ),
        errors=errors,
        manual_review=manual_review,
        model_response=response_payload(model_response),
        expected=expected_view(case),
        actual=actual.model_dump(mode="json"),
    )


def parse_claims(raw: str | None) -> list[DescriptionAttribute]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
        parsed = DescriptionExtraction.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return []
    return list(parsed.attributes)


def response_payload(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def expected_view(case: ExtractionCase) -> dict[str, Any]:
    return {
        "attributes": {
            name: attribute.model_dump(mode="json") for name, attribute in case.expected.items()
        },
        "missing_fields": list(case.missing_fields),
        "conflicts": [conflict.model_dump(mode="json") for conflict in case.conflicts],
        "forbidden_values": dict(case.forbidden_values),
    }


def build_pair_metrics(true_positive: int, false_positive: int, false_negative: int) -> PairMetrics:
    precision_score = precision(true_positive, true_positive + false_positive)
    recall_score = recall(true_positive, true_positive + false_negative)
    return PairMetrics(
        precision=precision_score,
        recall=recall_score,
        f1=f1_score(precision_score, recall_score),
        tp=true_positive,
        fp=false_positive,
        fn=false_negative,
    )


def _rate(found: int, total: int) -> float:
    if total == 0:
        return 1.0
    return found / total


def _compare_attributes(
    expected: list[tuple[str, ExtractedAttribute]],
    actual: list[ExtractedAttribute],
) -> tuple[dict[str, int], list[ValueMismatch], list[ValueMismatch]]:
    expected_pairs = [
        _PairView(pair_key(name, attr.value), name, attr.value) for name, attr in expected
    ]
    actual_pairs = [
        _PairView(pair_key(attr.name, attr.value), attr.name, attr.value) for attr in actual
    ]
    return _match_pairs(expected_pairs, actual_pairs)


def _compare_conflicts(
    expected: list[AttributeConflict],
    actual: list[AttributeConflict],
) -> tuple[dict[str, int], list[ConflictRecord], list[ConflictRecord]]:
    expected_rows = [_ConflictView(conflict) for conflict in expected]
    actual_rows = [_ConflictView(conflict) for conflict in actual]
    expected_counts: Counter[tuple[str, str, str]] = Counter(row.key for row in expected_rows)
    actual_counts: Counter[tuple[str, str, str]] = Counter(row.key for row in actual_rows)
    true_positive = sum(min(expected_counts[key], actual_counts[key]) for key in expected_counts)
    remaining = Counter(expected_counts)
    spurious: list[ConflictRecord] = []
    for row in actual_rows:
        if remaining[row.key]:
            remaining[row.key] -= 1
            continue
        spurious.append(row.record)
    missed: list[ConflictRecord] = []
    expected_by_key: dict[tuple[str, str, str], list[_ConflictView]] = {}
    for row in expected_rows:
        expected_by_key.setdefault(row.key, []).append(row)
    for key, count in remaining.items():
        missed.extend(row.record for row in expected_by_key[key][:count])
    counts = {
        "tp": true_positive,
        "fp": sum(actual_counts.values()) - true_positive,
        "fn": sum(expected_counts.values()) - true_positive,
    }
    return counts, missed, spurious


class _PairView:
    def __init__(self, key: tuple[str, str], name: str, value: str) -> None:
        self.key = key
        self.name = name
        self.value = value


class _ConflictView:
    def __init__(self, conflict: AttributeConflict) -> None:
        self.key = (
            normalize_name(conflict.name),
            normalize_value(conflict.specs_value),
            normalize_value(conflict.description_value),
        )
        self.record = ConflictRecord(
            name=conflict.name,
            specs_value=conflict.specs_value,
            description_value=conflict.description_value,
        )


def _match_pairs(
    expected: list[_PairView],
    actual: list[_PairView],
) -> tuple[dict[str, int], list[ValueMismatch], list[ValueMismatch]]:
    expected_counts: Counter[tuple[str, str]] = Counter(item.key for item in expected)
    actual_counts: Counter[tuple[str, str]] = Counter(item.key for item in actual)
    true_positive = sum(
        min(expected_counts[key], actual_counts[key]) for key in expected_counts
    )
    false_positives: list[ValueMismatch] = []
    false_negatives: list[ValueMismatch] = []
    remaining = Counter(expected_counts)
    for item in actual:
        if remaining[item.key]:
            remaining[item.key] -= 1
            continue
        false_positives.append(ValueMismatch(name=item.name, actual=item.value))
    expected_items: dict[tuple[str, str], list[_PairView]] = {}
    for item in expected:
        expected_items.setdefault(item.key, []).append(item)
    for key, count in remaining.items():
        for item in expected_items[key][:count]:
            false_negatives.append(ValueMismatch(name=item.name, expected=item.value))
    return (
        {
            "tp": true_positive,
            "fp": sum(actual_counts.values()) - true_positive,
            "fn": sum(expected_counts.values()) - true_positive,
        },
        false_positives,
        false_negatives,
    )


def _spec_preservation(case: ExtractionCase, actual: ExtractionResult) -> tuple[int, int]:
    actual_by_name = {normalize_name(attr.name): attr for attr in actual.attributes}
    expected = [
        (name, attr)
        for name, attr in case.expected.items()
        if attr.source is AttributeSource.SPECS
    ]
    preserved = 0
    for name, attr in expected:
        found = actual_by_name.get(normalize_name(name))
        if found is None or found.source is not AttributeSource.SPECS:
            continue
        if normalize_value(found.value) == normalize_value(attr.value):
            preserved += 1
    return len(expected), preserved


def _numeric_preservation(
    expected: list[tuple[str, ExtractedAttribute]],
    actual: list[ExtractedAttribute],
) -> tuple[int, int, list[ValueMismatch]]:
    actual_by_name = {normalize_name(attr.name): attr for attr in actual}
    numeric = [(name, attr) for name, attr in expected if has_number(attr.value)]
    exact = 0
    altered: list[ValueMismatch] = []
    for name, attr in numeric:
        found = actual_by_name.get(normalize_name(name))
        if found is not None and normalize_value(found.value) == normalize_value(attr.value):
            exact += 1
            continue
        if found is not None:
            altered.append(
                ValueMismatch(name=name, expected=attr.value, actual=found.value)
            )
    return len(numeric), exact, altered


def _manual_review(
    expected: list[tuple[str, ExtractedAttribute]],
    actual: list[ExtractedAttribute],
) -> list[ManualReviewItem]:
    actual_by_name = {normalize_name(attr.name): attr for attr in actual}
    reviews: list[ManualReviewItem] = []
    for name, attr in expected:
        found = actual_by_name.get(normalize_name(name))
        if found is None:
            continue
        reason = near_match_reason(attr.value, found.value)
        if reason is None:
            continue
        reviews.append(
            ManualReviewItem(
                name=name,
                expected=attr.value,
                actual=found.value,
                reason=reason,
            )
        )
    return reviews


def _conflict_manual_review(
    missed: list[ConflictRecord],
    spurious: list[ConflictRecord],
) -> list[ManualReviewItem]:
    spurious_by_name = {normalize_name(item.name): item for item in spurious}
    reviews: list[ManualReviewItem] = []
    for missed_item in missed:
        found = spurious_by_name.get(normalize_name(missed_item.name))
        if found is None:
            continue
        reason = near_match_reason(missed_item.description_value, found.description_value)
        if reason is None:
            continue
        reviews.append(
            ManualReviewItem(
                name=missed_item.name,
                expected=missed_item.description_value,
                actual=found.description_value,
                reason=reason,
            )
        )
    return reviews


def _forbidden_hits(
    case: ExtractionCase,
    claims: list[DescriptionAttribute],
    actual: ExtractionResult,
) -> list[ClaimRecord]:
    forbidden = {
        (normalize_name(name), normalize_value(value))
        for name, value in case.forbidden_values.items()
    }
    if not forbidden:
        return []
    hits: list[ClaimRecord] = []
    seen: set[tuple[str, str]] = set()
    candidates = [(item.name, item.value, item.evidence) for item in claims]
    candidates.extend((item.name, item.value, item.evidence) for item in actual.attributes)
    for name, value, evidence in candidates:
        key = (normalize_name(name), normalize_value(value))
        if key not in forbidden or key in seen:
            continue
        seen.add(key)
        hits.append(ClaimRecord(name=name, value=value, evidence=evidence))
    return hits


def _claim_record(claim: DescriptionAttribute) -> ClaimRecord:
    return ClaimRecord(name=claim.name, value=claim.value, evidence=claim.evidence)
