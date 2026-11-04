"""Проверка ответа судьи без доверия к его самооценке.

Цитаты должны встречаться в карточке. Противоречивый разбор
не считается достоверной семантической оценкой.
"""

from dataclasses import dataclass

from app.evals.judge import ClaimKind, JudgeAssessment, JudgeVerdict
from app.schemas.product import GenerationResult


@dataclass(frozen=True)
class JudgeCheckResult:
    """Итог детерминированной проверки вердикта судьи."""

    fabricated_quotes: tuple[str, ...]
    inconsistent: bool
    confirmed_semantic_issue: bool
    needs_manual_review: bool
    review_reason: str
    judge_error: bool


def card_corpus(card: GenerationResult) -> str:
    """Текст карточки, в котором ищутся цитаты судьи."""

    parts = [
        card.seo_title,
        card.seo_description or "",
        card.commercial_block or "",
        *card.search_queries,
    ]
    return "\n".join(parts)


def quote_in_card(quote: str, card: GenerationResult) -> bool:
    fragment = quote.strip()
    if not fragment:
        return False
    return fragment in card_corpus(card)


def check_judge_assessment(
    assessment: JudgeAssessment,
    card: GenerationResult,
) -> JudgeCheckResult:
    """Отделить подтвержденные претензии от сбоя разбора судьи."""

    fabricated = _fabricated_quotes(assessment, card)
    verified_claims = [
        claim
        for claim in assessment.unsupported_claims
        if quote_in_card(claim.quote, card)
    ]
    evidence_conflict = _evidence_conflicts_claims(assessment, card)
    has_critical = bool(assessment.critical_errors)
    has_verified = bool(verified_claims)
    low_score = assessment.factual_consistency_score < 3
    inconsistent = _inconsistent(
        assessment,
        has_verified=has_verified,
        has_critical=has_critical,
        low_score=low_score,
        fabricated=bool(fabricated),
        evidence_conflict=evidence_conflict,
    )
    reasons: list[str] = []
    if fabricated:
        reasons.append("Судья сослался на цитату, которой нет в карточке.")
    if evidence_conflict:
        reasons.append("evidence противоречит unsupported_claims.")
    if inconsistent and not fabricated and not evidence_conflict:
        reasons.append("Вердикт судьи не согласован с претензиями и оценками.")
    if assessment.verdict is JudgeVerdict.REVIEW and not reasons:
        reasons.append("Судья пометил вердикт как требующий ручной проверки.")

    judge_error = bool(fabricated) or evidence_conflict
    needs_review = bool(reasons) or inconsistent or assessment.verdict is JudgeVerdict.REVIEW
    confirmed = (
        not inconsistent
        and not fabricated
        and not evidence_conflict
        and (has_verified or has_critical)
    )
    return JudgeCheckResult(
        fabricated_quotes=fabricated,
        inconsistent=inconsistent,
        confirmed_semantic_issue=confirmed,
        needs_manual_review=needs_review,
        review_reason=" ".join(reasons),
        judge_error=judge_error,
    )


def _fabricated_quotes(
    assessment: JudgeAssessment,
    card: GenerationResult,
) -> tuple[str, ...]:
    quotes: list[str] = []
    for claim in assessment.unsupported_claims:
        if not quote_in_card(claim.quote, card):
            quotes.append(claim.quote)
    for item in assessment.evidence:
        if not quote_in_card(item.quote, card):
            quotes.append(item.quote)
    return tuple(quotes)


def _evidence_conflicts_claims(
    assessment: JudgeAssessment,
    card: GenerationResult,
) -> bool:
    bad_kinds = {ClaimKind.UNSUPPORTED_CLAIM, ClaimKind.DIRECT_CONTRADICTION}
    good_kinds = {ClaimKind.CONFIRMED_FACT, ClaimKind.ACCEPTABLE_GENERALIZATION}
    claim_quotes = {
        claim.quote.strip()
        for claim in assessment.unsupported_claims
        if claim.kind in bad_kinds and quote_in_card(claim.quote, card)
    }
    for item in assessment.evidence:
        if item.kind in good_kinds and item.quote.strip() in claim_quotes:
            return True
    return False


def _inconsistent(
    assessment: JudgeAssessment,
    *,
    has_verified: bool,
    has_critical: bool,
    low_score: bool,
    fabricated: bool,
    evidence_conflict: bool,
) -> bool:
    if fabricated or evidence_conflict:
        return True
    verdict = assessment.verdict
    if verdict is JudgeVerdict.PASS and (has_verified or has_critical or low_score):
        return True
    if verdict is JudgeVerdict.FAIL and not has_verified and not has_critical:
        return True
    if low_score and not has_verified and not has_critical:
        return True
    return False
