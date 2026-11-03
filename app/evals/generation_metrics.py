"""Детерминированные метрики генерации.

Закрытые по недостатку данных сценарии не входят в знаменатель
успешности, схемы и ошибок генерации. Нулевой знаменатель дает null,
а не ноль и не исключение.
"""

from pydantic import BaseModel, ConfigDict

from app.clients.llm import LLMUsage


class ModelPrice(BaseModel):
    """Тариф за 1 000 000 токенов. Идентификатор модели задает вызывающий код."""

    model_config = ConfigDict(extra="forbid")

    prompt_per_million: float
    completion_per_million: float


class CountRate(BaseModel):
    numerator: int
    denominator: int
    value: float | None


class UsageCost(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost: float | None
    priced: bool


class GenerationRates(BaseModel):
    """Доли с явными знаменателями. value is None при пустом знаменателе."""

    valid_generation_rate: CountRate
    schema_compliance_rate: CountRate
    seo_title_length_compliance: CountRate
    seo_description_length_compliance: CountRate
    search_queries_count_compliance: CountRate
    generation_error_rate: CountRate
    api_error_rate: CountRate
    model_error_rate: CountRate
    insufficient_data_rate: CountRate
    judge_error_rate: CountRate
    semantic_issue_rate: CountRate


def count_rate(numerator: int, denominator: int) -> CountRate:
    if denominator == 0:
        return CountRate(numerator=numerator, denominator=0, value=None)
    return CountRate(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator,
    )


def usage_cost(usages: list[LLMUsage], price: ModelPrice | None) -> UsageCost:
    prompt_tokens = sum(item.prompt_tokens for item in usages)
    completion_tokens = sum(item.completion_tokens for item in usages)
    if price is None or not usages:
        return UsageCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=None,
            priced=False,
        )
    cost = (
        prompt_tokens * price.prompt_per_million
        + completion_tokens * price.completion_per_million
    ) / 1_000_000
    return UsageCost(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=cost,
        priced=True,
    )


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
