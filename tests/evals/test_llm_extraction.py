"""Реальная оценка Extractor.

Запуск: uv run pytest -m llm -v

Тест не задает порог качества. Успешный прогон значит, что каждый сценарий
получил ответ и отчет записан в reports/evals/. Метрики в отчете — результат
этого запуска, а не заранее подтвержденное качество модели.
Без OPENAI_API_KEY сетевой вызов не выполняется.
"""

import pytest

from app.clients.llm import OpenAILLMClient
from app.core.config import settings
from app.evals.models import ScenarioStatus
from app.evals.runner import default_report_dir, load_cases, run_evaluation


@pytest.mark.llm
async def test_llm_extraction_writes_report() -> None:
    """Сравнить Extractor с ground truth на живой модели и сохранить JSON-отчет."""

    api_key = settings.openai_api_key.strip()
    if not api_key or "your-api-key" in api_key:
        pytest.skip(
            "OPENAI_API_KEY is not set. Real LLM evaluation was not executed, "
            "and model quality is not confirmed."
        )

    cases = load_cases()
    client = OpenAILLMClient(
        api_key=api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
    )
    report = await run_evaluation(
        cases,
        client,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        report_dir=default_report_dir(),
    )

    failures = [
        f"{scenario.case_id} [{scenario.status}]: {scenario.error_message}"
        for scenario in report.scenarios
        if scenario.status is not ScenarioStatus.SCORED
    ]
    assert not failures, "evaluation did not score every scenario:\n" + "\n".join(failures)
    assert report.metrics is not None
    assert report.scenario_count == len(cases)
    assert report.scored_count == len(cases)
