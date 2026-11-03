"""Явный запуск оценки генерации. Импорт модуля сеть не вызывает.

Одна модель:

    uv run python -m app.evals.generation_cli --model provider/model --judge-model provider/judge

Несколько моделей:

    uv run python -m app.evals.generation_cli --models provider/a,provider/b --judge-model judge
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.clients.llm import OpenAILLMClient
from app.core.config import settings
from app.evals.generation_dataset import load_generation_dataset
from app.evals.generation_runner import (
    GenerationComparison,
    default_report_dir,
    generation_prompt_hash,
    judge_prompt_hash,
    load_prices,
    run_generation_evaluation,
    write_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Оценка SEO-генератора на датасете")
    parser.add_argument("--model", action="append", default=[], help="Модель генерации")
    parser.add_argument(
        "--models",
        default="",
        help="Список моделей через запятую. Перекрывает EVAL_GENERATOR_MODELS.",
    )
    parser.add_argument("--judge-model", default="", help="Модель судьи, отличная от генератора")
    parser.add_argument("--prices", type=Path, default=None, help="JSON тарифов за 1M токенов")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Фиксированная температура. Не гарантирует одинаковые ответы.",
    )
    return parser


def selected_models(args: argparse.Namespace) -> list[str]:
    if args.models.strip():
        return [item.strip() for item in args.models.split(",") if item.strip()]
    if args.model:
        return [item.strip() for item in args.model if item.strip()]
    return [item.strip() for item in settings.eval_generator_models.split(",") if item.strip()]


def selected_judge(args: argparse.Namespace) -> str:
    return args.judge_model.strip() or settings.eval_judge_model.strip()


async def run_cli(args: argparse.Namespace) -> int:
    models = selected_models(args)
    judge_model = selected_judge(args)
    if not models or not judge_model:
        print(
            "Укажите модели генерации и судью: --model/--models и --judge-model "
            "или EVAL_GENERATOR_MODELS и EVAL_JUDGE_MODEL.",
            file=sys.stderr,
        )
        return 2
    api_key = settings.openai_api_key.strip()
    if not api_key or "your-api-key" in api_key:
        print("OPENAI_API_KEY не задан. Запросы к API не выполнялись.", file=sys.stderr)
        return 2

    cases = load_generation_dataset()
    prices = load_prices(args.prices) if args.prices else None
    report_dir = args.report_dir or default_report_dir()
    created = datetime.now(UTC)
    reports = []
    for model in models:
        generator = OpenAILLMClient(
            api_key=api_key,
            model=model,
            base_url=settings.openai_base_url,
            temperature=args.temperature,
        )
        judge = OpenAILLMClient(
            api_key=api_key,
            model=judge_model,
            base_url=settings.openai_base_url,
            temperature=args.temperature,
        )
        reports.append(
            await run_generation_evaluation(
                cases,
                generator,
                judge,
                model=model,
                judge_model=judge_model,
                temperature=args.temperature,
                prices=prices,
                base_url=settings.openai_base_url,
                report_dir=report_dir,
                created_at=created,
            )
        )
    comparison = GenerationComparison(
        created_at=created,
        dataset_hash=reports[0].dataset_hash,
        prompt_hash=generation_prompt_hash(),
        judge_prompt_hash=judge_prompt_hash(),
        judge_model=judge_model,
        temperature=args.temperature,
        models=models,
        reports=reports,
    )
    path = write_comparison(comparison, report_dir)
    print(path)
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run_cli(args)))


if __name__ == "__main__":
    main()
