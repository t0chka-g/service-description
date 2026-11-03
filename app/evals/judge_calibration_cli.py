"""Явный запуск калибровки LLM-судьи. Импорт модуля сеть не вызывает.

Один сценарий:

    uv run python -m app.evals.judge_calibration_cli --judge-model provider/model \\
        --case cal_unit_substitution

Все сценарии датасета:

    uv run python -m app.evals.judge_calibration_cli --judge-model provider/model
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.clients.llm import OpenAILLMClient
from app.core.config import settings
from app.evals.judge_calibration import (
    default_calibration_report_dir,
    load_judge_calibration_dataset,
    load_prices,
    run_judge_calibration_report,
    select_calibration_cases,
    write_calibration_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Калибровка LLM-судьи на эталонных карточках"
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Модель судьи. Иначе EVAL_JUDGE_MODEL или OPENAI_MODEL.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Идентификатор сценария. Можно указать несколько раз.",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Список case_id через запятую.",
    )
    parser.add_argument("--prices", type=Path, default=None, help="JSON тарифов за 1M токенов")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Фиксированная температура. Не гарантирует одинаковые ответы.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Путь к JSON калибровочного датасета.",
    )
    return parser


def selected_judge(args: argparse.Namespace) -> str:
    return (
        args.judge_model.strip()
        or settings.eval_judge_model.strip()
        or settings.openai_model.strip()
    )


def selected_case_ids(args: argparse.Namespace) -> list[str]:
    if args.cases.strip():
        return [item.strip() for item in args.cases.split(",") if item.strip()]
    return [item.strip() for item in args.case if item.strip()]


async def run_cli(args: argparse.Namespace) -> int:
    judge_model = selected_judge(args)
    if not judge_model:
        print(
            "Укажите модель судьи: --judge-model или EVAL_JUDGE_MODEL / OPENAI_MODEL.",
            file=sys.stderr,
        )
        return 2
    api_key = settings.openai_api_key.strip()
    if not api_key or "your-api-key" in api_key:
        print("OPENAI_API_KEY не задан. Запросы к API не выполнялись.", file=sys.stderr)
        return 2

    try:
        cases = select_calibration_cases(
            load_judge_calibration_dataset(args.dataset),
            selected_case_ids(args) or None,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not cases:
        print("Калибровочный датасет пуст.", file=sys.stderr)
        return 2

    prices = load_prices(args.prices) if args.prices else None
    report_dir = args.report_dir or default_calibration_report_dir()
    judge = OpenAILLMClient(
        api_key=api_key,
        model=judge_model,
        base_url=settings.openai_base_url,
        temperature=args.temperature,
    )
    report = await run_judge_calibration_report(
        cases,
        judge,
        model=judge_model,
        temperature=args.temperature,
        prices=prices,
        base_url=settings.openai_base_url,
        report_dir=None,
        created_at=datetime.now(UTC),
    )
    path = write_calibration_report(report, report_dir)
    print(path)
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run_cli(args)))


if __name__ == "__main__":
    main()
