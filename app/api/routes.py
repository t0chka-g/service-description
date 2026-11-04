"""HTTP-маршруты приложения и страница контент-менеджера."""

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse
from openai import APITimeoutError
from pydantic import BaseModel, Field

from app.api.dependencies import get_extractor, get_generator
from app.api.errors import TIMEOUT_DETAIL, ApiError, llm_http_exception
from app.api.examples import (
    COMPLETE_EXAMPLE,
    INSUFFICIENT_EXAMPLE,
    NEEDS_REVIEW_EXAMPLE,
    REQUEST_EXAMPLES,
)
from app.clients.llm import LLMError
from app.schemas.product import GenerationInput, GenerationResult, ProductInput
from app.services.extractor import CharacteristicExtractor
from app.services.generator import SEOGenerator

router = APIRouter()
INDEX_PATH = Path(__file__).resolve().parent.parent / "web" / "index.html"
logger = logging.getLogger("uvicorn.error")


class HealthStatus(BaseModel):
    """Ответ проверки работоспособности."""

    status: str = Field(description="Состояние процесса.", examples=["ok"])


@router.get(
    "/health",
    tags=["system"],
    summary="Проверка работоспособности",
    response_model=HealthStatus,
)
async def health() -> HealthStatus:
    """Проверка работоспособности API."""

    return HealthStatus(status="ok")


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Отдать страницу генерации карточки."""

    return FileResponse(INDEX_PATH)


@router.post(
    "/api/v1/products/generate",
    response_model=GenerationResult,
    tags=["products"],
    summary="Сгенерировать SEO-карточку товара",
    responses={
        200: {
            "description": (
                "Карточка собрана. Статусы needs_review и insufficient_data тоже "
                "возвращаются здесь: это частичный результат, а не ошибка запроса. "
                "У частичного результата seo_description равен null, причина лежит в warnings."
            ),
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/GenerationResult"},
                    "examples": {
                        "complete": {
                            "summary": "Полная демонстрационная карточка",
                            "description": (
                                "Синтетический пример. Не характеристики реального прибора."
                            ),
                            "value": COMPLETE_EXAMPLE,
                        },
                        "needs_review": {
                            "summary": "Частичный результат: описание не прошло проверку",
                            "description": (
                                "seo_description равен null. Причина указана в warnings, "
                                "характеристики сохранены."
                            ),
                            "value": NEEDS_REVIEW_EXAMPLE,
                        },
                        "insufficient_data": {
                            "summary": "Демонстрационный термометр, данных мало",
                            "value": INSUFFICIENT_EXAMPLE,
                        },
                    },
                }
            },
        },
        502: {
            "model": ApiError,
            "description": "Ответ языковой модели не прошел проверку контракта.",
        },
        503: {
            "model": ApiError,
            "description": "Языковая модель недоступна или запрос к провайдеру не выполнен.",
        },
        504: {
            "model": ApiError,
            "description": "Провайдер языковой модели не ответил вовремя.",
        },
    },
)
async def generate_product(
    product: Annotated[ProductInput, Body(openapi_examples=REQUEST_EXAMPLES)],
    extractor: Annotated[CharacteristicExtractor, Depends(get_extractor)],
    generator: Annotated[SEOGenerator, Depends(get_generator)],
) -> GenerationResult:
    """Собрать SEO-карточку по входным данным товара.

    Цепочка запроса: извлечение характеристик, проверка подтверждений,
    затем генерация текста. Имена для извлечения берутся из ключей specs.

    Ответы:
    - 200 — карточка (complete) или частичный результат (needs_review, insufficient_data);
    - 422 — тело запроса не соответствует схеме;
    - 502 — ответ модели не удалось принять;
    - 503 — провайдер модели недоступен;
    - 504 — провайдер не ответил вовремя.
    """

    try:
        return await _generate_card(product, extractor, generator)
    except LLMError as exc:
        logger.exception("Генерация отклонена: %s", exc)
        raise llm_http_exception(exc) from exc
    except (TimeoutError, APITimeoutError) as exc:
        logger.exception("Провайдер не ответил вовремя")
        raise HTTPException(status_code=504, detail=TIMEOUT_DETAIL) from exc


async def _generate_card(
    product: ProductInput,
    extractor: CharacteristicExtractor,
    generator: SEOGenerator,
) -> GenerationResult:
    prepared, fields = _prepare_product(product)
    extracted = await extractor.extract(prepared, fields)
    return await generator.generate(
        GenerationInput(
            name=prepared.name,
            manufacturer=prepared.manufacturer,
            attributes=extracted.attributes,
            missing_fields=extracted.missing_fields,
            conflicts=extracted.conflicts,
        )
    )


def _prepare_product(product: ProductInput) -> tuple[ProductInput, list[str]]:
    specs: dict[str, str] = {}
    for name, value in product.specs.items():
        key = name.strip()
        if not key or key in specs:
            continue
        specs[key] = value
    prepared = product.model_copy(
        update={
            "name": product.name.strip(),
            "manufacturer": product.manufacturer.strip(),
            "description": product.description.strip(),
            "specs": specs,
        }
    )
    return prepared, list(specs)
