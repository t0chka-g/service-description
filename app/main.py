"""Основной модуль FastAPI"""

from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

STATIC_DIR = Path(__file__).resolve().parent / "static"

"""Создание FastAPI приложения"""
app = FastAPI(
    title="AI Product Card Generator",
    description=(
        "Сервис собирает SEO-карточку товара B2B-каталога по названию, "
        "производителю, характеристикам и техническому описанию. "
    ),
    version=version("service-description"),
    openapi_tags=[
        {
            "name": "products",
            "description": "Генерация карточки товара по подтвержденным данным.",
        },
        {
            "name": "system",
            "description": "Служебные проверки.",
        },
    ],
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
