"""Основной модуль FastAPI"""
from importlib.metadata import version

from fastapi import FastAPI

from app.api.routes import router


"""Создание FastAPI приложения"""
app = FastAPI(
    title="AI Product Card Generator",
    description="Генерация SEO-карточек товаров B2B-интернет-магазина с помощью LLM",
    version=version("service-description"),
)

app.include_router(router)
