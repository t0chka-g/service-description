"""Генерация SEO-контента карточки товара по проверенному результату Extractor."""

from app.clients.llm import LLMClient
from app.schemas.product import GenerationInput, GenerationResult


class SEOGenerator:
    """Черновик SEO-карточки из подтвержденных характеристик.

    Генератор не читает исходное техническое описание и не извлекает
    характеристики заново. Контракт поведения зафиксирован в
    tests/unit/test_generator.py. Реализация появится на шаге GREEN.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def generate(self, data: GenerationInput) -> GenerationResult:
        """Собрать карточку. Пока намеренно не реализовано."""

        raise NotImplementedError("SEOGenerator.generate is not implemented")
