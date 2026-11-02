"""Извлечение характеристик товара из текстового описания через LLM."""

from app.clients.llm import LLMClient
from app.schemas.product import ExtractionResult, ProductInput


class CharacteristicExtractor:
    """Извлекает и сверяет технические характеристики товара."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def extract(
        self,
        product: ProductInput,
        required_fields: list[str],
    ) -> ExtractionResult:
        """
        Вернуть характеристики из specs и description.

        Правила:
        - значения из specs сохраняются без изменения;
        - значения из description требуют подтверждающего фрагмента;
        - при противоречии источников фиксируется conflict, без тихого выбора;
        - неизвестные характеристики не угадываются.
        """
        raise NotImplementedError("CharacteristicExtractor.extract is not implemented yet")
