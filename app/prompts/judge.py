"""Промпт независимого судьи SEO-карточки.

Судья видит только фактический вход и текст карточки.
Идентификаторы сценариев, запрещенные формулировки, допустимые
обобщения и ожидаемые verdict в промпт не входят.
"""

from app.evals.generation_dataset import GenerationDatasetCase
from app.schemas.product import GenerationResult


def build_judge_prompt(case: GenerationDatasetCase, card: GenerationResult) -> str:
    """Собрать задание судьи без готовой разметки оценки."""

    lines = [
        "Ты независимый проверяющий SEO-карточки B2B-товара на русском языке.",
        "Оцени только переданные факты и текст карточки.",
        "Не дополняй карточку и не исправляй ее.",
        "",
        "Различай четыре вида утверждений:",
        "- confirmed_fact: утверждение прямо следует из подтвержденных характеристик,",
        "  названия или производителя.",
        "- acceptable_generalization: осторожное обобщение без нового числа,",
        "  единицы, сертификата, гарантии или коммерческого преимущества.",
        "- unsupported_claim: утверждение не следует из входа.",
        "  Отсутствие сведения не делает утверждение ложным, но и не подтверждает его.",
        "- direct_contradiction: утверждение противоречит подтвержденному значению,",
        "  называет альтернативу из другого источника как единственный факт",
        "  или подменяет характеристику.",
        "",
        "Проверь фактическую достоверность:",
        "числа, единицы, привязку характеристик, выдуманные возможности,",
        "сертификаты, стандарты, гарантии и коммерческие преимущества.",
        "Проверь SEO: релевантность заголовка и описания, keyword stuffing,",
        "бессмысленные повторы, релевантность запросов.",
        "Проверь язык: грамотность, тон B2B, воду, рекламные преувеличения,",
        "пригодность текста для контент-менеджера.",
        "",
        "Оценки — целые числа от 0 до 5.",
        "5 значит, что замечаний этой группы нет.",
        "Прямое противоречие или выдуманное число снижает factual_consistency_score.",
        "Каждое неподтвержденное утверждение и каждое противоречие укажи",
        "дословной цитатой из карточки и объяснением, чего не хватает во входе.",
        "verdict: pass, если карточка опирается на факты и допустимые обобщения;",
        "fail, если есть прямое противоречие или критическая фактическая ошибка;",
        "review, если вывод спорный и его нужно проверить человеку.",
        "Твоя оценка не является истиной и может быть пересмотрена вручную.",
        "",
        f"Название: {case.input.name}",
        f"Производитель: {case.input.manufacturer}",
        "Подтвержденные факты:",
    ]
    if case.confirmed_facts:
        lines.extend(f"- {fact.name} = {fact.value}" for fact in case.confirmed_facts)
    else:
        lines.append("- нет")

    lines.append("Сведения, отсутствующие во входных данных:")
    if case.missing_data:
        lines.extend(f"- {item}" for item in case.missing_data)
    else:
        lines.append("- нет")

    lines.append("Расхождения в исходных данных:")
    if case.conflicting_values:
        for item in case.conflicting_values:
            lines.append(
                f"- {item.name}: в подтвержденных данных «{item.confirmed_value}»; "
                f"в другом источнике встречается «{item.rejected_value}»."
            )
    else:
        lines.append("- нет")

    lines.extend(
        [
            "",
            "Карточка:",
            f"seo_title: {card.seo_title}",
            f"seo_description: {card.seo_description}",
            "search_queries:",
        ]
    )
    if card.search_queries:
        lines.extend(f"- {query}" for query in card.search_queries)
    else:
        lines.append("- нет")
    lines.append(f"commercial_block: {card.commercial_block}")
    return "\n".join(lines)
