# Named Entities	Academic citations
# Named Entities	Place names with apostrophes
# Named Entities	Special names across cultures
# Named Entities	Technical product names

from typing import List, Optional

from attr import dataclass

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

academic_citations = Perturbation(
    "Academic Citations",
    available_languages=[LANGS.eng_Latn],
    automatable=False,
    category="Named Entities",
)
place_names_with_apostrophes = Perturbation(
    "Place names with apostrophes",
    available_languages=[LANGS.eng_Latn],
    automatable=True,
    category="Named Entities",
)
special_names_across_cultures = Perturbation(
    "Special names across cultures",
    available_languages=[LANGS.eng_Latn],
    automatable=False,
    category="Named Entities",
)
technical_product_names = Perturbation(
    "Technical product names",
    available_languages=[LANGS.eng_Latn],
    automatable=True,
    category="Named Entities",
)


NamedEntities = {
    "academic_citations": academic_citations,
    "place_names_with_apostrophes": place_names_with_apostrophes,
    "special_names_across_cultures": special_names_across_cultures,
    "technical_product_names": technical_product_names,
}
