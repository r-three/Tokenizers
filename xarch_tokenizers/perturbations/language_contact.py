# Language Contact	Borrowing
# Language Contact	Calques
# Language Contact	Transliteration

from typing import List, Optional

from attr import dataclass

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

borrowing = Perturbation(
    "Borrowing",
    available_languages=[LANGS.eng_Latn],
    automatable=True,
    category="Language Contact",
)
calques = Perturbation(
    "Calques",
    available_languages=[LANGS.eng_Latn],
    automatable=True,
    category="Language Contact",
)
transliteration = Perturbation(
    "Transliteration",
    available_languages=[LANGS.eng_Latn],
    automatable=True,
    category="Language Contact",
)


LanguageContact = {
    "borrowing": borrowing,
    "calques": calques,
    "transliteration": transliteration,
}
