# Social Media & Informal Text	Character substitution
# Social Media & Informal Text	Deliberate misspellings
# Social Media & Informal Text	Emoji substitution
# Social Media & Informal Text	Letter repetition for emphasis
# Social Media & Informal Text	Word concatenation
# Social Media & Informal Text	Word reordering
# Social Media & Informal Text	Colloquial
# Social Media & Informal Text	phonetic spelling

from typing import List, Optional

from attr import dataclass
from pyparsing import col

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

character_substitution = Perturbation(
    "Character substitution",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr, LANGS.zh, LANGS.fa],
    automatable=True,
    category="Social Media & Informal Text",
)
deliberate_misspellings = Perturbation(
    "Deliberate misspellings",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr, LANGS.zh, LANGS.fa],
    automatable=True,
    category="Social Media & Informal Text",
)

emoji_substitution = Perturbation(
    "Emoji substitution",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr, LANGS.zh, LANGS.fa],
    automatable=True,
    category="Social Media & Informal Text",
)
word_reordering = Perturbation(
    "Word reordering",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr, LANGS.zh, LANGS.fa],
    automatable=True,
    category="Social Media & Informal Text",
)
colloquial = Perturbation(
    "Colloquial",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr, LANGS.zh, LANGS.fa],
    automatable=False,
    category="Social Media & Informal Text",
)
phonetic_spelling = Perturbation(
    "Phonetic spelling",
    available_languages=[
        LANGS.en,
        LANGS.it,
    ],
    category="Social Media & Informal Text",
    automatable=True,
)
## tODO: chinese?? farsi??
letter_repetition_for_emphasis = Perturbation(
    "Letter repetition for emphasis",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr],
    automatable=True,
    category="Social Media & Informal Text",
)
word_concatenation = Perturbation(
    "Word concatenation",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr],
    automatable=True,
    category="Social Media & Informal Text",
)


SocialMediaInformalText = {
    "character_substitution": character_substitution,
    "deliberate_misspellings": deliberate_misspellings,
    "emoji_substitution": emoji_substitution,
    "letter_repetition_for_emphasis": letter_repetition_for_emphasis,
    "word_concatenation": word_concatenation,
}
