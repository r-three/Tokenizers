# Script / Orthography	Abbreviations (with periods)
# Script / Orthography	Brand names with punctuation
# Script / Orthography	Code/language/script switching
# Script / Orthography	Diacritics presence/absence
# Script / Orthography	Equivalent expressions
# Script / Orthography	Historical spelling
# Script / Orthography	Homoglyphs
# Script / Orthography	Proper nouns with unusual capitalization
# Script / Orthography	Regional spelling variations
# Script / Orthography	Word Spacing/Zero-width characters/Extra Space
# Script / Orthography	Romanization

from typing import List, Optional

from attr import dataclass

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

# Homoglyphs for English
ENGLISH_HOMOGLYPHS = {
    "a": "аα",
    "A": "АΑ",
    "b": "Ь",
    "B": "В",
    "c": "с",
    "C": "С",
    "d": "ԁ",
    "e": "еэ",
    "E": "Е",
    "g": "ԍ",
    "h": "һ",
    "H": "Н",
    "i": "іι",
    "I": "І",
    "j": "ј",
    "J": "Ј",
    "k": "к",
    "K": "К",
    "l": "l1|",
    "m": "т",
    "M": "М",
    "n": "п",
    "N": "Ν",
    "o": "о0",
    "O": "О0",
    "p": "р",
    "P": "Р",
    "r": "г",
    "s": "ѕ",
    "S": "Ѕ",
    "t": "т",
    "T": "Т",
    "u": "υ",
    "v": "ν",
    "w": "ԝ",
    "x": "х",
    "X": "Х",
    "y": "у",
    "Y": "У",
    "z": "ᴢ",
}

MULTILINGUAL_HOMOGLYPHS = {
    LANGS.en: ENGLISH_HOMOGLYPHS,
}


def generate_homoglyph_errors(word, language: Optional[LANGS] = LANGS.en):
    """
    Generate possible homoglyph substitutions for a word.

    Args:
        word (str): Input word
        language (str): Language code

    Returns:
        list: List of possible homoglyph variants
    """
    if language not in MULTILINGUAL_HOMOGLYPHS:
        return [word]

    homoglyph_dict = MULTILINGUAL_HOMOGLYPHS[language]
    variants = set()

    for i, char in enumerate(word):
        if char in homoglyph_dict:
            for homoglyph in homoglyph_dict[char]:
                variant = word[:i] + homoglyph + word[i + 1 :]
                variants.add(variant)

    return list(variants)


abbreviations_with_periods = Perturbation(
    "Abbreviations (with periods)",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr],
    automatable=True,
    category="Script / Orthography",
)
brand_names_with_punctuation = Perturbation(
    "Brand names with punctuation",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr],
    automatable=True,
    category="Script / Orthography",
)
code_language_script_switching = Perturbation(
    "Code/language/script switching",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Script / Orthography",
)
diacritics_presence_absence = Perturbation(
    "Diacritics presence/absence",
    available_languages=[LANGS.en, LANGS.it, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Script / Orthography",
)
equivalent_expressions = Perturbation(
    "Equivalent expressions",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Script / Orthography",
)
historical_spelling = Perturbation(
    "Historical spelling",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Script / Orthography",
)
# TODO: not so much in turkish
homoglyphs = Perturbation(
    "Homoglyphs",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh],
    automatable=True,
    category="Script / Orthography",
    func=generate_homoglyph_errors,
)
proper_nouns_with_unusual_capitalization = Perturbation(
    "Proper nouns with unusual capitalization",
    available_languages=[LANGS.en, LANGS.it, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Script / Orthography",
)
regional_spelling_variations = Perturbation(
    "Regional spelling variations",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Script / Orthography",
)
word_spacing_zero_width_characters_extra_space = Perturbation(
    "Word Spacing/zero-width characters/extra space",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Script / Orthography",
)
romanization = Perturbation(
    "Romanization",
    available_languages=[LANGS.it, LANGS.zh, LANGS.fa],
    automatable=True,
    category="Script / Orthography",
)


ScriptOrthography = {
    "abbreviations_with_periods": abbreviations_with_periods,
    "brand_names_with_punctuation": brand_names_with_punctuation,
    "code_language_script_switching": code_language_script_switching,
    "diacritics_presence_absence": diacritics_presence_absence,
    "equivalent_expressions": equivalent_expressions,
    "historical_spelling": historical_spelling,
    "homoglyphs": homoglyphs,
    "proper_nouns_with_unusual_capitalization": proper_nouns_with_unusual_capitalization,
    "regional_spelling_variations": regional_spelling_variations,
    "word_spacing_zero_width_characters_extra_space": word_spacing_zero_width_characters_extra_space,
    "romanization": romanization,
}
