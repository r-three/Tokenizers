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

import random
from typing import List, Optional

import numpy as np
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
    LANGS.eng_Latn: ENGLISH_HOMOGLYPHS,
}


def alpha_indices(s: str) -> List[int]:
    return [i for i, c in enumerate(s) if c.isalpha()]


def generate_homoglyph_errors(
    question: str,
    language: Optional[LANGS] = LANGS.eng_Latn,
    n_variations: int = 1,
    max_errors: Optional[int] = None,
):
    """
    Generate possible homoglyph substitutions for a question.
    """
    if language not in MULTILINGUAL_HOMOGLYPHS:
        return [question]

    homoglyph_dict = MULTILINGUAL_HOMOGLYPHS[language]
    if max_errors is None:
        max_errors = len([i for i, c in enumerate(question) if c in homoglyph_dict])

    variants = set()

    def get_indices() -> List[int]:
        idx = [i for i, c in enumerate(question) if c in homoglyph_dict]
        return np.random.choice(idx, size=(max_errors,), replace=False)

    for _ in range(n_variations):
        variant = question
        idx = get_indices()
        for i in idx:
            variant = variant[:i] + homoglyph_dict[question[i]] + variant[i + 1 :]
        variants.add(variant)

    return list(variants)


def generate_diacritic_perturbations(
    question: str,
    language: Optional[LANGS] = LANGS.eng_Latn,
    n_variations: int = 1,
    max_errors: Optional[int] = None,
):
    """
    Generate variations by adding/removing/changing diacritical marks

    Args:
        question: Input text
        language: Language code (affects which diacritics are used)
        n_variations: Number of variations to generate
        max_errors: Maximum number of diacritic changes per variation.
                   Pass None to switch all possible diacritics.
    """

    # Base character to diacritic mappings by language
    diacritic_maps = {
        LANGS.eng_Latn: {
            # English - common borrowed diacritics
            "a": ["à", "á", "â", "ã", "ä", "å", "ā", "ă", "ą"],
            "e": ["è", "é", "ê", "ë", "ē", "ĕ", "ė", "ę", "ě"],
            "i": ["ì", "í", "î", "ï", "ī", "ĭ", "į", "ı"],
            "o": ["ò", "ó", "ô", "õ", "ö", "ō", "ŏ", "ő", "ø"],
            "u": ["ù", "ú", "û", "ü", "ū", "ŭ", "ů", "ű", "ų"],
            "n": ["ñ", "ń", "ň", "ņ"],
            "c": ["ç", "ć", "č", "ĉ", "ċ"],
            "s": ["š", "ś", "ş", "ŝ"],
            "z": ["ž", "ź", "ż"],
            "A": ["À", "Á", "Â", "Ã", "Ä", "Å", "Ā", "Ă", "Ą"],
            "E": ["È", "É", "Ê", "Ë", "Ē", "Ĕ", "Ė", "Ę", "Ě"],
            "I": ["Ì", "Í", "Î", "Ï", "Ī", "Ĭ", "Į", "İ"],
            "O": ["Ò", "Ó", "Ô", "Õ", "Ö", "Ō", "Ŏ", "Ő", "Ø"],
            "U": ["Ù", "Ú", "Û", "Ü", "Ū", "Ŭ", "Ů", "Ű", "Ų"],
            "N": ["Ñ", "Ń", "Ň", "Ņ"],
            "C": ["Ç", "Ć", "Č", "Ĉ", "Ċ"],
            "S": ["Š", "Ś", "Ş", "Ŝ"],
            "Z": ["Ž", "Ź", "Ż"],
        },
        LANGS.tur_Latn: {
            # Turkish specific
            "a": ["à", "á", "â", "ã", "ä", "å"],
            "e": ["è", "é", "ê", "ë"],
            "i": ["ì", "í", "î", "ï", "ı"],
            "o": ["ò", "ó", "ô", "õ", "ö"],
            "u": ["ù", "ú", "û", "ü"],
            "c": ["ç"],
            "s": ["ş"],
            "g": ["ğ"],
            "I": ["İ", "Ì", "Í", "Î", "Ï"],
            "A": ["À", "Á", "Â", "Ã", "Ä", "Å"],
            "E": ["È", "É", "Ê", "Ë"],
            "O": ["Ò", "Ó", "Ô", "Õ", "Ö"],
            "U": ["Ù", "Ú", "Û", "Ü"],
            "C": ["Ç"],
            "S": ["Ş"],
            "G": ["Ğ"],
        },
        LANGS.pes_Arab: {
            # Persian - adding common Arabic diacritics
            "ا": ["آ", "أ", "إ"],
            "ه": ["ة"],
            "ی": ["ي"],
            "ك": ["ک"],
        },
        LANGS.ita_Latn: {
            # Italian
            "a": ["à", "á"],
            "e": ["è", "é"],
            "i": ["ì", "í"],
            "o": ["ò", "ó"],
            "u": ["ù", "ú"],
            "A": ["À", "Á"],
            "E": ["È", "É"],
            "I": ["Ì", "Í"],
            "O": ["Ò", "Ó"],
            "U": ["Ù", "Ú"],
        },
    }

    # Reverse mapping - diacritic to base character (for removal)
    reverse_maps = {}
    for lang, char_map in diacritic_maps.items():
        reverse_maps[lang] = {}
        for base_char, diacritics in char_map.items():
            for diacritic in diacritics:
                reverse_maps[lang][diacritic] = base_char

    def get_changeable_positions(text: str):
        """Get positions where diacritics can be added or removed"""
        positions = []
        lang_map = diacritic_maps.get(language, diacritic_maps[LANGS.eng_Latn])
        reverse_map = reverse_maps.get(language, reverse_maps[LANGS.eng_Latn])

        for i, char in enumerate(text):
            # Can add diacritic to base character
            if char in lang_map:
                positions.append(("add", i, char))
            # Can remove diacritic from accented character
            elif char in reverse_map:
                positions.append(("remove", i, char))

        return positions

    def apply_diacritic_change(text: str, change_type: str, pos: int, char: str):
        """Apply a single diacritic change"""
        lang_map = diacritic_maps.get(language, diacritic_maps[LANGS.eng_Latn])
        reverse_map = reverse_maps.get(language, reverse_maps[LANGS.eng_Latn])

        if change_type == "add" and char in lang_map:
            # Replace base character with random diacritic version
            new_char = random.choice(lang_map[char])
            return text[:pos] + new_char + text[pos + 1 :]
        elif change_type == "remove" and char in reverse_map:
            # Replace diacritic character with base character
            new_char = reverse_map[char]
            return text[:pos] + new_char + text[pos + 1 :]

        return text

    variations = set()
    attempts = 0
    max_attempts = n_variations * 20

    while len(variations) < n_variations and attempts < max_attempts:
        attempts += 1
        text_variant = question
        changeable_positions = get_changeable_positions(text_variant)

        if not changeable_positions:
            variations.add(question)  # No changes possible
            continue

        # Determine number of changes
        if max_errors is None:
            # Change all possible positions
            num_changes = len(changeable_positions)
        else:
            num_changes = min(max_errors, len(changeable_positions))
            num_changes = random.randint(1, num_changes)

        # Apply changes
        selected_positions = random.sample(changeable_positions, num_changes)

        # Sort by position in reverse order to maintain indices
        selected_positions.sort(key=lambda x: x[1], reverse=True)

        for change_type, pos, char in selected_positions:
            text_variant = apply_diacritic_change(text_variant, change_type, pos, char)

        variations.add(text_variant)

    return list(variations)


# =============================================================================
# ZERO-WIDTH PERTURBATIONS
# =============================================================================


def generate_zero_width_perturbations(
    question: str,
    language: Optional[LANGS] = LANGS.eng_Latn,
    n_variations: int = 1,
    max_insertions: int = 3,
    insertion_probability: float = 0.3,
):
    """
    Generate variations by inserting zero-width characters

    Args:
        question: Input text
        language: Language code (not used but kept for consistency)
        n_variations: Number of variations to generate
        max_insertions: Maximum number of zero-width characters per variation
        insertion_probability: Probability of inserting at each valid position
    """

    # Zero-width characters
    zero_width_chars = [
        "\u200b",  # Zero Width Space (ZWSP)
        "\u200c",  # Zero Width Non-Joiner (ZWNJ)
        "\u200d",  # Zero Width Joiner (ZWJ)
        "\u2060",  # Word Joiner (WJ)
        "\ufeff",  # Zero Width No-Break Space (BOM)
        "\u200e",  # Left-to-Right Mark (LRM)
        "\u200f",  # Right-to-Left Mark (RLM)
        "\u061c",  # Arabic Letter Mark (ALM)
    ]

    def get_insertion_positions(text: str):
        """Get valid positions for zero-width character insertion"""
        positions = []

        for i in range(len(text) + 1):
            # Can insert at beginning, end, or between characters
            # Skip positions immediately before/after spaces for readability
            if i == 0 or i == len(text):
                positions.append(i)
            elif i > 0 and i < len(text):
                prev_char = text[i - 1]
                next_char = text[i] if i < len(text) else " "

                # Don't insert right after space or before space
                if prev_char != " " and next_char != " ":
                    positions.append(i)
                # Do insert between word characters
                elif prev_char.isalnum() and next_char.isalnum():
                    positions.append(i)

        return positions

    def insert_zero_width_chars(text: str, positions: List[int], chars: List[str]):
        """Insert zero-width characters at specified positions"""
        # Sort positions in reverse order to maintain indices
        sorted_data = sorted(zip(positions, chars), key=lambda x: x[0], reverse=True)

        result = text
        for pos, char in sorted_data:
            result = result[:pos] + char + result[pos:]

        return result

    variations = set()
    attempts = 0
    max_attempts = n_variations * 20

    while len(variations) < n_variations and attempts < max_attempts:
        attempts += 1
        insertion_positions = get_insertion_positions(question)

        if not insertion_positions:
            variations.add(question)
            continue

        # Decide how many insertions to make
        num_insertions = random.randint(
            1, min(max_insertions, len(insertion_positions))
        )

        # Select positions based on probability
        selected_positions = []
        available_positions = insertion_positions.copy()

        for _ in range(num_insertions):
            if not available_positions:
                break

            if random.random() < insertion_probability:
                pos = random.choice(available_positions)
                selected_positions.append(pos)
                available_positions.remove(pos)

        if not selected_positions:
            # Force at least one insertion
            selected_positions = [random.choice(insertion_positions)]

        # Select zero-width characters for each position
        selected_chars = [random.choice(zero_width_chars) for _ in selected_positions]

        # Apply insertions
        variant = insert_zero_width_chars(question, selected_positions, selected_chars)
        variations.add(variant)

    return list(variations)


abbreviations_with_periods = Perturbation(
    "Abbreviations (with periods)",
    available_languages=[LANGS.eng_Latn, LANGS.ita_Latn, LANGS.tur_Latn],
    automatable=True,
    category="Script / Orthography",
)
brand_names_with_punctuation = Perturbation(
    "Brand names with punctuation",
    available_languages=[LANGS.eng_Latn, LANGS.ita_Latn, LANGS.tur_Latn],
    automatable=True,
    category="Script / Orthography",
)
code_language_script_switching = Perturbation(
    "Code/language/script switching",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Script / Orthography",
)
diacritics_presence_absence = Perturbation(
    "Diacritics presence/absence",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Script / Orthography",
    func=generate_diacritic_perturbations,
)
equivalent_expressions = Perturbation(
    "Equivalent expressions",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Script / Orthography",
)
historical_spelling = Perturbation(
    "Historical spelling",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Script / Orthography",
)
# TODO: not so much in turkish
homoglyphs = Perturbation(
    "Homoglyphs",
    available_languages=[LANGS.eng_Latn, LANGS.ita_Latn, LANGS.zho_Hans],
    automatable=True,
    category="Script / Orthography",
    func=generate_homoglyph_errors,
)
proper_nouns_with_unusual_capitalization = Perturbation(
    "Proper nouns with unusual capitalization",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Script / Orthography",
)
regional_spelling_variations = Perturbation(
    "Regional spelling variations",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Script / Orthography",
)
word_spacing_zero_width_characters_extra_space = Perturbation(
    "Word Spacing/zero-width characters/extra space",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Script / Orthography",
    func=generate_zero_width_perturbations,
)
romanization = Perturbation(
    "Romanization",
    available_languages=[LANGS.ita_Latn, LANGS.zho_Hans, LANGS.pes_Arab],
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
