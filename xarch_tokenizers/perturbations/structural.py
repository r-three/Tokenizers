# Structural Text Elements	Capitalization
# Structural Text Elements	Email addresses
# Structural Text Elements	Headers and section titles
# Structural Text Elements	List markers
# Structural Text Elements	Text decorations
# Structural Text Elements	Unicode formatting
# Structural Text Elements	Unusual formatting
# Structural Text Elements	URLs and file paths


import random
from typing import List, Optional

import numpy as np
import spacy
from attr import dataclass

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)
from xarch_tokenizers.perturbations.formatter import (
    TEXT_DECORATION_STYLE,
    UNICODE_FORMAT_STYLE,
    TextDecorationFormatter,
    UnicodeFormatter,
)


# Utility functions
def generate_capitalization_errors(
    question: str,
    language: Optional[LANGS] = LANGS.eng_Latn,
    lower_case_special_words: bool = True,
    inside_capitalization: bool = True,
    n_variations: int = 1,
    max_errors_in_one_sample: int = 1,
    sampling_rate=1.0,
):
    variants = set()
    perturbed_indices = set()

    def pick_index():
        ind_to_change = None
        while True:
            ind_to_change = np.random.randint(0, len(question))
            if (
                ind_to_change not in perturbed_indices
                or question[ind_to_change].isalpha()
            ):
                break
            if len(perturbed_indices) == len(question):
                break
        return ind_to_change

    if inside_capitalization:
        for _ in range(n_variations):
            ind_to_change = pick_index()
            if ind_to_change is None:
                break
            perturbed_indices.add(ind_to_change)
            char = question[ind_to_change]
            char = char.upper() if char.islower() else char.lower()
            variant = question[:ind_to_change] + char + question[ind_to_change + 1 :]
            variants.add(variant)
    for _ in range(n_variations):
        indx = np.random.choice(len(question), max_errors_in_one_sample, replace=False)
        indx.sort()
        variant = question
        for ind in indx:
            char = question[ind]
            char = char.upper() if char.islower() else char.lower()
            variant = variant[:ind] + char + variant[ind + 1 :]
        variants.add(variant)

    return list(variants)


def generate_formatted_variations(
    question: str,
    language: Optional[LANGS] = LANGS.eng_Latn,
    n_variations: int = 1,
) -> List[List[str]]:
    """
    Generate formatted variations for question
    """
    formatter = TextDecorationFormatter("ansi")
    variations = set()
    for _ in range(n_variations):
        style = random.choice(list(TEXT_DECORATION_STYLE))
        variant = formatter.format_text(question, style)
        variations.add(variant)

    return list(variations)


def generate_unicode_variations(
    question: str,
    language: Optional[LANGS] = LANGS.eng_Latn,
    n_variations: int = 1,
) -> List[List[str]]:
    """
    Generate Unicode variations for question
    """
    formatter = UnicodeFormatter()
    variations = set()
    for i in range(n_variations):
        # pick a random Unicode style
        style = random.choice(list(UNICODE_FORMAT_STYLE))
        # TODO: later only perturb part of it
        variant = formatter.format_text(question, style)
        variations.add(variant)

    return list(variations)


capitalization = Perturbation(
    "Capitalization",
    available_languages=[LANGS.eng_Latn, LANGS.ita_Latn, LANGS.tur_Latn],
    automatable=True,
    category="Structural Text Elements",
    func=generate_capitalization_errors,
)
email_addresses = Perturbation(
    "Email addresses",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Structural Text Elements",
)
headers_and_section_titles = Perturbation(
    "Headers and section titles",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=False,
    category="Structural Text Elements",
)
list_markers = Perturbation(
    "List markers",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=False,
    category="Structural Text Elements",
)
text_decorations = Perturbation(
    "Text decorations",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Structural Text Elements",
    func=generate_formatted_variations,
)
unicode_formatting = Perturbation(
    "Unicode formatting",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Structural Text Elements",
    func=generate_unicode_variations,
)
# ASCII art, tables, diagrams
unusual_formatting = Perturbation(
    "Unusual formatting",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Structural Text Elements",
)
urls_and_file_paths = Perturbation(
    "URLs and file paths",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=False,
    category="Structural Text Elements",
)

StructuralTextElements = {
    "capitalization": capitalization,
    "email_addresses": email_addresses,
    "headers_and_section_titles": headers_and_section_titles,
    "list_markers": list_markers,
    "text_decorations": text_decorations,
    "unicode_formatting": unicode_formatting,
    "unusual_formatting": unusual_formatting,
    "urls_and_file_paths": urls_and_file_paths,
}
