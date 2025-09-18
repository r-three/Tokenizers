# Mathematical & Scientific Notation	Chemical formulas
# Mathematical & Scientific Notation	Equations with mixed notation
# Mathematical & Scientific Notation	Numerical formats
# Mathematical & Scientific Notation	Scientific notation
# Mathematical & Scientific Notation	Unit combinations

from typing import List, Optional

from attr import dataclass

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

chemical_formulas = Perturbation(
    "Chemical formulas",
    available_languages=[LANGS.eng_Latn],
    category="Mathematical & Scientific Notation",
)
equations_with_mixed_notation = Perturbation(
    "Equations with mixed notation",
    available_languages=[LANGS.eng_Latn],
    category="Mathematical & Scientific Notation",
)
numerical_formats = Perturbation(
    "Numerical formats",
    available_languages=[LANGS.eng_Latn],
    category="Mathematical & Scientific Notation",
)
scientific_notation = Perturbation(
    "Scientific notation",
    available_languages=[LANGS.eng_Latn],
    category="Mathematical & Scientific Notation",
)
unit_combinations = Perturbation(
    "Unit combinations",
    available_languages=[LANGS.eng_Latn],
    category="Mathematical & Scientific Notation",
)


MathematicalScientificNotation = {
    "chemical_formulas": chemical_formulas,
    "equations_with_mixed_notation": equations_with_mixed_notation,
    "numerical_formats": numerical_formats,
    "scientific_notation": scientific_notation,
    "unit_combinations": unit_combinations,
}
