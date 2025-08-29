# Context-Dependent Ambiguities	Cultural references
# Context-Dependent Ambiguities	Domain-specific punctuation
# Context-Dependent Ambiguities	Quotes & parentheses
# Context-Dependent Ambiguities	Rare n-grams
# Context-Dependent Ambiguities	Sentence boundaries
# Context-Dependent Ambiguities	Wiki-jargon

from typing import List, Optional

from attr import dataclass

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

cultural_references = Perturbation(
    "Cultural references",
    available_languages=[LANGS.en],
    automatable=False,
    category="Context-Dependent Ambiguities",
)
domain_specific_punctuation = Perturbation(
    "Domain-Specific Punctuation",
    available_languages=[LANGS.en],
    automatable=False,
    category="Context-Dependent Ambiguities",
)
quotes_and_parentheses = Perturbation(
    "Quotes & Parentheses",
    available_languages=[LANGS.en],
    automatable=True,
    category="Context-Dependent Ambiguities",
)
rare_n_grams = Perturbation(
    "Rare n-grams",
    available_languages=[LANGS.en],
    automatable=True,
    category="Context-Dependent Ambiguities",
)
sentence_boundaries = Perturbation(
    "Sentence boundaries",
    available_languages=[LANGS.en],
    automatable=False,
    category="Context-Dependent Ambiguities",
)
wiki_jargon = Perturbation(
    "Wiki-jargon",
    available_languages=[LANGS.en],
    automatable=False,
    category="Context-Dependent Ambiguities",
)


ContextDependentAmbiguities = {
    "cultural_references": cultural_references,
    "domain_specific_punctuation": domain_specific_punctuation,
    "quotes_and_parentheses": quotes_and_parentheses,
    "rare_n_grams": rare_n_grams,
    "sentence_boundaries": sentence_boundaries,
    "wiki_jargon": wiki_jargon,
}
