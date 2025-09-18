from .ambiguities import *
from .coding import *
from .common import *
from .informal import *
from .language_contact import *
from .math_science import *
from .morphological import *
from .named_entities import *
from .noise import *
from .script_orthography import *
from .structural import *

ALL_SUBCATEGORIES = [
    "cultural_references",
    "domain_specific_punctuation",
    "quotes_and_parentheses",
    "rare_n_grams",
    "sentence_boundaries",
    "wiki_jargon",
    "character_substitution",
    "deliberate_misspellings",
    "emoji_substitution",
    "word_reordering",
    "colloquial",
    "phonetic_spelling",
    "letter_repetition_for_emphasis",
    "word_concatenation",
    "borrowing",
    "calques",
    "transliteration",
    "chemical_formulas",
    "equations_with_mixed_notation",
    "numerical_formats",
    "scientific_notation",
    "unit_combinations",
    "clitics",
    "compounds",
    "contractions",
    "affixation_edge_cases",
    "derivations",
    "inflections",
    "academic_citations",
    "place_names_with_apostrophes",
    "special_names_across_cultures",
    "technical_product_names",
    "keyboard_proximity_errors",
    "noise_injection",
    "ocr_errors",
    "permutations",
    "typographical_errors",
    "abbreviations_with_periods",
    "brand_names_with_punctuation",
    "code_language_script_switching",
    "diacritics_presence_absence",
    "equivalent_expressions",
    "historical_spelling",
    "homoglyphs",
    "proper_nouns_with_unusual_capitalization",
    "regional_spelling_variations",
    "word_spacing_zero_width_characters_extra_space",
    "romanization",
    "capitalization",
    "email_addresses",
    "headers_and_section_titles",
    "list_markers",
    "text_decorations",
    "unicode_formatting",
    "unusual_formatting",
    "urls_and_file_paths",
]

PERTURBATION_MAPPINGS = (
    ContextDependentAmbiguities
    | SocialMediaInformalText
    | LanguageContact
    | MathematicalScientificNotation
    | MorphologicalChallenges
    | NamedEntities
    | Noise
    | ScriptOrthography
    | StructuralTextElements
)
# Context-Dependent Ambiguities	Cultural references
# Context-Dependent Ambiguities	Domain-specific punctuation
# Context-Dependent Ambiguities	Quotes & parentheses
# Context-Dependent Ambiguities	Rare n-grams
# Context-Dependent Ambiguities	Sentence boundaries
# Context-Dependent Ambiguities	Wiki-jargon
# Language Contact	Borrowing
# Language Contact	Calques
# Language Contact	Transliteration
# Mathematical & Scientific Notation	Chemical formulas
# Mathematical & Scientific Notation	Equations with mixed notation
# Mathematical & Scientific Notation	Numerical formats
# Mathematical & Scientific Notation	Scientific notation
# Mathematical & Scientific Notation	Unit combinations
# 	Grammatical errors
# Morphological Challenges	Clitics
# Morphological Challenges	Compounds
# Morphological Challenges	Contractions
# Named Entities	Technical product names
# Noise	Keyboard proximity errors
# Noise	Noise injection
# Noise	OCR Errors
# Noise	Permutations
# Noise	Typographical errors
# Script / Orthography	Abbreviations (with periods)
# Script / Orthography	Brand names with punctuation
# Script / Orthography	Equivalent expressions
# Script / Orthography	Historical spelling
# Script / Orthography	Homoglyphs
# Script / Orthography	Proper nouns with unusual capitalization
# Structural Text Elements	Capitalization
# Script / Orthography	Regional spelling variations
# Script / Orthography	Word Spacing/Zero-width characters/Extra Space
# Social Media & Informal Text	Character substitution
# Social Media & Informal Text	Deliberate misspellings
# Social Media & Informal Text	Emoji substitution
# Social Media & Informal Text	Letter repetition for emphasis
# Social Media & Informal Text	Word concatenation
# Social Media & Informal Text	Word reordering
# Social Media & Informal Text	Colloquial
# Social Media & Informal Text	phonetic spelling
# Structural Text Elements	Email addresses
# Structural Text Elements	Headers and section titles
# Structural Text Elements	List markers
# Structural Text Elements	Text decorations
# Structural Text Elements	Unicode formatting
# Structural Text Elements	Unusual formatting
# Structural Text Elements	URLs and file paths
