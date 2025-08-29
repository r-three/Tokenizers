# Structural Text Elements	Capitalization
# Structural Text Elements	Email addresses
# Structural Text Elements	Headers and section titles
# Structural Text Elements	List markers
# Structural Text Elements	Text decorations
# Structural Text Elements	Unicode formatting
# Structural Text Elements	Unusual formatting
# Structural Text Elements	URLs and file paths


from typing import List, Optional

from attr import dataclass

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

capitalization = Perturbation(
    "Capitalization",
    available_languages=[LANGS.en, LANGS.it, LANGS.tr],
    automatable=True,
    category="Structural Text Elements",
)
email_addresses = Perturbation(
    "Email addresses",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Structural Text Elements",
)
headers_and_section_titles = Perturbation(
    "Headers and section titles",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=False,
    category="Structural Text Elements",
)
list_markers = Perturbation(
    "List markers",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=False,
    category="Structural Text Elements",
)
text_decorations = Perturbation(
    "Text decorations",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Structural Text Elements",
)
unicode_formatting = Perturbation(
    "Unicode formatting",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Structural Text Elements",
)
unusual_formatting = Perturbation(
    "Unusual formatting",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Structural Text Elements",
)
urls_and_file_paths = Perturbation(
    "URLs and file paths",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
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
