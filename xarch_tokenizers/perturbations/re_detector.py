import re
import string
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter, defaultdict

import ujson

PATTERNS = {
    # === CONTRACTIONS & ABBREVIATIONS ===
    "contractions": re.compile(r"\b\w+[''](?:t|re|ve|ll|d|s|m|n)\b", re.IGNORECASE),
    "contractions_expanded": re.compile(
        r"\b(?:do not|will not|cannot|would not|should not|could not)\b", re.IGNORECASE
    ),
    "abbreviations_with_periods": re.compile(
        r"\b[A-Z]\.(?:[A-Z]\.)+\b"
    ),  # U.S.A., Ph.D.
    "abbreviations_without_periods": re.compile(r"\b[A-Z]{2,}\b"),  # USA, PhD
    # === EMAILS, URLS, TECHNICAL ===
    # "emails": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    # "urls": re.compile(r'https?://[^\s<>"\']+'),
    # "file_paths_unix": re.compile(r"/(?:[^/\s]+/)*[^/\s]*"),
    # "file_paths_windows": re.compile(r"[A-Za-z]:\\(?:[^\\:\s]+\\)*[^\\:\s]*"),
    # "ip_addresses": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    # === CODE CONVENTIONS ===
    "camelCase": re.compile(r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b"),
    "PascalCase": re.compile(r"\b[A-Z][a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b"),
    "snake_case": re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),
    "SCREAMING_SNAKE_CASE": re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"),
    "kebab-case": re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b"),
    # === CODE SYNTAX ===
    "code_comments_cpp": re.compile(r"//[^\n]*"),
    "code_comments_python": re.compile(r"#[^\n]*"),
    "code_comments_multiline": re.compile(r"/\*.*?\*/", re.DOTALL),
    "code_comments_sql": re.compile(r"--[^\n]*"),
    "string_literals_double": re.compile(r'"(?:[^"\\]|\\.)*"'),
    "string_literals_single": re.compile(r"'(?:[^'\\]|\\.)*'"),
    "escape_sequences": re.compile(r'\\[nrtbfav\'"\\]'),
    "hex_values": re.compile(r"\b0x[0-9a-fA-F]+\b"),
    "binary_values": re.compile(r"\b0b[01]+\b"),
    # === SCIENTIFIC NOTATION ===
    "scientific_standard": re.compile(r"\b\d+\.?\d*[eE][+-]?\d+\b"),
    "scientific_unicode_mult": re.compile(r"\b\d+\.?\d*\s*×\s*10[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+\b"),
    "scientific_caret": re.compile(r"\b\d+\.?\d*\s*[×*x]\s*10\^[+-]?\d+\b"),
    "scientific_superscript": re.compile(r"\b10[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+\b"),
    "scientific_latex": re.compile(r"\b\d+\.?\d*\s*\\times\s*10\^?\{?[+-]?\d+\}?\b"),
    # === UNITS & MEASUREMENTS ===
    "units_with_superscript": re.compile(
        r"\b(?:m|km|ft|kg)[²³⁴⁵⁶⁷⁸⁹]?(?:/s[²³⁴⁵⁶⁷⁸⁹]?)?"
    ),
    "units_with_numbers": re.compile(r"\b(?:m|km|ft|kg)[2-9]?(?:/s[2-9]?)?"),
    "units_with_caret": re.compile(r"\b(?:m|km|ft|kg)\^[2-9](?:/s\^[2-9])?"),
    "temperature_units": re.compile(
        r"\b\d+\s*°?[CF]\b|\b\d+\s*K\b|\b\d+\s*deg\s*[CF]\b"
    ),
    # === CHEMICAL FORMULAS ===
    "chemical_subscript_unicode": re.compile(r"\b[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉]+\b"),
    "chemical_subscript_numeric": re.compile(r"\b[A-Z][a-z]?\d+\b"),
    "chemical_subscript_underscore": re.compile(r"\b[A-Z][a-z]?_\d+\b"),
    "chemical_ions": re.compile(r"\b[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉]*[⁺⁻]+\b"),
    "chemical_common_h2o": re.compile(r"\bH[₂2_2]O\b"),
    "chemical_common_co2": re.compile(r"\bCO[₂2_2]\b"),
    # === NUMERICAL FORMATS ===
    "numbers_indian_format": re.compile(
        r"\b\d{1,2},\d{2},\d{3}(?:,\d{3})*\b"
    ),  # 1,23,456
    "numbers_western_format": re.compile(r"\b\d{1,3}(?:,\d{3})+\b"),  # 123,456
    "numbers_european_format": re.compile(r"\b\d{1,3}(?:\.\d{3})+,\d+\b"),  # 123.456,78
    "numbers_space_separated": re.compile(r"\b\d{1,3}(?: \d{3})+\b"),  # 123 456
    "decimal_comma": re.compile(r"\b\d+,\d+\b"),  # European decimal comma
    "decimal_point": re.compile(r"\b\d+\.\d+\b"),  # Standard decimal point
    # === DATE & TIME FORMATS ===
    "date_dmy": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),  # DD/MM/YYYY
    "date_mdy": re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    ),  # MM/DD/YYYY (same pattern)
    "date_iso": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # YYYY-MM-DD
    "date_dot_separated": re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"),  # DD.MM.YYYY
    "time_24h": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),  # HH:MM or HH:MM:SS
    "time_12h": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*[AaPp][Mm]\b"),  # 12:00 PM
    "timezone_abbrev": re.compile(r"\b[A-Z]{3,4}\b(?=\s|$)"),  # EST, UTC, etc.
    # === TYPOGRAPHICAL ERRORS ===
    "char_repetition": re.compile(r"\b\w*([a-zA-Z])\1{2,}\w*\b"),  # sooo, goood
    "double_spaces": re.compile(r"  +"),
    "missing_space_punct": re.compile(r"[.!?][a-zA-Z]"),
    "extra_space_punct": re.compile(r" +[.!?,:;]"),
    "multiple_punctuation": re.compile(r"[.!?]{2,}|[,;:]{2,}"),
    "mixed_case_words": re.compile(r"\b[a-z][A-Z][a-z]+\b|\b[A-Z][a-z][A-Z][a-z]+\b"),
    "missing_capitalization": re.compile(r"[.!?]\s+[a-z]"),
    "all_caps_words": re.compile(r"\b[A-Z]{3,}\b"),
    # === UNICODE & ENCODING ===
    "unicode_quotes_curly": re.compile(r'[""' "‚„‹›«»]"),
    "unicode_quotes_straight": re.compile(r'["\']'),
    "unicode_dashes": re.compile(r"[–—―‒]"),
    "unicode_spaces": re.compile(
        r"[\u00A0\u2000-\u200B\u2028\u2029\u202F\u205F\u3000]"
    ),
    "unicode_mathematical": re.compile(r"[∀∂∃∅∇∈∉∋∏∑−∗∘∙√∝∞∟∠∡∢∣∤∥∦∧∨∩∪∫∬∭∮∯∰∱∲∳]"),
    "unicode_arrows": re.compile(
        r"[←↑→↓↔↕↖↗↘↙↚↛↜↝↞↟↠↡↢↣↤↥↦↧↨↩↪↫↬↭↮↯↰↱↲↳↴↵↶↷↸↹↺↻↼↽↾↿⇀⇁⇂⇃⇄⇅⇆⇇⇈⇉⇊⇋⇌⇍⇎⇏]"
    ),
    # === EMOJIS & UNICODE SYMBOLS ===
    "emoji_faces": re.compile(r"[\U0001F600-\U0001F64F]"),  # Emoticons
    "emoji_people": re.compile(
        r"[\U0001F466-\U0001F469][\U0001F3FB-\U0001F3FF]?"
    ),  # People with skin tones
    "emoji_misc_symbols": re.compile(
        r"[\U0001F300-\U0001F5FF]"
    ),  # Misc symbols and pictographs
    "emoji_transport": re.compile(r"[\U0001F680-\U0001F6FF]"),  # Transport and map
    "emoji_supplemental": re.compile(
        r"[\U0001F900-\U0001F9FF]"
    ),  # Supplemental symbols
    "emoji_flags": re.compile(r"[\U0001F1E6-\U0001F1FF]{2}"),  # Country flags
    "emoji_zodiac": re.compile(r"[\u2648-\u2653]"),  # Zodiac signs
    "emoji_hearts": re.compile(
        r"[\u2665\u2764\U0001F90D\U0001F90E\U0001F494\U0001F495\U0001F496\U0001F497\U0001F498\U0001F499\U0001F49A\U0001F49B\U0001F49C\U0001F49D\U0001F49E\U0001F49F]"
    ),
    "emoji_numbers": re.compile(
        r"[\u0030\u0031\u0032\u0033\u0034\u0035\u0036\u0037\u0038\u0039]\uFE0F?\u20E3"
    ),  # Keycap numbers
    # === UNICODE BLOCKS & RANGES ===
    "unicode_latin_extended": re.compile(
        r"[\u0100-\u017F\u0180-\u024F]"
    ),  # Latin Extended A & B
    "unicode_cyrillic": re.compile(r"[\u0400-\u04FF\u0500-\u052F]"),  # Cyrillic
    "unicode_arabic": re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
    ),  # Arabic
    "unicode_chinese_cjk": re.compile(r"[\u4E00-\u9FFF]"),  # CJK Unified Ideographs
    "unicode_japanese_hiragana": re.compile(r"[\u3040-\u309F]"),  # Hiragana
    "unicode_japanese_katakana": re.compile(r"[\u30A0-\u30FF]"),  # Katakana
    "unicode_korean_hangul": re.compile(r"[\uAC00-\uD7AF]"),  # Hangul syllables
    "unicode_thai": re.compile(r"[\u0E00-\u0E7F]"),  # Thai
    "unicode_devanagari": re.compile(r"[\u0900-\u097F]"),  # Devanagari (Hindi)
    "unicode_hebrew": re.compile(r"[\u0590-\u05FF]"),  # Hebrew
    "unicode_greek": re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]"),  # Greek
    # === UNICODE SPECIAL CHARACTERS ===
    "unicode_variation_selectors": re.compile(
        r"[\uFE00-\uFE0F\U000E0100-\U000E01EF]"
    ),  # Variation selectors
    "unicode_combining_marks": re.compile(
        r"[\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF]"
    ),  # Combining marks
    "unicode_zero_width": re.compile(
        r"[\u200B-\u200D\u2060\uFEFF]"
    ),  # Zero-width characters
    "unicode_bidi_overrides": re.compile(
        r"[\u202A-\u202E\u2066-\u2069]"
    ),  # Bidirectional overrides
    "unicode_private_use": re.compile(
        r"[\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]"
    ),  # Private use areas
    # === BYTE-LEVEL PATTERNS ===
    "byte_sequences_escaped": re.compile(r"\\x[0-9a-fA-F]{2}"),  # \x41, \xFF
    "byte_sequences_percent": re.compile(r"%[0-9a-fA-F]{2}"),  # URL encoding: %20, %41
    "byte_order_marks": re.compile(r"[\uFEFF\uFFFE]"),  # BOM markers
    "replacement_characters": re.compile(r"[\uFFFD]"),  # Unicode replacement character
    "control_characters": re.compile(r"[\x00-\x1F\x7F-\x9F]"),  # Control characters
    "high_bit_ascii": re.compile(
        r"[\x80-\xFF]"
    ),  # High-bit ASCII (likely encoding issues)
    # === UNICODE NORMALIZATION ISSUES ===
    "unicode_homoglyphs_latin_cyrillic": re.compile(
        r"[\u0430\u043E\u0440\u0440\u0441\u0445\u0079]"
    ),  # Cyrillic that looks like Latin
    "unicode_fullwidth_chars": re.compile(
        r"[\uFF01-\uFF5E]"
    ),  # Fullwidth ASCII variants
    "unicode_subscripts": re.compile(
        r"[\u2080-\u209C]"
    ),  # Subscript numbers and letters
    "unicode_superscripts": re.compile(
        r"[\u2070-\u207F\u00B2\u00B3\u00B9]"
    ),  # Superscript numbers and letters
    "unicode_fractions": re.compile(
        r"[\u00BC-\u00BE\u2150-\u217F]"
    ),  # Fraction characters
    # === BRAND NAMES & SPECIAL FORMATTING ===
    # "brand_names_special": re.compile(
    #     r"\b(?:iPhone|iPad|eBay|PayPal|YouTube|LinkedIn|WiFi|GPS)\b"
    # ),
    # "brand_punctuation": re.compile(r"\b\w+[!*&+]\w*\b"),  # Yahoo!, AT&T
    "hashtags": re.compile(r"#\w+"),
    "mentions": re.compile(r"@\w+"),
    "emoticons": re.compile(r"[:;=8][-o\*\']?[)\](\[dDpP/\\OoX*]"),
    # === LIST MARKERS & FORMATTING ===
    "bullet_points": re.compile(r"^\s*[•·‣⁃▪▫▸▹▪▫◦‣⁃]\s", re.MULTILINE),
    "numbered_lists": re.compile(r"^\s*\d+[.)]\s", re.MULTILINE),
    "lettered_lists": re.compile(r"^\s*[a-zA-Z][.)]\s", re.MULTILINE),
    "roman_numerals": re.compile(r"\b(?:[IVX]+|i{1,3}v?|vi{0,3}|i?x)\b"),
    # === ACADEMIC & CITATIONS ===
    "citations_apa": re.compile(r"\([^)]*\d{4}[^)]*\)"),  # (Author, 2023)
    "citations_numeric": re.compile(r"\[\d+\]"),  # [1], [23]
    "citations_multiple": re.compile(r"\[\d+(?:,\s*\d+)*\]"),  # [1, 2, 3]
    "doi_patterns": re.compile(r"\b10\.\d{4,}\/[^\s]+"),
    "isbn_patterns": re.compile(
        r"\bISBN[-:]?\s*\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dX]\b"
    ),
    # === INTERNET SLANG & ABBREVIATIONS ===
    "leetspeak": re.compile(r"\b\w*[0-9]+\w*\b"),  # l33t, h4ck3r
    "text_speak": re.compile(
        r"\b(?:u|ur|r|n|b4|2day|2morrow|w8|gr8|l8r|thx|plz)\b", re.IGNORECASE
    ),
    "internet_acronyms": re.compile(
        r"\b(?:lol|omg|wtf|btw|fyi|imo|imho|afaik|ttyl|brb)\b", re.IGNORECASE
    ),
    "repeated_letters": re.compile(r"\b\w*([a-zA-Z])\1{2,}\w*\b"),  # hellooo, yesss
    # === FORMATTING & MARKUP ===
    "markdown_bold": re.compile(r"\*\*[^*]+\*\*|__[^_]+__"),
    "markdown_italic": re.compile(r"\*[^*]+\*|_[^_]+_"),
    "markdown_code": re.compile(r"`[^`]+`"),
    "html_tags": re.compile(r"<[^>]+>"),
    "xml_tags": re.compile(r"</?[a-zA-Z][^>]*>"),
    # === COMPOUND WORDS & HYPHENATION ===
    "compound_hyphenated": re.compile(r"\b\w+-\w+(?:-\w+)*\b"),
    "compound_no_space": re.compile(
        r"\b(?:healthcare|timestamp|database|username|password|website|email|online|offline)\b",
        re.IGNORECASE,
    ),
    "compound_with_space": re.compile(
        r"\b(?:health care|time stamp|data base|user name|pass word|web site|e mail|on line|off line)\b",
        re.IGNORECASE,
    ),
    # === CULTURAL & REGIONAL VARIATIONS ===
    "british_spelling": re.compile(
        r"\b\w*(?:our|ise|yse|nce)\b"
    ),  # colour, organise, analyse, defence
    "american_spelling": re.compile(
        r"\b\w*(?:or|ize|yze|nse)\b"
    ),  # color, organize, analyze, defense
    # "quotes_british": re.compile(r"'[^']*'"),  # Single quotes primary
    # "quotes_american": re.compile(r'"[^"]*"'),  # Double quotes primary
    # TODO: latex
}


class Detector(ABC):
    @abstractmethod
    def detect(self, text: str) -> bool:
        pass


class ReDetector(Detector):
    def __init__(self):
        self.patterns = PATTERNS

    def detect(self, text: str) -> bool:
        return True

    def detect_all_patterns(self, text: str) -> dict:
        """Run all pattern detectors on text chunk"""
        results = {}

        # Run all regex patterns
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            results[pattern_name] = len(matches)

        # Add non-regex detectors
        results.update(self._detect_unicode_variations(text))
        # results.update(self._detect_rare_words(text))
        results.update(self._detect_mixed_scripts(text))
        results.update(self._detect_whitespace_variations(text))

        return results

    def _detect_unicode_variations(self, text, compute_levenshtein=False):
        """Detect Unicode normalization variations with Levenshtein distance"""
        nfc_form = unicodedata.normalize("NFC", text)
        nfd_form = unicodedata.normalize("NFD", text)
        nfkc_form = unicodedata.normalize("NFKC", text)
        nfkd_form = unicodedata.normalize("NFKD", text)

        res = {
            "unicode_nfc_differs": int(text != nfc_form),
            "unicode_nfd_differs": int(text != nfd_form),
            "unicode_nfkc_differs": int(text != nfkc_form),
            "unicode_nfkd_differs": int(text != nfkd_form),
            "unicode_combining_chars": len(
                [c for c in text if unicodedata.combining(c)]
            ),
            "unicode_control_chars": len(
                [c for c in text if unicodedata.category(c).startswith("C")]
            ),
        }
        if compute_levenshtein:
            res.update(
                {
                    "unicode_nfc_levenshtein": 0
                    if text == nfc_form
                    else self._levenshtein_distance(text, nfc_form),
                    "unicode_nfd_levenshtein": 0
                    if text == nfc_form
                    else self._levenshtein_distance(text, nfd_form),
                    "unicode_nfkc_levenshtein": 0
                    if text == nfc_form
                    else self._levenshtein_distance(text, nfkc_form),
                    "unicode_nfkd_levenshtein": 0
                    if text == nfc_form
                    else self._levenshtein_distance(text, nfkd_form),
                }
            )
        return res

    def _levenshtein_distance(self, s1, s2):
        """Compute Levenshtein distance between two strings efficiently"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        # Use only two rows to save memory
        previous_row = list(range(len(s2) + 1))
        current_row = [0] * (len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row[0] = i + 1
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row[j + 1] = min(insertions, deletions, substitutions)
            previous_row, current_row = current_row, previous_row

        return previous_row[-1]

    def _detect_mixed_scripts(self, text):
        """Detect mixed writing systems and homoglyph confusion"""
        scripts = defaultdict(int)
        homoglyph_issues = 0

        # Common homoglyph pairs (Latin vs Cyrillic)
        homoglyph_pairs = {
            "a": "а",  # Latin a vs Cyrillic а
            "o": "о",  # Latin o vs Cyrillic о
            "p": "р",  # Latin p vs Cyrillic р
            "c": "с",  # Latin c vs Cyrillic с
            "x": "х",  # Latin x vs Cyrillic х
            "y": "у",  # Latin y vs Cyrillic у
            "e": "е",  # Latin e vs Cyrillic е
        }

        for char in text:
            if char.isalpha():
                try:
                    script = unicodedata.name(char, "").split()[0]
                    scripts[script] += 1

                    # Check for potential homoglyph confusion
                    if char.lower() in homoglyph_pairs.values():
                        homoglyph_issues += 1

                except (ValueError, IndexError):
                    scripts["UNKNOWN"] += 1

        unique_scripts = len(scripts)

        return {
            "mixed_scripts": int(unique_scripts > 1),
            "unique_scripts_count": unique_scripts,
            "latin_chars": scripts.get("LATIN", 0),
            "cyrillic_chars": scripts.get("CYRILLIC", 0),
            "arabic_chars": scripts.get("ARABIC", 0),
            "chinese_chars": scripts.get("CJK", 0),
            "greek_chars": scripts.get("GREEK", 0),
            "hebrew_chars": scripts.get("HEBREW", 0),
            "potential_homoglyphs": homoglyph_issues,
        }

    def _detect_whitespace_variations(self, text):
        """Detect unusual whitespace patterns"""
        regular_spaces = text.count(" ")
        tabs = text.count("\t")
        newlines = text.count("\n")
        unicode_spaces = len(
            re.findall(r"[\u00A0\u2000-\u200B\u2028\u2029\u202F\u205F\u3000]", text)
        )

        return {
            "regular_spaces": regular_spaces,
            "tab_characters": tabs,
            "newline_characters": newlines,
            "unicode_spaces": unicode_spaces,
            "mixed_whitespace": int(tabs > 0 and regular_spaces > 0),
        }

    def _detect_emoji_patterns(self, text):
        """Detect various emoji usage patterns"""
        # Count different emoji categories
        total_emojis = 0
        skin_tone_modifiers = 0
        emoji_sequences = 0

        # Simple emoji counting
        for char in text:
            if ord(char) >= 0x1F600:  # Basic emoji range
                total_emojis += 1

        # Skin tone modifiers (🏻🏼🏽🏾🏿)
        skin_tone_modifiers = len(re.findall(r"[\U0001F3FB-\U0001F3FF]", text))

        # Emoji sequences (multiple emojis together)
        emoji_sequences = len(re.findall(r"[\U0001F600-\U0001F64F]{2,}", text))

        return {
            "total_emoji_chars": total_emojis,
            "skin_tone_modifiers": skin_tone_modifiers,
            "emoji_sequences": emoji_sequences,
        }

    def _detect_encoding_issues(self, text):
        """Detect potential encoding problems"""
        # Common encoding issue patterns
        mojibake_patterns = [
            "Ã¡",
            "Ã©",
            "Ã­",
            "Ã³",
            "Ãº",  # UTF-8 interpreted as Latin-1
            "â€™",
            "â€œ",
            "â€",  # Smart quotes mangled
            "Â ",
            "Â·",
            "ÂÂ",  # Non-breaking space issues
        ]

        encoding_issues = 0
        for pattern in mojibake_patterns:
            encoding_issues += text.count(pattern)

        # Check for suspicious byte sequences
        suspicious_bytes = len(re.findall(r"[\uFFFD]", text))  # Replacement characters
        null_bytes = text.count("\x00")

        return {
            "mojibake_patterns": encoding_issues,
            "replacement_chars": suspicious_bytes,
            "null_bytes": null_bytes,
            "high_byte_chars": len(re.findall(r"[\x80-\xFF]", text)),
        }

    def get_pattern_frequencies(self, results, total_chars):
        """Convert counts to frequencies per character"""
        frequencies = {}

        for pattern_name, count in results.items():
            frequencies[f"{pattern_name}_freq"] = (
                count / total_chars if total_chars > 0 else 0
            )
            frequencies[f"{pattern_name}_count"] = count

        return frequencies


# Example usage for streaming processing
class StreamingPatternAnalyzer:
    def __init__(self):
        self.detector = ReDetector()
        self.total_chars = 0
        self.cumulative_counts = defaultdict(int)

    def process_chunk(self, text_chunk):
        """Process a single chunk of text"""
        if not text_chunk:
            return {}

        chunk_results = self.detector.detect_all_patterns(text_chunk)
        self.total_chars += len(text_chunk)

        # Accumulate counts
        for pattern_name, count in chunk_results.items():
            self.cumulative_counts[pattern_name] += count

        return chunk_results

    def get_final_statistics(self):
        """Get final frequency statistics"""
        frequencies = {}

        for pattern_name, total_count in self.cumulative_counts.items():
            frequencies[f"{pattern_name}_total"] = total_count
            frequencies[f"{pattern_name}_frequency"] = (
                total_count / self.total_chars if self.total_chars > 0 else 0
            )

        frequencies["total_characters_processed"] = self.total_chars
        frequencies["patterns_detected"] = len(self.cumulative_counts)

        return frequencies


# Usage example
def analyze_text_file(file_path, limit: int = None):
    """Analyze a text file for all perturbation patterns"""
    analyzer = StreamingPatternAnalyzer()

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if limit and line_num > limit:
                break

            analyzer.process_chunk(line)

            if line_num % 1000 == 0:
                print(f"Processed {line_num:,} lines")

    return analyzer.get_final_statistics()


# Example for JSONL processing
def analyze_jsonl_dataset(file_path, text_field="text"):
    """Analyze JSONL dataset for perturbation patterns"""

    analyzer = StreamingPatternAnalyzer()
    processed_docs = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = ujson.loads(line)
                if text_field in obj and obj[text_field]:
                    analyzer.process_chunk(obj[text_field])
                    processed_docs += 1

                    if processed_docs % 50000 == 0:
                        print(f"Processed {processed_docs:,} documents")

            except ujson.JSONDecodeError:
                continue

    final_stats = analyzer.get_final_statistics()
    final_stats["total_documents_processed"] = processed_docs

    return final_stats
