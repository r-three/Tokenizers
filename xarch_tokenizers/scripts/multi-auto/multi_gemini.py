import argparse
import functools
import random
import re
from curses import meta
from pathlib import Path

from sklearn import base

# python xarch_tokenizers/scripts/multi-auto/multi_gemini.py data/automated_perturbations/tr_canonical.tsv data/automated_perturbations/tr_perturbed.tsv --language tr -n 11
# python xarch_tokenizers/scripts/multi-auto/multi_gemini.py data/automated_perturbations/it_canonical.tsv data/automated_perturbations/it_perturbed.tsv --language it -n 11
# python xarch_tokenizers/scripts/multi-auto/multi_gemini.py data/automated_perturbations/en_canonical.tsv data/automated_perturbations/en_perturbed.tsv --language tr -n 11
# python xarch_tokenizers/scripts/multi-auto/multi_gemini.py data/automated_perturbations/zh_canonical.tsv data/automated_perturbations/zh_perturbed.tsv --language zh -n 11
# python xarch_tokenizers/scripts/multi-auto/multi_gemini.py data/automated_perturbations/fa_canonical.tsv data/automated_perturbations/fa_perturbed.tsv --language fa -n 11

# Set a fixed seed for reproducibility of random perturbations
random.seed(42)

# --- Punctuation Helper & Decorator ---
SUBCATEGORY_MAPPING = {
    "typo_insert": "Character injection",
    "typo_delete": "Character deletion",
    "typo_substitute": "Unintentional character substitution",
    "keyboard": "Keyboard proximity errors",
    "ocr": "OCR Errors",
    "homoglyph": "Homoglyphs",
    "permutation": "Permutation",
    "capitalization": "Capitalization",
    "zerowidth": "Word Spacing/Zero-width characters/Extra Space",
    "random_repeat_9": "Random repetition x 9",
    "random_repeat_4": "Random repetition x 4",
    "random_repeat_5": "Random repetition x 5",
    "random_repeat_2": "Random repetition x 2",
    "random_repeat_11": "Random repetition x 11",
    "random_repeat_8": "Random repetition x 8",
    "random_repeat_10": "Random repetition x 10",
    "random_repeat_6": "Random repetition x 6",
    "random_repeat_3": "Random repetition x 3",
    "internal_space_1": "Internal space x 1",
    "internal_space_2": "Internal space x 2",
    "internal_space_3": "Internal space x 3",
    "internal_space_4": "Internal space x 4",
    "internal_space_5": "Internal space x 5",
    "internal_space_6": "Internal space x 6",
    "internal_space_7": "Internal space x 7",
    "internal_space_8": "Internal space x 8",
    "internal_space_9": "Internal space x 9",
    "internal_space_10": "Internal space x 10",
    "internal_zerowidth_1": "Internal zerowidth x 1",
    "internal_zerowidth_2": "Internal zerowidth x 2",
    "internal_zerowidth_3": "Internal zerowidth x 3",
    "internal_zerowidth_4": "Internal zerowidth x 4",
    "internal_zerowidth_5": "Internal zerowidth x 5",
    "internal_zerowidth_6": "Internal zerowidth x 6",
    "internal_zerowidth_7": "Internal zerowidth x 7",
    "internal_zerowidth_8": "Internal zerowidth x 8",
    "internal_zerowidth_9": "Internal zerowidth x 9",
    "internal_zerowidth_10": "Internal zerowidth x 10",
    "exhaustive_repeat_11": "",
    "exhaustive_repeat_10": "",
    "exhaustive_repeat_9": "",
    "exhaustive_repeat_8": "",
    "exhaustive_repeat_7": "",
    "exhaustive_repeat_6": "",
    "exhaustive_repeat_5": "",
    "exhaustive_repeat_4": "",
    "exhaustive_repeat_3": "",
    "exhaustive_repeat_2": "",
    "random_repeat_7": "Random repetition x 7",
    "swap_quote": "Swap quote",
    "swap_quote_pair": "Swap quote pair",
    "canonical": "Canonical",
    "accent_variation": "Accent Variation",
    "random_noise_diacritics_addition": "Random Diacritics Addition",
    "random_natural_diacritics_addition": "Random Semi-Natural Diacritics Addition",
    "exhaustive_natural_diacritics_delete": "Diacritics Removal",
}


CATEGORY_MAPPING = {
    "typo_insert": "Noise",
    "typo_delete": "Noise",
    "typo_substitute": "Noise",
    "keyboard": "Noise",
    "ocr": "Noise",
    "homoglyph": "Script / Orthography",
    "permutation": "Script / Orthography",
    "capitalization": "Structural Text Elements",
    "zerowidth": "Script / Orthography",
    "random_repeat_9": "Random Repetition",
    "random_repeat_4": "Random Repetition",
    "random_repeat_5": "Random Repetition",
    "random_repeat_2": "Random Repetition",
    "random_repeat_11": "Random Repetition",
    "random_repeat_8": "Random Repetition",
    "random_repeat_10": "Random Repetition",
    "random_repeat_6": "Random Repetition",
    "random_repeat_3": "Random Repetition",
    "internal_space_1": "Internal Space",
    "internal_space_2": "Internal Space",
    "internal_space_3": "Internal Space",
    "internal_space_4": "Internal Space",
    "internal_space_5": "Internal Space",
    "internal_space_6": "Internal Space",
    "internal_space_7": "Internal Space",
    "internal_space_8": "Internal Space",
    "internal_space_9": "Internal Space",
    "internal_space_10": "Internal Space",
    "internal_zerowidth_1": "Internal Zerowidth",
    "internal_zerowidth_2": "Internal Zerowidth",
    "internal_zerowidth_3": "Internal Zerowidth",
    "internal_zerowidth_4": "Internal Zerowidth",
    "internal_zerowidth_5": "Internal Zerowidth",
    "internal_zerowidth_6": "Internal Zerowidth",
    "internal_zerowidth_7": "Internal Zerowidth",
    "internal_zerowidth_8": "Internal Zerowidth",
    "internal_zerowidth_9": "Internal Zerowidth",
    "internal_zerowidth_10": "Internal Zerowidth",
    "exhaustive_repeat_11": "",
    "exhaustive_repeat_10": "",
    "exhaustive_repeat_9": "",
    "exhaustive_repeat_8": "",
    "exhaustive_repeat_7": "",
    "exhaustive_repeat_6": "",
    "exhaustive_repeat_5": "",
    "exhaustive_repeat_4": "",
    "exhaustive_repeat_3": "",
    "exhaustive_repeat_2": "",
    "random_repeat_7": "Random Repetition",
    "swap_quote": "Quotes",
    "swap_quote_pair": "Quotes",
    "canonical": "",
    "accent_variation": "Script / Orthography",
    "random_noise_diacritics_addition": "Script / Orthography",
    "random_natural_diacritics_addition": "Script / Orthography",
    "exhaustive_natural_diacritics_delete": "Script / Orthography",
}


def _strip_punctuation(word):
    """Separates leading/trailing punctuation from the core word."""
    # This regex finds a prefix of non-word chars, a core (including apostrophes), and a suffix.
    match = re.match(r"^([\W_]*)([\w']*)([\W_]*)$", word)
    if match:
        return match.groups()  # (prefix, core, suffix)
    return ("", word, "")  # Fallback


def preserves_punctuation(func):
    """
    A decorator that strips punctuation from a word, applies the perturbation
    function to the core word, and then reattaches the punctuation.
    """

    @functools.wraps(func)
    def wrapper(word, *args, **kwargs):
        prefix, core_word, suffix = _strip_punctuation(word)
        if (
            not core_word
        ):  # Don't perturb if there's no core word (e.g., just punctuation)
            # For random functions, we need to decide on a consistent return format.
            # Let's return a tuple that won't cause errors downstream.
            if "random" in func.__name__:
                return (
                    (word, "", 0) if func.__name__ == "p_random_repeat" else (word, "")
                )
            return []

        # Call the original perturbation function on the core word
        result = func(core_word, *args, **kwargs)

        # Re-attach punctuation
        if "exhaustive" in func.__name__:
            # For exhaustive funcs returning a list of tuples
            return [
                (f"{prefix}{p_word}{suffix}", replacement)
                if len(item) == 2
                else (f"{prefix}{p_word}{suffix}", replacement, item[2])
                for item in result
                for p_word, replacement in [item[:2]]
            ]

        else:
            # For random funcs returning a tuple
            if len(result) == 3:
                perturbed_word, replacement, count = result
                return f"{prefix}{perturbed_word}{suffix}", replacement, count
            else:
                perturbed_word, replacement = result
                return f"{prefix}{perturbed_word}{suffix}", replacement

    return wrapper


# --- Multi-Lingual Perturbation Configuration ---
ISO_LANGS = {
    "it": "ita_Latn",
    "en": "eng_Latn",
    "tr": "tur_Latn",
    "zh": "zho_Hans",
    "fa": "pes_Arab",
}

ALL_POSSIBLE_ACCENTS = {
    "a": ["â", "à", "á", "ā", "ă", "ą"],
    "e": ["ê", "è", "é", "ë", "ē", "ĕ", "ė", "ę", "ě"],
    "u": ["û", "ù", "ú", "ü", "ū", "ŭ", "ů", "ų"],
    "ü": ["û", "ù", "ú", "ü", "ū", "ŭ", "ů", "ų"],
    "i": ["î", "ì", "í", "ï", "ī", "ĭ", "į", "ı", "İ"],
    "o": ["ô", "ò", "ó", "ö", "ō", "ŏ", "ő", "ø"],
    "ö": ["ô", "ò", "ó", "ö", "ō", "ŏ", "ő", "ø"],
    "c": ["ç", "ć", "č", "ĉ", "ċ"],
    "s": ["ş", "ś", "š", "ŝ", "ș", "ß"],
    "g": ["ğ", "ģ", "ġ", "ĝ"],
    #   "n": ["ñ", "ń", "ň", "ņ", "ŋ"],
    #   "l": ["ł", "ĺ", "ľ", "ļ", "ŀ"],
    #   "z": ["ž", "ź", "ż", "ẑ"],
    #   "d": ["đ", "ď", "ḍ"],
    #   "t": ["ţ", "ť", "ț", "ṭ"],
    #   "r": ["ř", "ŕ", "ŗ"],
    "y": ["ý", "ÿ", "ŷ"],
    #   "h": ["ħ", "ĥ"],
    #   "j": ["ĵ"],
    #   "k": ["ķ"],
    #   "w": ["ŵ"],
    #   "b": ["ḅ"],
    #   "f": ["ḟ"],
    #   "m": ["ṁ"],
    #   "p": ["ṗ"],
    #   "v": ["ṽ"],
    #   "x": ["ẋ"]
}

LANG_CONFIGS = {
    "it": {
        "alphabet": "abcdefghijklmnopqrstuvwxyzàèéìòù",
        "keyboard": {
            "a": "qwsz",
            "b": "vgn",
            "c": "xdfv",
            "d": "serfcx",
            "e": "wsdr",
            "f": "drtgvc",
            "g": "ftyhbv",
            "h": "gyujnb",
            "i": "ujko",
            "l": "kopà",
            "m": "njk,",
            "n": "bhjm",
            "o": "iklp",
            "p": "olà",
            "q": "wa",
            "r": "edfgt",
            "s": "wedxza",
            "t": "rfgyh",
            "u": "yihj",
            "v": "cfgb",
            "x": "zsdc",
            "y": "tghu",
            "z": "asx",
            "è": "é",
            "é": "è",
            "à": "ò",
            "ò": "à",
            "ù": "ì",
            "ì": "ù",
        },
        "ocr": {
            # Letter to number confusions
            'o': '0',
            'l': '1',
            'i': '1',
            'a': '4',
            's': '5',
            'b': '8',
            # Number to letter confusions (reverse mappings)
            '0': 'O',
            '1': 'l',
            '4': 'A',
            '5': 'S',
            '9': 'g',
            '3': 'E',
            '8': 'B',
            # Multi-character confusions
            'rn': 'm',
            'ri': 'n',
            'cl': 'd',
            'li': 'b',
            'nn': 'm',
            'ii': 'n',
            'ni': 'u',
            'ui': 'w',
            'vv': 'w',
            'oo': 'œ',
            # Additional single character confusions
            'O': '0',
            'I': '1',
            'S': '5',
            'G': '6',
            'B': '8',
            'q': 'g',
            'g': 'q',
            'c': 'e',
            'e': 'c',
            'm': 'rn',
            'n': 'ri',
            'd': 'cl',
        },
        "homoglyphs": {"o": "о", "l": "I", "a": "а", "e": "е", "c": "с"},
        # Italian-specific data
        "contractions_split": {
            "del": "de il",
            "dello": "de lo",
            "della": "de la",
            "dei": "de i",
            "degli": "de gli",
            "delle": "de le",
            "dell'": "de l'",
            "al": "a il",
            "allo": "a lo",
            "alla": "a la",
            "ai": "a i",
            "agli": "a gli",
            "alle": "a le",
            "all'": "a l'",
            "dal": "da il",
            "dallo": "da lo",
            "dalla": "da la",
            "dai": "da i",
            "dagli": "da gli",
            "dalle": "da le",
            "dall'": "da l'",
            "nel": "in il",
            "nello": "in lo",
            "nella": "in la",
            "nei": "in i",
            "negli": "in gli",
            "nelle": "in le",
            "nell'": "in l'",
            "col": "con il",
            "coi": "con i",
            "sul": "su il",
            "sullo": "su lo",
            "sulla": "su la",
            "sui": "su i",
            "sugli": "su gli",
            "sulle": "su le",
            "sull'": "su l'",
        },
        "clitics_split": {"glielo": "glie lo", "gliela": "glie la", "melo": "me lo"},
        "avere_h_omission": {"ho": "o", "ha": "a", "hanno": "anno"},
        "sms": {"per": "x", "non": "nn", "comunque": "cmq", "grazie": "grz"},
        "phonetic": {
            "che": "ke",
            "chi": "ki",
            "scena": "shena",
            "ghe": "ge",
            "ghi": "gi",
        },
        "accent_variations": {
            "a": ["a", "à", "á", "a'"],
            "e": ["e", "è", "é", "e'"],
            "i": ["i", "ì", "í", "i'"],
            "o": ["o", "ò", "ó", "o'"],
            "u": ["u", "ù", "ú", "u'"],
        },
        "base_vowel_map": {
            "à": "a",
            "á": "a",
            "â": "a",
            "ä": "a",
            "ã": "a",
            "è": "e",
            "é": "e",
            "ê": "e",
            "ë": "e",
            "ì": "i",
            "í": "i",
            "î": "i",
            "ï": "i",
            "ò": "o",
            "ó": "o",
            "ô": "o",
            "ö": "o",
            "õ": "o",
            "ù": "u",
            "ú": "u",
            "û": "u",
            "ü": "u",
        },
        "title_variations": [
            {"signore", "signor", "sig.", "sig", "sig.re", "signora", "sig.ra"},
            {
                "dottore",
                "dottor",
                "dott.",
                "dott",
                "dr.",
                "dr",
                "dottoressa",
                "dott.ssa",
                "dr.ssa",
            },
            {"professore", "professor", "prof.", "prof", "professoressa", "prof.ssa"},
            {"ingegnere", "ingegner", "ing.", "ing"},
            {"avvocato", "avv.", "avv", "avvocatessa"},
        ],
    },
    "en": {
        "alphabet": "abcdefghijklmnopqrstuvwxyz",
        "keyboard": {
            "a": "qwsz",
            "b": "vgn",
            "c": "xdfv",
            "d": "serfcx",
            "e": "wsdr",
            "f": "drtgvc",
            "g": "ftyhbv",
            "h": "gyujnb",
            "i": "ujko",
            "j": "uikmnh",
            "k": "iolmj",
            "l": "kop",
            "m": "njk",
            "n": "bhjm",
            "o": "iklp",
            "p": "ol",
            "q": "wa",
            "r": "edfgt",
            "s": "wedxza",
            "t": "rfgyh",
            "u": "yihj",
            "v": "cfgb",
            "w": "qase",
            "x": "zsdc",
            "y": "tghu",
            "z": "asx",
        },
        "ocr": {"o": "0", "l": "1", "i": "1", "S": "5", "s": "5", "G": "6", "B": "8"},
        "homoglyphs": {"l": "I", "O": "0", "o": "о", "a": "а"},
    },
    "zh": {
        "alphabet": "abcdefghijklmnopqrstuvwxyzü",  # Pinyin alphabet
        "keyboard": {  # Simulating Pinyin input errors
            "ü": "v",
            "sh": "s",
            "zh": "z",
            "ch": "c",
            "n": "ng",
            "l": "r",
        },
        "ocr": {  # Visually similar characters
            "日": "曰",
            "土": "士",
            "人": "入",
            "天": "夫",
            "太": "大",
        },
        "homoglyphs": {"干": "千", "燥": "躁"},
    },
    "fa": {
        "alphabet": "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی",
        "keyboard": {
            "ا": "غدلت",
            "ب": "قریل",
            "پ": "تدو",
            "ت": "عپان",
            "ث": "یصق",
            "ج": "گحچ",
            "چ": "ج",
            "ح": "کخج",
            "خ": "مهح",
            "د": "اذپ",
            "ذ": "لرد",
            "ر": "بزذ",
            "ز": "یطر",
            "ژ": "یطر",
            "س": "صطشی",
            "ش": "ضظس",
            "ص": "سضث",
            "ض": "شص",
            "ط": "سظز",
            "ظ": "شط",
            "ع": "تغه",
            "غ": "افع",
            "ف": "لقغ",
            "ق": "بثف",
            "ک": "حمگ",
            "گ": "جک",
            "ل": "فذبا",
            "م": "خنک",
            "ن": "هوتم",
            "و": "نپ",
            "ه": "نعخ",
            "ی": "ثزسب",
        },
        "ocr": {
            "ا": "ل1",
            "ب": "پتث",
            "پ": "بتث",
            "ت": "بپث",
            "ث": "بپت",
            "ج": "حخچ",
            "چ": "جحخ",
            "ح": "جچخ",
            "خ": "جچح",
            "د": "ذو",
            "ذ": "دو",
            "ر": "ز",
            "ز": "ر",
            "ژ": "ر",
            "س": "ش",
            "ش": "س",
            "ص": "ض",
            "ض": "ص",
            "ط": "ظ",
            "ظ": "ط",
            "ع": "غ",
            "غ": "ع",
            "ف": "ق",
            "ق": "ف",
            "ک": "گ",
            "گ": "ک",
            "ل": "1ا",
            "م": "هـ",
            "ن": "ة",
            "و": "ؤ",
            "ه": "ة",
            "ی": "ى",
        },
        "homoglyphs": {
            # Persian letters and their homoglyphs
            "ی": "يئى",  # Persian yeh + Arabic yeh, yeh with hamza, alef maksura
            "ي": "یئى",  # Arabic yeh + Persian yeh, yeh with hamza, alef maksura
            "ئ": "یيى",  # Yeh with hamza + Persian yeh, Arabic yeh, alef maksura
            "ى": "یيئ",  # Alef maksura + Persian yeh, Arabic yeh, yeh with hamza
            "ک": "ك",  # Persian kaf + Arabic kaf
            "ك": "ک",  # Arabic kaf + Persian kaf
            "ا": "أآإٱ",  # Plain alef + all alef variants with hamza/madda
            "أ": "اآإٱ",  # Alef with hamza above + other alef variants
            "آ": "اأإٱ",  # Alef with madda + other alef variants
            "إ": "اأآٱ",  # Alef with hamza below + other alef variants
            "ٱ": "اأآإ",  # Alef wasla + other alef variants
            "ه": "ةہھ",  # Arabic heh + teh marbuta, Urdu heh variants
            "ة": "هہھ",  # Teh marbuta + heh variants
            "ہ": "هةھ",  # Urdu heh + Arabic heh, teh marbuta
            "ھ": "هةہ",  # Heh with yeh above + other heh variants
            "و": "ؤ",  # Plain waw + waw with hamza
            "ؤ": "و",  # Waw with hamza + plain waw
            # Persian digits and their homoglyphs
            "۰": "٠0",  # Persian zero + Arabic-Indic zero, Latin zero
            "۱": "١1",  # Persian one + Arabic-Indic one, Latin one
            "۲": "٢2",  # Persian two + Arabic-Indic two, Latin two
            "۳": "٣3",  # Persian three + Arabic-Indic three, Latin three
            "۴": "٤4",  # Persian four + Arabic-Indic four, Latin four
            "۵": "٥5",  # Persian five + Arabic-Indic five, Latin five
            "۶": "٦6",  # Persian six + Arabic-Indic six, Latin six
            "۷": "٧7",  # Persian seven + Arabic-Indic seven, Latin seven
            "۸": "٨8",  # Persian eight + Arabic-Indic eight, Latin eight
            "۹": "٩9",  # Persian nine + Arabic-Indic nine, Latin nine
            # Arabic-Indic digits and their homoglyphs
            "٠": "۰0",  # Arabic-Indic zero + Persian zero, Latin zero
            "١": "۱1",  # Arabic-Indic one + Persian one, Latin one
            "٢": "۲2",  # Arabic-Indic two + Persian two, Latin two
            "٣": "۳3",  # Arabic-Indic three + Persian three, Latin three
            "٤": "۴4",  # Arabic-Indic four + Persian four, Latin four
            "٥": "۵5",  # Arabic-Indic five + Persian five, Latin five
            "٦": "۶6",  # Arabic-Indic six + Persian six, Latin six
            "٧": "۷7",  # Arabic-Indic seven + Persian seven, Latin seven
            "٨": "۸8",  # Arabic-Indic eight + Persian eight, Latin eight
            "٩": "۹9",  # Arabic-Indic nine + Persian nine, Latin nine
            # Latin digits and their homoglyphs
            "0": "۰٠",  # Latin zero + Persian zero, Arabic-Indic zero
            "1": "۱١",  # Latin one + Persian one, Arabic-Indic one
            "2": "۲٢",  # Latin two + Persian two, Arabic-Indic two
            "3": "۳٣",  # Latin three + Persian three, Arabic-Indic three
            "4": "۴٤",  # Latin four + Persian four, Arabic-Indic four
            "5": "۵٥",  # Latin five + Persian five, Arabic-Indic five
            "6": "۶٦",  # Latin six + Persian six, Arabic-Indic six
            "7": "۷٧",  # Latin seven + Persian seven, Arabic-Indic seven
            "8": "۸٨",  # Latin eight + Persian eight, Arabic-Indic eight
            "9": "۹٩",  # Latin nine + Persian nine, Arabic-Indic nine
            # Punctuation homoglyphs
            "؟": "?",  # Arabic question mark + Latin question mark
            "?": "؟",  # Latin question mark + Arabic question mark
            "؛": ";",  # Arabic semicolon + Latin semicolon
            ";": "؛",  # Latin semicolon + Arabic semicolon
            "،": ",",  # Arabic comma + Latin comma
            ",": "،",  # Latin comma + Arabic comma
            "٪": "%",  # Arabic percent + Latin percent
            "%": "٪",  # Latin percent + Arabic percent
            "«": '»"',  # Arabic left quote + right quote, Latin quote
            "»": '«"',  # Arabic right quote + left quote, Latin quote
            '"': "«»",  # Latin quote + Arabic quotes
        },
        "persian_homophone_letters": {
            # Letters that make the /z/ sound
            "ز": "ضذظ",  # ze
            "ض": "زذظ",  # zad
            "ذ": "زضظ",  # zal
            "ظ": "زضذ",  # za
            # Letters that make the /s/ sound
            "س": "صث",  # sin
            "ص": "سث",  # sad
            "ث": "سص",  # se
            # Letters that make the /t/ sound
            "ت": "ط",  # te
            "ط": "ت",  # ta
            # Letters that make the /h/ sound
            "ه": "ح",  # he
            "ح": "ه",  # he jimi (dotless he)
            # Letters that can carry the /a/ sound
            "ا": "ع",  # alef
            "ع": "ا",  # ain (when used as vowel carrier)
        },
        "random_noise_diacritics_addition": "Random Diacritics Addition": {
            'ا': ['ا', 'اَ', 'اِ', 'اُ', 'اً', 'اٍ', 'اٌ', 'اْ', 'اّ', 'اٰ'],
            'ب': ['ب', 'بَ', 'بِ', 'بُ', 'بً', 'بٍ', 'بٌ', 'بْ', 'بّ'],
            'پ': ['پ', 'پَ', 'پِ', 'پُ', 'پً', 'پٍ', 'پٌ', 'پْ', 'پّ'],
            'ت': ['ت', 'تَ', 'تِ', 'تُ', 'تً', 'تٍ', 'تٌ', 'تْ', 'تّ'],
            'ث': ['ث', 'ثَ', 'ثِ', 'ثُ', 'ثً', 'ثٍ', 'ثٌ', 'ثْ', 'ثّ'],
            'ج': ['ج', 'جَ', 'جِ', 'جُ', 'جً', 'جٍ', 'جٌ', 'جْ', 'جّ'],
            'چ': ['چ', 'چَ', 'چِ', 'چُ', 'چً', 'چٍ', 'چٌ', 'چْ', 'چّ'],
            'ح': ['ح', 'حَ', 'حِ', 'حُ', 'حً', 'حٍ', 'حٌ', 'حْ', 'حّ'],
            'خ': ['خ', 'خَ', 'خِ', 'خُ', 'خً', 'خٍ', 'خٌ', 'خْ', 'خّ'],
            'د': ['د', 'دَ', 'دِ', 'دُ', 'دً', 'دٍ', 'دٌ', 'دْ', 'دّ'],
            'ذ': ['ذ', 'ذَ', 'ذِ', 'ذُ', 'ذً', 'ذٍ', 'ذٌ', 'ذْ', 'ذّ'],
            'ر': ['ر', 'رَ', 'رِ', 'رُ', 'رً', 'رٍ', 'رٌ', 'رْ', 'رّ'],
            'ز': ['ز', 'زَ', 'زِ', 'زُ', 'زً', 'زٍ', 'زٌ', 'زْ', 'زّ'],
            'ژ': ['ژ', 'ژَ', 'ژِ', 'ژُ', 'ژً', 'ژٍ', 'ژٌ', 'ژْ', 'ژّ'],
            'س': ['س', 'سَ', 'سِ', 'سُ', 'سً', 'سٍ', 'سٌ', 'سْ', 'سّ'],
            'ش': ['ش', 'شَ', 'شِ', 'شُ', 'شً', 'شٍ', 'شٌ', 'شْ', 'شّ'],
            'ص': ['ص', 'صَ', 'صِ', 'صُ', 'صً', 'صٍ', 'صٌ', 'صْ', 'صّ'],
            'ض': ['ض', 'ضَ', 'ضِ', 'ضُ', 'ضً', 'ضٍ', 'ضٌ', 'ضْ', 'ضّ'],
            'ط': ['ط', 'طَ', 'طِ', 'طُ', 'طً', 'طٍ', 'طٌ', 'طْ', 'طّ'],
            'ظ': ['ظ', 'ظَ', 'ظِ', 'ظُ', 'ظً', 'ظٍ', 'ظٌ', 'ظْ', 'ظّ'],
            'ع': ['ع', 'عَ', 'عِ', 'عُ', 'عً', 'عٍ', 'عٌ', 'عْ', 'عّ'],
            'غ': ['غ', 'غَ', 'غِ', 'غُ', 'غً', 'غٍ', 'غٌ', 'غْ', 'غّ'],
            'ف': ['ف', 'فَ', 'فِ', 'فُ', 'فً', 'فٍ', 'فٌ', 'فْ', 'فّ'],
            'ق': ['ق', 'قَ', 'قِ', 'قُ', 'قً', 'قٍ', 'قٌ', 'قْ', 'قّ'],
            'ک': ['ک', 'کَ', 'کِ', 'کُ', 'کً', 'کٍ', 'کٌ', 'کْ', 'کّ'],
            'گ': ['گ', 'گَ', 'گِ', 'گُ', 'گً', 'گٍ', 'گٌ', 'گْ', 'گّ'],
            'ل': ['ل', 'لَ', 'لِ', 'لُ', 'لً', 'لٍ', 'لٌ', 'لْ', 'لّ'],
            'م': ['م', 'مَ', 'مِ', 'مُ', 'مً', 'مٍ', 'مٌ', 'مْ', 'مّ'],
            'ن': ['ن', 'نَ', 'نِ', 'نُ', 'نً', 'نٍ', 'نٌ', 'نْ', 'نّ'],
            'و': ['و', 'وَ', 'وِ', 'وُ', 'وً', 'وٍ', 'وٌ', 'وْ', 'وّ'],
            'ه': ['ه', 'هَ', 'هِ', 'هُ', 'هً', 'هٍ', 'هٌ', 'هْ', 'هّ'],
            'ی': ['ی', 'یَ', 'یِ', 'یُ', 'یً', 'یٍ', 'یٌ', 'یْ', 'یّ']
        }

    },
    "tr": {
        "alphabet": "abcçdefgğhıijklmnoöprsştuüvyz",
        "keyboard": {
            "a": "qwsz",
            "s": "wedxza",
            "d": "serfcx",
            "f": "drtgvc",
            "g": "ftyhbv",
            "ğ": "üişp",
            "h": "gyujnb",
            "j": "uıkmnh",
            "k": "ıolmj",
            "l": "kopşö",
            "ş": "pilç,",
            "i": "şçǧü",
            "z": "asx",
            "x": "zsdc",
            "c": "xdfv",
            "v": "cgb",
            "b": "vhn",
            "n": "bjm",
            "m": "nkö",
            "ö": "mlç",
            "ç": "öşi",
        },
        "ocr": {
            "o": "0",
            "ı": "1",
            "i": "1",
            "s": "5",
            "g": "ğ",
            "c": "ç",
            "S": "Ş",
            "İ": "I",
        },
        "homoglyphs": {"o": "о", "a": "а", "e": "е", "c": "с"},
        "accent_variations": {
            "c": ["ç"],
            "s": ["ş"],
            "ş": ["sh"],
            "ç": ["ch"],
            "u": ["ü"],
            "o": ["ö"],
            "i": ["i"],
            "g": ["ǧ", "ģ"],
            "a": ["â"],
        },
        "base_vowel_map": {
            "ç": "c",
            "ş": "s",
            "ü": "u",
            "ö": "o",
            "ı": "i",
            "ǧ": "g",
            "â": "a",
        },
    },
}

# --- Quote Swapping Configuration ---
SWAPPABLE_SINGLE_QUOTES = ["'", "‘", "’", "‹", "›", "′", "‚", "‵", "❛", "❜", "‛"]
SWAPPABLE_DOUBLE_QUOTES = [
    '"',
    "“",
    "”",
    "«",
    "»",
    "″",
    "‴",
    "⁗",
    "「",
    "」",
    "『",
    "』",
    "„",
    "‶",
    "❝",
    "❞",
    "‟",
]
QUOTE_PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("«", "»"),
    ("「", "」"),
    ("『", "』"),
]
LANG_PREFIX_MAP = {"en": 0, "fa": 1, "tr": 2, "it": 3, "zh": 4}


# --- RANDOM Perturbation Functions ---
@preserves_punctuation
def p_typographical_error_insert_random(word, lang_config):
    alphabet = lang_config.get("alphabet", "abcdefghijklmnopqrstuvwxyz")
    if word.isdigit() or (word.endswith("%") and word[:-1].isdigit()):
        return word, ""
    if not word:
        return word, ""
    pos = random.randint(0, len(word))
    char = random.choice(alphabet)
    return word[:pos] + char + word[pos:], char


@preserves_punctuation
def p_typographical_error_delete_random(word, lang_config):
    if word.isdigit() or (word.endswith("%") and word[:-1].isdigit()):
        return word, ""
    if len(word) < 2:
        return word, ""
    pos = random.randint(0, len(word) - 1)
    return word[:pos] + word[pos + 1 :], ""


@preserves_punctuation
def p_typographical_error_substitute_random(word, lang_config):
    alphabet = lang_config.get("alphabet", "abcdefghijklmnopqrstuvwxyz")
    if word.isdigit() or (word.endswith("%") and word[:-1].isdigit()):
        return word, ""
    if not word:
        return word, ""
    pos = random.randint(0, len(word) - 1)
    char = random.choice(alphabet)
    return word[:pos] + char + word[pos + 1 :], char


@preserves_punctuation
def p_keyboard_proximity_error_random(word, lang_config):
    keyboard_map = lang_config.get("keyboard", {})
    if not keyboard_map:
        return word, ""

    chars = list(word)
    eligible_indices = [
        i for i, char in enumerate(chars) if char.lower() in keyboard_map
    ]
    if not eligible_indices:
        return word, ""
    idx = random.choice(eligible_indices)
    char_to_replace = chars[idx].lower()
    neighbor = random.choice(keyboard_map[char_to_replace])
    chars[idx] = neighbor.upper() if chars[idx].isupper() else neighbor
    return "".join(chars), chars[idx]


@preserves_punctuation
def p_ocr_error_random(word, lang_config):
    ocr_map = lang_config.get("ocr", {})
    if not ocr_map:
        return word, ""

    keys = list(ocr_map.keys())
    random.shuffle(keys)
    for target in keys:
        if target in word:
            replacement = ocr_map[target]
            return word.replace(target, replacement, 1), replacement
    return word, ""


@preserves_punctuation
def p_add_homoglyph_random(word, lang_config):
    homoglyph_map = lang_config.get("homoglyphs", {})
    if not homoglyph_map:
        return word, ""

    chars = list(word)
    eligible_indices = [i for i, char in enumerate(chars) if char in homoglyph_map]
    if not eligible_indices:
        return word, ""
    idx = random.choice(eligible_indices)
    replacement = homoglyph_map[chars[idx]]
    chars[idx] = replacement
    return "".join(chars), replacement


@preserves_punctuation
def p_permutation_error_random(word, lang_config):
    if word.isdigit() or (word.endswith("%") and word[:-1].isdigit()):
        return word, ""
    if len(word) < 2:
        return word, ""
    pos = random.randint(0, len(word) - 2)
    p_word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
    return p_word, p_word


@preserves_punctuation
def p_change_capitalization_random(word, lang_config):
    op = random.choice(["lower", "upper", "title"])
    if op == "lower":
        p_word = word.lower()
    elif op == "upper":
        p_word = word.upper()
    else:
        p_word = word.title()
    return p_word, p_word


@preserves_punctuation
def p_add_zero_width_char_random(word, lang_config):
    if len(word) < 2:
        return word, ""
    pos = random.randint(1, len(word) - 1)
    replacement = "\u200b"
    return word[:pos] + replacement + word[pos:], replacement


@preserves_punctuation
def p_random_repeat(word, lang_config):
    if not word:
        return word, "", 0

    pos = random.randint(0, len(word) - 1)
    char_to_repeat = word[pos]
    num_repeats = random.randint(2, 11)

    p_word = word[:pos] + (char_to_repeat * num_repeats) + word[pos + 1 :]
    replacement = char_to_repeat * num_repeats

    return p_word, replacement, num_repeats


# --- EXHAUSTIVE Perturbation Functions (Language-Agnostic) ---
def p_swap_quote_exhaustive(word, lang_config):
    results = set()
    for i, char in enumerate(word):
        if char in SWAPPABLE_SINGLE_QUOTES:
            for quote_char in SWAPPABLE_SINGLE_QUOTES:
                if quote_char != char:
                    p_word = word[:i] + quote_char + word[i + 1 :]
                    results.add((p_word, quote_char))
        elif char in SWAPPABLE_DOUBLE_QUOTES:
            for quote_char in SWAPPABLE_DOUBLE_QUOTES:
                if quote_char != char:
                    p_word = word[:i] + quote_char + word[i + 1 :]
                    results.add((p_word, quote_char))
    return sorted(list(results), key=lambda x: x[0])


def p_swap_quote_pair_exhaustive(word, lang_config):
    if len(word) < 2:
        return []
    original_left, original_right = None, None
    for left, right in QUOTE_PAIRS:
        if word.startswith(left) and word.endswith(right):
            original_left, original_right = left, right
            break
    if original_left is None:
        return []
    core_content = word[len(original_left) : -len(original_right)]
    original_pair_found = (original_left, original_right)
    results = set()
    for new_left, new_right in QUOTE_PAIRS:
        if (new_left, new_right) != original_pair_found:
            p_word = f"{new_left}{core_content}{new_right}"
            replacement = f"{new_left}...{new_right}"
            results.add((p_word, replacement))
    return sorted(list(results), key=lambda x: x[0])


def p_internal_space_exhaustive(word, lang_config):
    if not word:
        return []

    prefix, core, suffix = _strip_punctuation(word)
    results = set()

    # Case 1: Word has a core and attached punctuation.
    if core and (prefix or suffix):
        if prefix:
            for i in range(1, 11):
                results.add(
                    (
                        f"{prefix}{' ' * i}{core}{suffix}",
                        f"{prefix}{' ' * i}{core}{suffix}",
                        i,
                    )
                )
        if suffix:
            for i in range(1, 11):
                results.add(
                    (
                        f"{prefix}{core}{' ' * i}{suffix}",
                        f"{prefix}{core}{' ' * i}{suffix}",
                        i,
                    )
                )
        if prefix and suffix:
            for i in range(1, 11):
                results.add(
                    (
                        f"{prefix}{' ' * i}{core}{' ' * i}{suffix}",
                        f"{prefix}{' ' * i}{core}{' ' * i}{suffix}",
                        i,
                    )
                )

    # Case 2: ANY word (including standalone punctuation) can have spaces added AFTER it.
    for i in range(1, 11):
        space = " " * i
        new_word = f"{word}{space}"
        results.add((new_word, new_word, i))

    return sorted(list(results), key=lambda x: (x[0], x[2]))


def p_internal_zero_width_exhaustive(word, lang_config):
    if not word:
        return []

    prefix, core, suffix = _strip_punctuation(word)
    results = set()
    zero_width_char = "\u200b"

    # Case 1: Word has a core and attached punctuation.
    if core and (prefix or suffix):
        if prefix:
            for i in range(1, 11):
                results.add(
                    (
                        f"{prefix}{zero_width_char * i}{core}{suffix}",
                        f"{prefix}{zero_width_char * i}{core}{suffix}",
                        i,
                    )
                )
        if suffix:
            for i in range(1, 11):
                results.add(
                    (
                        f"{prefix}{core}{zero_width_char * i}{suffix}",
                        f"{prefix}{core}{zero_width_char * i}{suffix}",
                        i,
                    )
                )
        if prefix and suffix:
            for i in range(1, 11):
                results.add(
                    (
                        f"{prefix}{zero_width_char * i}{core}{zero_width_char * i}{suffix}",
                        f"{prefix}{zero_width_char * i}{core}{zero_width_char * i}{suffix}",
                        i,
                    )
                )

    # Case 2: ANY word can have zero-width chars added AFTER it.
    for i in range(1, 11):
        results.add((f"{word}{zero_width_char * i}", f"{word}{zero_width_char * i}", i))

    return sorted(list(results), key=lambda x: (x[0], x[2]))


@preserves_punctuation
def p_exhaustive_repeat(word, lang_config):
    if not word:
        return []
    results = set()
    for i in range(len(word)):
        char_to_repeat = word[i]
        # Repeat from 2 times up to 11 times in total
        for num_repeats in range(2, 12):
            p_word = word[:i] + (char_to_repeat * num_repeats) + word[i + 1 :]
            replacement = char_to_repeat * num_repeats
            results.add((p_word, replacement, num_repeats))
    return sorted(list(results), key=lambda x: (x[0], x[2]))


# --- EXHAUSTIVE Perturbation Functions (Italian-Specific) ---
def _num_to_words_it_helper(num):
    if num == 0:
        return ""
    units = [
        "",
        "uno",
        "due",
        "tre",
        "quattro",
        "cinque",
        "sei",
        "sette",
        "otto",
        "nove",
    ]
    teens = [
        "dieci",
        "undici",
        "dodici",
        "tredici",
        "quattordici",
        "quinzici",
        "sedici",
        "diciassette",
        "diciotto",
        "diciannove",
    ]
    tens = [
        "",
        "dieci",
        "venti",
        "trenta",
        "quaranta",
        "cinquanta",
        "sessanta",
        "settanta",
        "ottanta",
        "novanta",
    ]
    if num < 10:
        return units[num]
    if num < 20:
        return teens[num - 10]
    if num < 100:
        ten, unit = divmod(num, 10)
        if unit in [1, 8]:
            return tens[ten][:-1] + units[unit]
        return tens[ten] + units[unit]
    if num < 1000:
        hundred, rest = divmod(num, 100)
        hundred_str = "cento" if hundred == 1 else units[hundred] + "cento"
        return hundred_str + _num_to_words_it_helper(rest)
    return ""


@preserves_punctuation
def p_number_format_exhaustive_it(word, lang_config):
    clean_word = re.sub(r"[.,\s]", "", word)
    if not clean_word.isdigit():
        return []
    num = int(clean_word)
    options = set()
    if num > 999:
        options.add(f"{num:,}".replace(",", "."))
        options.add(f"{num:,}")
        options.add(f"{num:,}".replace(",", " "))
    options.add(str(num))
    if num >= 1000 and num % 1000 == 0:
        prefix_num = num // 1000
        options.add(f"{prefix_num}mila")
        options.add(f"{prefix_num}k")
        options.add(f"{prefix_num}K")
        prefix_words = _num_to_words_it_helper(prefix_num)
        if prefix_words:
            options.add(prefix_words + "mila")
            options.add(prefix_words + " mila")
    if 0 <= num <= 100:
        if num == 0:
            options.add("zero")
        else:
            word_form = _num_to_words_it_helper(num)
            if word_form:
                options.add(word_form)
    return sorted([(opt, opt) for opt in options if opt != word])


def p_percentage_exhaustive_it(word, lang_config):
    match = re.fullmatch(r"(\d+)(%)", word)
    if not match:
        return []
    num_str, symbol = match.groups()
    results = set()
    num_variations_tuples = p_number_format_exhaustive_it(num_str, lang_config) + [
        (num_str, num_str)
    ]
    symbol_variations_tuples = p_percentage_symbol_exhaustive_it(
        symbol, lang_config
    ) + [(symbol, symbol)]
    for num_var, _ in num_variations_tuples:
        for sym_var, _ in symbol_variations_tuples:
            if sym_var == "%":
                results.add(f"{num_var}%")
                results.add(f"{num_var} %")
                results.add(f"%{num_var}")
                results.add(f"% {num_var}")
            else:
                results.add(f"{num_var}{sym_var}")
                results.add(f"{num_var} {sym_var}")
                if sym_var.isalpha():
                    results.add(f"{num_var}{sym_var.capitalize()}")
    return sorted([(opt, opt) for opt in results if opt != word])


@preserves_punctuation
def p_percentage_symbol_exhaustive_it(word, lang_config):
    if word == "%":
        return [(r, r) for r in ["per cento", "percento", "xcento"]]
    return []


@preserves_punctuation
def p_sms_spelling_exhaustive_it(word, lang_config):
    sms_map = lang_config.get("sms", {})
    word_lower = word.lower()
    if word_lower in sms_map:
        replacement = str(sms_map[word_lower])
        p_word = (
            replacement.capitalize()
            if word and word[0].isupper() and replacement.isalpha()
            else replacement
        )
        return [(p_word, p_word)]
    return []


@preserves_punctuation
def p_h_omission_exhaustive_it(word, lang_config):
    h_omission_map = lang_config.get("avere_h_omission", {})
    results = set()
    word_lower = word.lower()
    if word_lower in h_omission_map:
        replacement = h_omission_map[word_lower]
        p_word = replacement.capitalize() if word and word[0].isupper() else replacement
        results.add((p_word, replacement))
    if "ch" in word_lower:
        results.add((word.replace("ch", "c").replace("Ch", "C"), "c"))
    if "gh" in word_lower:
        results.add((word.replace("gh", "g").replace("Gh", "G"), "g"))
    return sorted(list(results), key=lambda x: x[0])


@preserves_punctuation
def p_accent_variation_exhaustive_it(word, lang_config):
    accent_map = lang_config.get("accent_variations", {})
    base_vowel_map = lang_config.get("base_vowel_map", {})
    results = set()
    for i, char in enumerate(word):
        char_lower = char.lower()
        base_vowel = base_vowel_map.get(
            char_lower, char_lower if char_lower in accent_map else None
        )
        if base_vowel:
            for var in accent_map[base_vowel]:
                if var != char_lower:
                    new_char = var.upper() if char.isupper() else var
                    p_word = word[:i] + new_char + word[i + 1 :]
                    results.add((p_word, new_char))
    return sorted(list(results), key=lambda x: x[0])


# --- EXHAUSTIVE Perturbation Functions (Applicable to many langs) ---


### splits the above function into two
@preserves_punctuation
def p_exhaustive_natural_diacritics_delete(word, lang_config):
    """strips the diactrics in an exhaustive fashion"""
    if not word:
        return []
    base_vowel_map = lang_config.get("base_vowel_map", None)
    if not base_vowel_map:
        return []
    results = set()
    for i, char in enumerate(word):
        char_lower = char.lower()
        base_vowel = base_vowel_map.get(char_lower, None)
        if base_vowel:
            p_word = word[:i] + base_vowel + word[i + 1 :]
            results.add((p_word, base_vowel))
    return sorted(list(results), key=lambda x: (x[0]))


@preserves_punctuation
def p_random_natural_diacritics_addition(word, lang_config):
    """adds accented variations"""
    accent_map = lang_config.get("accent_variations", {})
    chars = list(word)
    eligible_indices = [i for i, char in enumerate(chars) if char.lower() in accent_map]
    if not eligible_indices:
        return word, ""
    idx = random.choice(eligible_indices)
    char_to_replace = chars[idx].lower()
    diacritized = random.choice(accent_map[char_to_replace])
    chars[idx] = diacritized.upper() if chars[idx].isupper() else diacritized
    return "".join(chars), chars[idx]


@preserves_punctuation
def p_random_noise_diacritics_addition(word, lang_config):
    """randomly replaces any diacritizable character with its diacritized version"""
    results = set()
    chars = list(word)
    eligible_indices = [
        i for i, char in enumerate(chars) if char.lower() in ALL_POSSIBLE_ACCENTS
    ]
    if not eligible_indices:
        return word, ""
    idx = random.choice(eligible_indices)
    char_to_replace = chars[idx].lower()
    diacritized = random.choice(ALL_POSSIBLE_ACCENTS[char_to_replace])
    chars[idx] = diacritized.upper() if chars[idx].isupper() else diacritized
    return "".join(chars), chars[idx]


# --- Main Logic ---


def get_perturbations_for_language(lang_code):
    """Returns dictionaries of perturbation functions for a given language."""
    random_pert = {
        "typo_insert": p_typographical_error_insert_random,
        "typo_delete": p_typographical_error_delete_random,
        "typo_substitute": p_typographical_error_substitute_random,
        "keyboard": p_keyboard_proximity_error_random,
        "ocr": p_ocr_error_random,
        "homoglyph": p_add_homoglyph_random,
        "permutation": p_permutation_error_random,
        "capitalization": p_change_capitalization_random,
        "zerowidth": p_add_zero_width_char_random,
        "random_repeat": p_random_repeat,
        "random_natural_diacritics_addition": p_random_natural_diacritics_addition,
        "random_noise_diacritics_addition": p_random_noise_diacritics_addition,
    }

    exhaustive_pert = {
        "swap_quote": p_swap_quote_exhaustive,
        "swap_quote_pair": p_swap_quote_pair_exhaustive,
        "internal_space": p_internal_space_exhaustive,
        "internal_zerowidth": p_internal_zero_width_exhaustive,
        "exhaustive_natural_diacritics_delete": p_exhaustive_natural_diacritics_delete,
        # "exhaustive_repeat": p_exhaustive_repeat,
    }

    context_pert = {}

    # if lang_code == "it":
    #     exhaustive_pert.update(
    #         {
    #             # 'sms_spelling': p_sms_spelling_exhaustive_it,
    #             # 'h_omission': p_h_omission_exhaustive_it,
    #             # 'number': p_number_format_exhaustive_it,
    #             # 'percentage': p_percentage_exhaustive_it,
    #             # 'percentage_symbol': p_percentage_symbol_exhaustive_it,
    #             "accent_variation": p_accent_variation_exhaustive_it,
    #         }
    #     )

    return random_pert, exhaustive_pert, context_pert


def generate_perturbations(
    input_file,
    questions_output,
    lang_code,
    num_versions,
    specific_types,
    target_words_file,
    set_id,
):
    input_file = Path(input_file).resolve().absolute()
    if lang_code not in LANG_CONFIGS:
        print(
            f"Error: Language code '{lang_code}' is not supported. Supported codes: {list(LANG_CONFIGS.keys())}"
        )
        return

    lang_config = LANG_CONFIGS[lang_code]
    (
        RANDOM_WORD_PERTURBATIONS,
        EXHAUSTIVE_WORD_PERTURBATIONS,
        CONTEXT_AWARE_PERTURBATIONS,
    ) = get_perturbations_for_language(lang_code)
    ALL_PERTURBATIONS = {
        **RANDOM_WORD_PERTURBATIONS,
        **EXHAUSTIVE_WORD_PERTURBATIONS,
        **CONTEXT_AWARE_PERTURBATIONS,
    }

    if "." in questions_output:
        name, ext = questions_output.rsplit(".", 1)
        metadata_output = f"{name}_metadata.{ext}"
    else:
        metadata_output = f"{questions_output}_metadata"
    questions_output = Path(questions_output).resolve().absolute()
    metadata_output = Path(metadata_output).resolve().absolute()

    target_word_lines = []
    if target_words_file:
        try:
            with open(target_words_file, "r", encoding="utf-8") as f:
                target_word_lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Error: Target words file not found at '{target_words_file}'")
            return

    try:
        header = [
            "Question",
            "Correct",
            "Opt 1",
            "Opt 2",
            "Opt 3",
            "Perturbed Word",
            "Perturbed Word Index",
            "Perturbation",
            "Perturbed Char",
            "Set Id",
            "Variation Id",
            "Lang",
            "Id",
            "Subcategory",
            "Category",
            "Version",
        ]
        with (
            open(input_file, "r", encoding="utf-8") as infile,
            open(questions_output, "w", encoding="utf-8", newline="") as q_outfile,
            open(metadata_output, "w", encoding="utf-8", newline="") as m_outfile,
        ):
            m_outfile.write("\t".join(header) + "\n")
            total_generated = 0
            for line_num, line in enumerate(infile, 1):
                print(line_num, line)
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                question_id = parts[0]
                variation_counter = 1
                lang_prefix = LANG_PREFIX_MAP[lang_code]

                original_question = parts[1]
                words = original_question.split()
                if not words:
                    continue

                indices_to_perturb = []
                if (
                    target_word_lines
                    and line_num <= len(target_word_lines)
                    and target_word_lines[line_num - 1]
                ):
                    target_words_set = {
                        w.lower() for w in target_word_lines[line_num - 1].split()
                    }
                    indices_to_perturb = [
                        i
                        for i, w in enumerate(words)
                        if _strip_punctuation(w)[1].lower() in target_words_set
                    ]
                else:
                    target_word_str = (
                        parts[-1] if len(parts) > 6 and not target_word_lines else None
                    )
                    if target_word_str:
                        indices_to_perturb = [
                            i
                            for i, w in enumerate(words)
                            if _strip_punctuation(w)[1].lower()
                            == target_word_str.lower()
                        ]
                    else:
                        indices_to_perturb = list(range(len(words)))

                answers = parts[2:6]
                types_to_apply = (
                    specific_types if specific_types else ALL_PERTURBATIONS.keys()
                )
                ## also write the canonical q
                variation_id = f"{lang_prefix}.0"
                question_parts = [original_question, *answers]
                q_outfile.write("\t".join(map(str, question_parts)) + "\n")

                metadata_parts = [
                    "",
                    "",
                    "Canonical",
                    "",
                    str(question_id),
                    str(variation_id),
                    ISO_LANGS.get(lang_code),
                    f"{question_id}-{variation_id}",
                    "Canonical",
                    "",
                ]
                sanitized_metadata_parts = [
                    str(p).replace('"', "'") for p in metadata_parts
                ]
                m_outfile.write(
                    "\t".join(
                        map(
                            str,
                            question_parts + sanitized_metadata_parts,
                        )
                    )
                    + "\n"
                )
                for pert_type in types_to_apply:
                    if pert_type not in ALL_PERTURBATIONS:
                        continue

                    for index in indices_to_perturb:
                        original_word_affected = words[index]

                        if pert_type in RANDOM_WORD_PERTURBATIONS:
                            generated_results = set()
                            max_attempts = num_versions * 5
                            for _ in range(max_attempts):
                                if len(generated_results) >= num_versions:
                                    break
                                func = RANDOM_WORD_PERTURBATIONS[pert_type]
                                result_tuple = func(original_word_affected, lang_config)

                                if (
                                    result_tuple
                                    and result_tuple[0] != original_word_affected
                                ):
                                    generated_results.add(result_tuple)

                            for result_tuple in generated_results:
                                perturbed_word, replacement = (
                                    result_tuple[0],
                                    result_tuple[1],
                                )
                                final_pert_type = pert_type

                                if pert_type == "random_repeat":
                                    num_repeats = result_tuple[2]
                                    final_pert_type = f"random_repeat_{num_repeats}"

                                temp_words = words[:]
                                temp_words[index] = perturbed_word
                                perturbed_question = " ".join(temp_words)
                                question_parts = [perturbed_question] + answers
                                variation_id = f"{lang_prefix}.{variation_counter}"
                                q_outfile.write(
                                    "\t".join(map(str, question_parts)) + "\n"
                                )

                                metadata_parts = [
                                    original_word_affected,
                                    str(index),
                                    pert_type,
                                    replacement,
                                    str(question_id),
                                    str(variation_id),
                                    ISO_LANGS.get(lang_code),
                                    f"{question_id}-{variation_id}",
                                    SUBCATEGORY_MAPPING.get(pert_type),
                                    CATEGORY_MAPPING.get(pert_type),
                                ]
                                sanitized_metadata_parts = [
                                    str(p).replace('"', "'") for p in metadata_parts
                                ]
                                m_outfile.write(
                                    "\t".join(
                                        map(
                                            str,
                                            question_parts + sanitized_metadata_parts,
                                        )
                                    )
                                    + "\n"
                                )
                                total_generated += 1
                                variation_counter += 1

                        elif pert_type in EXHAUSTIVE_WORD_PERTURBATIONS:
                            func = EXHAUSTIVE_WORD_PERTURBATIONS[pert_type]
                            perturbed_results = func(
                                original_word_affected, lang_config
                            )
                            for result_tuple in perturbed_results:
                                if (
                                    len(result_tuple) == 3
                                ):  # Check for the special format from internal_space/zerowidth/repeat
                                    p_word, replacement, num_chars = result_tuple
                                    final_pert_type = f"{pert_type}_{num_chars}"
                                else:
                                    p_word, replacement = result_tuple
                                    final_pert_type = pert_type

                                if p_word != original_word_affected:
                                    temp_words = words[:]
                                    temp_words[index] = p_word
                                    perturbed_question = " ".join(temp_words)
                                    question_parts = [perturbed_question] + answers
                                    variation_id = f"{lang_prefix}.{variation_counter}"
                                    q_outfile.write(
                                        "\t".join(map(str, question_parts)) + "\n"
                                    )
                                    metadata_parts = [
                                        original_word_affected,
                                        str(index),
                                        pert_type,
                                        replacement,
                                        str(question_id),
                                        str(variation_id),
                                        ISO_LANGS.get(lang_code),
                                        f"{question_id}-{variation_id}",
                                        SUBCATEGORY_MAPPING.get(pert_type),
                                        CATEGORY_MAPPING.get(pert_type),
                                    ]
                                    sanitized_metadata_parts = [
                                        str(p).replace('"', "'") for p in metadata_parts
                                    ]
                                    m_outfile.write(
                                        "\t".join(
                                            map(
                                                str,
                                                question_parts
                                                + sanitized_metadata_parts,
                                            )
                                        )
                                        + "\n"
                                    )
                                    total_generated += 1
                                    variation_counter += 1

                        elif pert_type in CONTEXT_AWARE_PERTURBATIONS:
                            func = CONTEXT_AWARE_PERTURBATIONS[pert_type]
                            perturbed_results = func(words, index, lang_config)
                            for p_sentence, replacement in perturbed_results:
                                if p_sentence != " ".join(words):
                                    question_parts = [p_sentence] + answers
                                    variation_id = f"{lang_prefix}.{variation_counter}"
                                    q_outfile.write(
                                        "\t".join(map(str, question_parts)) + "\n"
                                    )
                                    metadata_parts = [
                                        original_word_affected,
                                        str(index),
                                        pert_type,
                                        replacement,
                                        str(question_id),
                                        str(variation_id),
                                        ISO_LANGS.get(lang_code),
                                        f"{question_id}-{variation_id}",
                                        SUBCATEGORY_MAPPING.get(pert_type),
                                        CATEGORY_MAPPING.get(pert_type),
                                    ]
                                    sanitized_metadata_parts = [
                                        str(p).replace('"', "'") for p in metadata_parts
                                    ]
                                    m_outfile.write(
                                        "\t".join(
                                            map(
                                                str,
                                                question_parts
                                                + sanitized_metadata_parts,
                                            )
                                        )
                                        + "\n"
                                    )
                                    total_generated += 1
                                    variation_counter += 1

        print(f"Successfully generated {total_generated} perturbed examples.")
        print(f"Output saved to: {questions_output} and {metadata_output}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tokenization perturbations for multiple languages."
    )
    parser.add_argument("input_file", help="Path to the input TSV file.")
    parser.add_argument(
        "questions_output", help="Path for the output TSV file with questions."
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=LANG_CONFIGS.keys(),
        help="Language of the input text.",
    )
    parser.add_argument(
        "-n",
        "--num_versions",
        type=int,
        default=1,
        help="Number of versions for RANDOM perturbation types.",
    )
    parser.add_argument(
        "--set-id", type=int, default=300, help="Starting number for the question ID."
    )
    parser.add_argument(
        "--types", nargs="+", help="Generate only specific perturbation types."
    )
    parser.add_argument(
        "--target-words-file", help="Path to a file with space-separated target words."
    )
    args = parser.parse_args()
    generate_perturbations(
        args.input_file,
        args.questions_output,
        args.language,
        args.num_versions,
        args.types,
        args.target_words_file,
        args.set_id,
    )
