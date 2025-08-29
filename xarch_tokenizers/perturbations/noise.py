# Noise	Keyboard proximity errors
# Noise	Noise injection
# Noise	OCR Errors
# Noise	Permutations
# Noise	Typographical errors

from typing import List, Optional

from attr import dataclass

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

# English QWERTY Keyboard Layout
ENGLISH_KEYBOARD_NEIGHBORS = {
    "a": "qwszx",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdr34",
    "f": "drtgvcx",
    "g": "ftyhbvc",
    "h": "gyujnbv",
    "i": "ujko89",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp90",
    "p": "ol0",
    "q": "wa12",
    "r": "edfgt45",
    "s": "wedxzaw",
    "t": "rfgyh56",
    "u": "yihj78",
    "v": "cfgb",
    "w": "qase23",
    "x": "zsdc",
    "y": "tghu67",
    "z": "asx",
    "1": "q2",
    "2": "qw31",
    "3": "we42",
    "4": "re53",
    "5": "rt64",
    "6": "ty75",
    "7": "yu86",
    "8": "ui97",
    "9": "io08",
    "0": "op9",
}

# Farsi/Persian Keyboard Layout (based on standard Persian QWERTY)
FARSI_KEYBOARD_NEIGHBORS = {
    # Persian letters with their neighboring keys
    "ا": "قوش",
    "ب": "لنوت",
    "پ": "چجحخ",
    "ت": "نبپث",
    "ث": "بپت",
    "ج": "چحخپ",
    "چ": "جحپ",
    "ح": "خعهجچ",
    "خ": "حعهج",
    "د": "کگی",
    "ذ": "ظضص",
    "ر": "ذضظ",
    "ز": "ژضظذ",
    "ژ": "زضظ",
    "س": "یشک",
    "ش": "سیکگ",
    "ص": "ضذظ",
    "ض": "صذظزژ",
    "ط": "ظذ",
    "ظ": "طذصضزژ",
    "ع": "حخهف",
    "غ": "فقدک",
    "ف": "غقعه",
    "ق": "فغدک",
    "ک": "گدقغس",
    "گ": "کدشس",
    "ل": "مت",
    "م": "لنب",
    "ن": "متب",
    "و": "اقش",
    "ه": "عحخف",
    "ی": "سشدک",
}

# Turkish Q-type Keyboard Layout
TURKISH_KEYBOARD_NEIGHBORS = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdr",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "ı": "ujkoş",
    "i": "ıujk",
    "j": "huıkmn",
    "k": "jiıolm",
    "l": "kopü",
    "m": "njk",
    "n": "bhjm",
    "o": "ıklpş",
    "ö": "lp",
    "p": "olö",
    "q": "wa",
    "r": "edfgt",
    "s": "wedxza",
    "t": "rfgyh",
    "u": "yıhj",
    "ü": "opşğ",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
    "ç": "ö",
    "ğ": "üş",
    "ş": "ıoğü",
}

# Chinese Pinyin Input (QWERTY-based but with tone considerations)
CHINESE_KEYBOARD_NEIGHBORS = {
    # Standard QWERTY for pinyin input
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdr",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "j": "huikmn",
    "k": "jiolm",
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
    # Tone marks (for pinyin with tones)
    "ā": "áǎà",
    "á": "āǎà",
    "ǎ": "āáà",
    "à": "āáǎ",
    "ē": "éěè",
    "é": "ēěè",
    "ě": "ēéè",
    "è": "ēéě",
    "ī": "íǐì",
    "í": "īǐì",
    "ǐ": "īíì",
    "ì": "īíǐ",
    "ō": "óǒò",
    "ó": "ōǒò",
    "ǒ": "ōóò",
    "ò": "ōóǒ",
    "ū": "úǔù",
    "ú": "ūǔù",
    "ǔ": "ūúù",
    "ù": "ūúǔ",
    "ü": "ǖǘǚǜ",
    "ǖ": "üǘǚǜ",
    "ǘ": "üǖǚǜ",
    "ǚ": "üǖǘǜ",
    "ǜ": "üǖǘǚ",
}

# OCR Errors for English
ENGLISH_OCR_ERRORS = {
    "o": "0",
    "l": "1I|",
    "i": "1|!",
    "a": "4@",
    "s": "5$",
    "g": "9q",
    "e": "3",
    "b": "8B6",
    "c": "o0",
    "d": "cl0",
    "h": "n",
    "m": "rn",
    "u": "v",
    "v": "u",
    "w": "vv",
    "q": "9g",
    "t": "+f",
    "f": "t",
    "p": "P",
    "k": "K",
    "x": "X",
    "z": "Z2",
    "r": "n",
    "n": "rh",
    "rn": "m",
    "ri": "n",
    "cl": "d",
    "ll": "H",
    "nn": "m",
}

# OCR Errors for Farsi/Persian
FARSI_OCR_ERRORS = {
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
}

# OCR Errors for Turkish
TURKISH_OCR_ERRORS = {
    "o": "0ö",
    "l": "1I|",
    "i": "1|!ı",
    "ı": "1|!i",
    "a": "4@",
    "s": "5$ş",
    "ş": "s$",
    "g": "9qğ",
    "ğ": "g9",
    "e": "3",
    "b": "8B6",
    "c": "o0ç",
    "ç": "c",
    "d": "cl0",
    "h": "n",
    "m": "rn",
    "u": "vü",
    "ü": "u",
    "v": "u",
    "w": "vv",
    "q": "9g",
    "t": "+f",
    "f": "t",
    "p": "P",
    "k": "K",
    "x": "X",
    "z": "Z2",
    "r": "n",
    "n": "rh",
    "ö": "o0",
}

# OCR Errors for Chinese Characters (common confusions)
CHINESE_OCR_ERRORS = {
    "人": "入八",
    "入": "人八",
    "八": "人入",
    "口": "日曰",
    "日": "口曰目",
    "目": "日自",
    "自": "目白",
    "白": "自百",
    "百": "白",
    "十": "土士",
    "土": "十士",
    "士": "十土",
    "大": "太犬",
    "太": "大",
    "犬": "大",
    "小": "少尐",
    "少": "小",
    "木": "本",
    "本": "木",
    "工": "土",
    "中": "申由甲",
    "申": "中由甲",
    "由": "中申甲",
    "甲": "中申由",
    "一": "l-",
    "二": "=",
    "三": "≡",
    "个": "介",
    "介": "个",
    "力": "刀",
    "刀": "力",
    "九": "丸",
    "丸": "九",
    "了": "T",
    "山": "Ш",
    "川": "Ⅲ",
    "千": "T干",
    "干": "千T",
    "手": "于",
    "于": "手",
    "王": "主玉",
    "主": "王玉",
    "玉": "王主",
}

# Combined dictionaries for easy access
MULTILINGUAL_KEYBOARDS = {
    LANGS.en: ENGLISH_KEYBOARD_NEIGHBORS,
    LANGS.fa: FARSI_KEYBOARD_NEIGHBORS,
    LANGS.tr: TURKISH_KEYBOARD_NEIGHBORS,
    LANGS.zh: CHINESE_KEYBOARD_NEIGHBORS,
}

MULTILINGUAL_OCR_ERRORS = {
    LANGS.en: ENGLISH_OCR_ERRORS,
    LANGS.fa: FARSI_OCR_ERRORS,
    LANGS.tr: TURKISH_OCR_ERRORS,
    LANGS.zh: CHINESE_OCR_ERRORS,
}


# Utility functions
def generate_keyboard_errors(word, language: Optional[LANGS] = LANGS.en, max_errors=1):
    """
    Generate possible keyboard proximity errors for a word.

    Args:
        word (str): Input word
        language (str): Language code
        max_errors (int): Maximum number of character substitutions

    Returns:
        list: List of possible misspelled variants
    """
    if language not in MULTILINGUAL_KEYBOARDS:
        return [word]

    keyboard = MULTILINGUAL_KEYBOARDS[language]
    variants = set()

    for i, char in enumerate(word):
        if char.lower() in keyboard:
            for neighbor in keyboard[char.lower()]:
                if char.isupper():
                    neighbor = neighbor.upper()
                variant = word[:i] + neighbor + word[i + 1 :]
                variants.add(variant)

    return list(variants)


def generate_ocr_errors(word, language: Optional[LANGS] = LANGS.en):
    """
    Generate possible OCR errors for a word.

    Args:
        word (str): Input word
        language (str): Language code

    Returns:
        list: List of possible OCR error variants
    """
    if language not in MULTILINGUAL_OCR_ERRORS:
        return [word]

    ocr_dict = MULTILINGUAL_OCR_ERRORS[language]
    variants = set()

    for original, alternatives in ocr_dict.items():
        if original in word:
            for alt in alternatives:
                variant = word.replace(original, alt)
                variants.add(variant)

    return list(variants)


keyboard_proximity_errors = Perturbation(
    "Keyboard proximity errors",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    func=generate_keyboard_errors,
    category="Noise",
)
noise_injection = Perturbation(
    "Noise injection",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Noise",
)
ocr_errors = Perturbation(
    "OCR errors",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    func=generate_ocr_errors,
    category="Noise",
)
permutations = Perturbation(
    "Permutations",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Noise",
)
typographical_errors = Perturbation(
    "Typographical errors",
    available_languages=[LANGS.en, LANGS.it, LANGS.zh, LANGS.fa, LANGS.tr],
    automatable=True,
    category="Noise",
)


Noise = {
    "keyboard_proximity_errors": keyboard_proximity_errors,
    "noise_injection": noise_injection,
    "ocr_errors": ocr_errors,
    "permutations": permutations,
    "typographical_errors": typographical_errors,
}


# Example usage
if __name__ == "__main__":
    # Test examples for each language
    test_words = {
        LANGS.en: "hello",
        LANGS.fa: "سلام",
        LANGS.tr: "merhaba",
        LANGS.zh: "你好",
    }

    for lang, word in test_words.items():
        print(f"\n{lang.value.upper()} - '{word}':")
        errors = generate_all_error_types(word, lang)
        for error_type, variants in errors.items():
            print(f"  {error_type}: {variants[:5]}")  # Show first 5 variants
