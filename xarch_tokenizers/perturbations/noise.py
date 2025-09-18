# Noise	Keyboard proximity errors
# Noise	Noise injection
# Noise	OCR Errors
# Noise	Permutations
# Noise	Typographical errors

import logging
from os import replace
from random import shuffle
from typing import List, Optional

import numpy as np
from attr import dataclass
from wandb import login

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

logger = logging.getLogger(__name__)
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

# more detailed Persian keyboard:
persian_keyboard_neighbors = {
    # First letter row (QWERTY equivalent)
    "ا": {"up": "غ", "down": "د", "left": "ل", "right": "ت"},
    "ب": {"up": "ق", "down": "ر", "left": "ی", "right": "ل"},
    "پ": {"up": "ت", "down": None, "left": "د", "right": "و"},
    "ت": {"up": "ع", "down": "پ", "left": "ا", "right": "ن"},
    "ث": {"up": "۳", "down": "ی", "left": "ص", "right": "ق"},
    "ج": {"up": "-", "down": "گ", "left": "ح", "right": "چ"},
    "چ": {"up": "=", "down": "\\", "left": "ج", "right": None},
    "ح": {"up": "۰", "down": "ک", "left": "خ", "right": "ج"},
    "خ": {"up": "۹", "down": "م", "left": "ه", "right": "ح"},
    "د": {"up": "ا", "down": None, "left": "ذ", "right": "پ"},
    "ذ": {"up": "ل", "down": None, "left": "ر", "right": "د"},
    "ر": {"up": "ب", "down": None, "left": "ز", "right": "ذ"},
    "ز": {"up": "ی", "down": None, "left": "ط", "right": "ر"},
    "ژ": {"up": "ی", "down": None, "left": "ط", "right": "ر"},
    "س": {"up": "ص", "down": "ط", "left": "ش", "right": "ی"},
    "ش": {"up": "ض", "down": "ظ", "left": None, "right": "س"},
    "ص": {"up": "۲", "down": "س", "left": "ض", "right": "ث"},
    "ض": {"up": "۱", "down": "ش", "left": None, "right": "ص"},
    "ط": {"up": "س", "down": None, "left": "ظ", "right": "ز"},
    "ظ": {"up": "ش", "down": None, "left": None, "right": "ط"},
    "ع": {"up": "۷", "down": "ت", "left": "غ", "right": "ه"},
    "غ": {"up": "۶", "down": "ا", "left": "ف", "right": "ع"},
    "ف": {"up": "۵", "down": "ل", "left": "ق", "right": "غ"},
    "ق": {"up": "۴", "down": "ب", "left": "ث", "right": "ف"},
    "ک": {"up": "ح", "down": "/", "left": "م", "right": "گ"},
    "گ": {"up": "ج", "down": None, "left": "ک", "right": "\\"},
    "ل": {"up": "ف", "down": "ذ", "left": "ب", "right": "ا"},
    "م": {"up": "خ", "down": ".", "left": "ن", "right": "ک"},
    "ن": {"up": "ه", "down": "و", "left": "ت", "right": "م"},
    "و": {"up": "ن", "down": None, "left": "پ", "right": "."},
    "ه": {"up": "۸", "down": "ن", "left": "ع", "right": "خ"},
    "ی": {"up": "ث", "down": "ز", "left": "س", "right": "ب"},
}

# Farsi/Persian Keyboard Layout (based on standard Persian QWERTY)
FARSI_KEYBOARD_NEIGHBORS = {
    "ا": "غدلت",
    "ب": "قریل",
    "پ": "تدو",
    "ت": "عپان",
    "ث": "۳یصق",
    "ج": "-گحچ",
    "چ": "=\\ج",
    "ح": "۰کخج",
    "خ": "۹مهح",
    "د": "اذپ",
    "ذ": "لرد",
    "ر": "بزذ",
    "ز": "یطر",
    "ژ": "یطر",
    "س": "صطشی",
    "ش": "ضظس",
    "ص": "۲سضث",
    "ض": "۱شص",
    "ط": "سظز",
    "ظ": "شط",
    "ع": "۷تغه",
    "غ": "۶افع",
    "ف": "۵لقغ",
    "ق": "۴بثف",
    "ک": "ح/مگ",
    "گ": "جک\\",
    "ل": "فذبا",
    "م": "خ.نک",
    "ن": "هوتم",
    "و": "نپ.",
    "ه": "۸نعخ",
    "ی": "ثزسب",
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

# Chinese Character Keyboard Neighbors
# For simulating character selection errors after correct pinyin input

CHINESE_CHARACTER_NEIGHBORS = {
    # Common characters grouped by similar pinyin or IME position
    # Characters that often appear together in IME candidate lists
    # 的/地/得 - de variations (most common mix-up)
    "的": "地得",
    "地": "的得",
    "得": "的地",
    # 在/再 - zai
    "在": "再",
    "再": "在",
    # 做/作 - zuo
    "做": "作",
    "作": "做",
    # 和/或/何 - he/huo
    "和": "或何",
    "或": "和何",
    "何": "和或",
    # 了/le variations
    "了": "乐勒",
    "乐": "了勒",
    "勒": "了乐",
    # 是/十/时/实/石 - shi
    "是": "十时实石",
    "十": "是时实石",
    "时": "是十实石",
    "实": "是十时石",
    "石": "是十时实",
    # 不/部/步/布 - bu
    "不": "部步布",
    "部": "不步布",
    "步": "不部布",
    "布": "不部步",
    # 一/以/已/义/意/易 - yi
    "一": "以已义意易",
    "以": "一已义意易",
    "已": "一以义意易",
    "义": "一以已意易",
    "意": "一以已义易",
    "易": "一以已义意",
    # 人/任/认 - ren
    "人": "任认",
    "任": "人认",
    "认": "人任",
    # 会/回/汇/慧 - hui
    "会": "回汇慧",
    "回": "会汇慧",
    "汇": "会回慧",
    "慧": "会回汇",
    # 个/各/哥/歌 - ge
    "个": "各哥歌",
    "各": "个哥歌",
    "哥": "个各歌",
    "歌": "个各哥",
    # 上/尚/商 - shang
    "上": "尚商",
    "尚": "上商",
    "商": "上尚",
    # 下/夏 - xia
    "下": "夏",
    "夏": "下",
    # 大/打/达 - da
    "大": "打达",
    "打": "大达",
    "达": "大打",
    # 小/校/笑 - xiao
    "小": "校笑",
    "校": "小笑",
    "笑": "小校",
    # 中/众/重/钟 - zhong
    "中": "众重钟",
    "众": "中重钟",
    "重": "中众钟",
    "钟": "中众重",
    # 国/果/过 - guo
    "国": "果过",
    "果": "国过",
    "过": "国果",
    # 家/加/假/价 - jia
    "家": "加假价",
    "加": "家假价",
    "假": "家加价",
    "价": "家加假",
    # 年/念 - nian
    "年": "念",
    "念": "年",
    # 月/约/越 - yue
    "月": "约越",
    "约": "月越",
    "越": "月约",
    # 日/入 - ri/ru
    "日": "入",
    "入": "日",
    # 水/谁 - shui
    "水": "谁",
    "谁": "水",
    # 火/或/活 - huo
    "火": "或活",
    "或": "火活",
    "活": "火或",
    # 土/图/兔/吐 - tu
    "土": "图兔吐",
    "图": "土兔吐",
    "兔": "土图吐",
    "吐": "土图兔",
    # 木/目/母 - mu
    "木": "目母",
    "目": "木母",
    "母": "木目",
    # 金/今/斤 - jin
    "金": "今斤",
    "今": "金斤",
    "斤": "金今",
    # 来/赖/莱 - lai
    "来": "赖莱",
    "赖": "来莱",
    "莱": "来赖",
    # 去/区/取/趣 - qu
    "去": "区取趣",
    "区": "去取趣",
    "取": "去区趣",
    "趣": "去区取",
    # 好/号 - hao
    "好": "号",
    "号": "好",
    # 看/刊/堪 - kan
    "看": "刊堪",
    "刊": "看堪",
    "堪": "看刊",
    # 说/朔/烁 - shuo
    "说": "朔烁",
    "朔": "说烁",
    "烁": "说朔",
    # 话/画/花/华/化 - hua
    "话": "画花华化",
    "画": "话花华化",
    "花": "话画华化",
    "华": "话画花化",
    "化": "话画花华",
    # 走/邹/奏 - zou
    "走": "邹奏",
    "邹": "走奏",
    "奏": "走邹",
    # 起/齐/气/期/启 - qi
    "起": "齐气期启",
    "齐": "起气期启",
    "气": "起齐期启",
    "期": "起齐气启",
    "启": "起齐气期",
    # 把/爸/把/吧 - ba
    "把": "爸吧",
    "爸": "把吧",
    "吧": "把爸",
    # 让/攘 - rang
    "让": "攘",
    "攘": "让",
    # 给/给 - gei/ji (pronunciation variants)
    "给": "给",  # Same character, different pronunciations
    # Numbers
    "零": "龄令",
    "一": "医衣依",
    "二": "儿尔耳",
    "三": "散伞",
    "四": "思死丝司",
    "五": "午武舞",
    "六": "流刘留",
    "七": "妻期器",
    "八": "发法发",
    "九": "久酒救",
    # Colors
    "红": "洪虹弘",
    "黄": "皇煌慌",
    "蓝": "兰览懒",
    "绿": "录路陆",
    "白": "百摆败",
    "黑": "嘿",
    # Common verbs with similar sounds
    "吃": "池迟",
    "喝": "合河何",
    "睡": "水谁",
    "醒": "星姓",
    "买": "卖麦",
    "卖": "买麦",
    "开": "凯楷",
    "关": "观官",
    "进": "近尽",
    "出": "初触",
    # Pronouns and common words
    "我": "握沃",
    "你": "拟逆",
    "他": "她它塔",
    "她": "他它塔",
    "它": "他她塔",
    "这": "者浙",
    "那": "哪纳",
    "哪": "那纳",
    "什": "十时实",
    "么": "末摸",
    "吗": "妈马",
    "呢": "泥尼",
    # Time words
    "今": "金斤",
    "明": "鸣名",
    "昨": "作做",
    "早": "澡枣燥",
    "晚": "完碗",
    "午": "五舞武",
    "夜": "也页",
    # Body parts
    "头": "投偷",
    "眼": "演掩",
    "口": "扣",
    "手": "受守首",
    "脚": "交教叫",
    "心": "新信",
    # Food items
    "饭": "犯范泛",
    "菜": "材才财",
    "肉": "柔",
    "鱼": "于余渔",
    "面": "棉免",
    "汤": "糖塘堂",
    "茶": "查察",
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
    LANGS.eng_Latn: ENGLISH_KEYBOARD_NEIGHBORS,
    LANGS.pes_Arab: FARSI_KEYBOARD_NEIGHBORS,
    LANGS.tur_Latn: TURKISH_KEYBOARD_NEIGHBORS,
    LANGS.zho_Hans: CHINESE_KEYBOARD_NEIGHBORS,
}

MULTILINGUAL_OCR_ERRORS = {
    LANGS.eng_Latn: ENGLISH_OCR_ERRORS,
    LANGS.pes_Arab: FARSI_OCR_ERRORS,
    LANGS.tur_Latn: TURKISH_OCR_ERRORS,
    LANGS.zho_Hans: CHINESE_OCR_ERRORS,
}


# Utility functions
def generate_keyboard_errors(
    question,
    language: Optional[LANGS] = LANGS.eng_Latn,
    n_variations: int = 1,
    max_errors_in_one_sample: int = 1,
    sampling_rate=1.0,
):
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
        return [question]
    if n_variations == -1:
        logger.info("Generating all possible variations.")

    keyboard = MULTILINGUAL_KEYBOARDS[language]
    variants = set()
    perturbed_indices = set()

    for _ in range(n_variations):
        indx = np.random.choice(len(question), max_errors_in_one_sample, replace=False)
        indx.sort()
        variant = question
        perturbed = False
        for ind in indx:
            char = question[ind]
            if char.lower() not in keyboard:
                continue
            replacement = np.random.choice(list(keyboard[char.lower()]), size=1)[0]
            # correct casing
            replacement = replacement.upper() if char.isupper() else replacement
            perturbed_indices.add(ind)
            variant = variant[:ind] + replacement + variant[ind + 1 :]
        if perturbed:
            variants.add(variant)

    return list(variants)


def generate_ocr_errors(
    question,
    language: Optional[LANGS] = LANGS.eng_Latn,
    n_variations: int = 1,
    max_errors_in_one_sample: int = 1,
):
    """
    Generate possible OCR errors for a question.

    Args:
        question (str): Input question
        language (str): Language code

    Returns:
        list: List of possible OCR error variants
    """
    if language not in MULTILINGUAL_OCR_ERRORS:
        return [question]

    ocr_dict = MULTILINGUAL_OCR_ERRORS[language]
    variants = set()

    # TODO: not replace all
    replacable_indices = [
        i for i, original in enumerate(question) if original in ocr_dict
    ]
    for _ in range(n_variations):
        replace_count = np.random.randint(1, max_errors_in_one_sample + 1)
        replace_inds = np.random.choice(
            replacable_indices, size=replace_count, replace=False
        )
        variant = question
        for i in replace_inds:
            char = question[i]
            alternatives = list(ocr_dict[char])
            variant = variant[:i] + np.random.choice(alternatives) + variant[i + 1 :]
        variants.add(variant)

    return list(variants)


keyboard_proximity_errors = Perturbation(
    "Keyboard proximity errors",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    func=generate_keyboard_errors,
    category="Noise",
)
noise_injection = Perturbation(
    "Noise injection",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Noise",
)
ocr_errors = Perturbation(
    "OCR errors",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    func=generate_ocr_errors,
    category="Noise",
)
permutations = Perturbation(
    "Permutations",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
    automatable=True,
    category="Noise",
)
typographical_errors = Perturbation(
    "Typographical errors",
    available_languages=[
        LANGS.eng_Latn,
        LANGS.ita_Latn,
        LANGS.zho_Hans,
        LANGS.pes_Arab,
        LANGS.tur_Latn,
    ],
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
