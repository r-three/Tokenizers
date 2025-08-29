## Can be automated
# Morphological Challenges	Clitics
# Morphological Challenges	Compounds
# Morphological Challenges	Contractions
## Cannot be automated
# Morphological Challenges	Affixation edge cases
# Morphological Challenges	Derivations
# Morphological Challenges	Inflections

from typing import List, Optional

from attr import dataclass

from xarch_tokenizers.perturbations.common import (
    LANGS,
    Perturbation,
    PerturbationCategory,
)

clitics = Perturbation(
    "Clitics",
    available_languages=[LANGS.en, LANGS.it],
    automatable=True,
    category="Morphological Challenges",
)
compounds = Perturbation(
    "Compounds",
    available_languages=[LANGS.en, LANGS.it],
    automatable=True,
    category="Morphological Challenges",
)
contractions = Perturbation(
    "Contractions",
    available_languages=[LANGS.en, LANGS.it],
    automatable=True,
    category="Morphological Challenges",
)
affixation_edge_cases = Perturbation(
    "Affixation edge cases",
    available_languages=[LANGS.en, LANGS.tr],
    automatable=False,
    category="Morphological Challenges",
)
derivations = Perturbation(
    "Derivations",
    available_languages=[LANGS.tr],
    automatable=False,
    category="Morphological Challenges",
)
inflections = Perturbation(
    "Inflections",
    available_languages=[LANGS.tr],
    automatable=False,
    category="Morphological Challenges",
)

MorphologicalChallenges = {
    "clitics": clitics,
    "compounds": compounds,
    "contractions": contractions,
    "affixation_edge_cases": affixation_edge_cases,
    "derivations": derivations,
    "inflections": inflections,
}

## Italian
ITALIAN_CONTRACTIONS_SPLIT = {
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
}

ITALIAN_CLITICS_SPLIT = {
    "glielo": "glie lo",
    "gliela": "glie la",
    "glieli": "glie li",
    "gliele": "glie le",
    "melo": "me lo",
    "mela": "me la",
    "meli": "me li",
    "mele": "me le",
    "telo": "te lo",
    "tela": "te la",
    "teli": "te li",
    "tele": "te le",
    "celo": "ce lo",
    "cela": "ce la",
    "celi": "ce li",
    "cele": "ce le",
    "velo": "ve lo",
    "vela": "ve la",
    "veli": "ve li",
    "vele": "ve le",
    "dammelo": "da me lo",
    "dimmelo": "di me lo",
    "fallo": "fa lo",
    "dallo": "da lo",
    "farcelo": "far ce lo",
    "portamelo": "porta me lo",
}

## English
ENGLISH_CONTRACTIONS_SPLIT = {
    # Common contractions with 'be'
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "when's": "when is",
    "why's": "why is",
    # Contractions with 'have'
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "could've": "could have",
    "should've": "should have",
    "would've": "would have",
    "might've": "might have",
    "must've": "must have",
    # Contractions with 'will'
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "that'll": "that will",
    # Contractions with 'would'
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    # Negative contractions
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "needn't": "need not",
    "daren't": "dare not",
    # Other common contractions
    "let's": "let us",
    "mayn't": "may not",
    "shan't": "shall not",
    "o'clock": "of the clock",
    "ma'am": "madam",
    "'twas": "it was",
    "'tis": "it is",
    "ne'er": "never",
    "o'er": "over",
    "e'er": "ever",
}

# English doesn't have extensive clitics like Romance languages,
# but has some phrasal combinations
ENGLISH_CLITICS_SPLIT = {
    "gimme": "give me",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "lemme": "let me",
    "dunno": "do not know",
    "kinda": "kind of",
    "sorta": "sort of",
    "outta": "out of",
    "lotta": "lot of",
    "helluva": "hell of a",
    "coupla": "couple of",
}

# Farsi/Persian Contractions and Clitics
FARSI_CONTRACTIONS_SPLIT = {
    # Ezafe constructions (connecting particles)
    "خانه‌ی": "خانه ی",  # house of
    "کتاب‌ی": "کتاب ی",  # book of
    "دوست‌ت": "دوست ت",  # your friend
    "خانه‌تان": "خانه تان",  # your house
    "کار‌م": "کار م",  # my work
    "دست‌ش": "دست ش",  # his/her hand
    # Compound verbs with mi- prefix
    "میخوام": "می خواه م",  # I want
    "میرم": "می رو م",  # I go
    "میام": "می آی م",  # I come
    "میدم": "می ده م",  # I give
    "میگم": "می گوی م",  # I say
    "نمیخوام": "نمی خواه م",  # I don't want
    "نمیرم": "نمی رو م",  # I don't go
    # Colloquial contractions
    "چیکار": "چه کار",  # what work/what to do
    "کجاست": "کجا است",  # where is
    "چراست": "چرا است",  # why is
    "کیست": "کی است",  # who is
}

FARSI_CLITICS_SPLIT = {
    # Pronominal clitics
    "دیدمش": "دید م ش",  # I saw him/her
    "گفتت": "گفت ت",  # told you
    "آوردیش": "آورد ی ش",  # you brought it
    "میدمت": "می ده م ت",  # I give you
    "بردمتون": "برد م تون",  # I took you (plural)
    # Object pronouns with verbs
    "دوستت‌دارم": "دوست ت دار م",  # I love you
    "نمی‌بینمت": "نمی بین م ت",  # I don't see you
    "می‌شناسمش": "می شناس م ش",  # I know him/her
}

# Turkish Contractions (Turkish has extensive agglutination)
TURKISH_CONTRACTIONS_SPLIT = {
    # Informal contractions
    "napıyorsun": "ne yapıyorsun",  # what are you doing
    "napıyorsunuz": "ne yapıyorsunuz",  # what are you (plural) doing
    "naber": "ne haber",  # what's up/what's the news
    "nedemek": "ne demek",  # what does it mean
    # Question word contractions
    "niçin": "ne için",  # why/for what
    "niye": "ne diye",
}

TURKISH_CLITICS_SPLIT = {}

# Chinese Contractions and Clitics (Simplified Chinese)
# Note: Chinese has fewer contractions but some colloquial combinations
CHINESE_CONTRACTIONS_SPLIT = {
    # Colloquial contractions
    "不是": "不 是",  # is not (already separate, but showing structure)
    "这个": "这 个",  # this one
    "那个": "那 个",  # that one
    "什么": "什 么",  # what
    "怎么": "怎 么",  # how
    "为什么": "为 什么",  # why
    # Regional/dialectal contractions (Mandarin)
    "咋样": "怎么 样",  # how about (informal)
    "干嘛": "干 什么",  # what are you doing
    "咋办": "怎么 办",  # what to do
    "木有": "没 有",  # don't have (internet slang)
    # Classical Chinese influenced contractions
    "无论": "无 论",  # regardless of
    "因为": "因 为",  # because
    "所以": "所 以",  # therefore
}

CHINESE_CLITICS_SPLIT = {
    # Particle combinations (less common in Chinese)
    "了吗": "了 吗",  # completion particle + question particle
    "过吗": "过 吗",  # experience particle + question particle
    "的话": "的 话",  # if (conditional marker)
    "而已": "而 已",  # that's all/merely
    # Pronoun + particle combinations
    "我们的": "我们 的",  # our/ours
    "你们的": "你们 的",  # your (plural)
    "他们的": "他们 的",  # their/theirs
    # Verb + complement combinations
    "看见": "看 见",  # see/catch sight of
    "听到": "听 到",  # hear
    "想起": "想 起",  # remember/think of
    "拿起": "拿 起",  # pick up
    "放下": "放 下",  # put down
}

# Combined dictionary for easy access
MULTILINGUAL_CONTRACTIONS = {
    "english": ENGLISH_CONTRACTIONS_SPLIT,
    "farsi": FARSI_CONTRACTIONS_SPLIT,
    "turkish": TURKISH_CONTRACTIONS_SPLIT,
    "chinese": CHINESE_CONTRACTIONS_SPLIT,
    "italian": ITALIAN_CONTRACTIONS_SPLIT,
}

MULTILINGUAL_CLITICS = {
    "english": ENGLISH_CLITICS_SPLIT,
    "farsi": FARSI_CLITICS_SPLIT,
    "turkish": TURKISH_CLITICS_SPLIT,
    "chinese": CHINESE_CLITICS_SPLIT,
    "italian": ITALIAN_CLITICS_SPLIT,
}


# Usage example function
def split_contractions(text, language="english"):
    """
    Split contractions in the given text based on the specified language.

    Args:
        text (str): Input text containing contractions
        language (str): Language code ('english', 'farsi', 'turkish', 'chinese')

    Returns:
        str: Text with contractions split
    """
    if language not in MULTILINGUAL_CONTRACTIONS:
        return text

    contractions_dict = MULTILINGUAL_CONTRACTIONS[language]
    clitics_dict = MULTILINGUAL_CLITICS[language]

    # Split contractions
    for contraction, split_form in contractions_dict.items():
        text = text.replace(contraction, split_form)

    # Split clitics
    for clitic, split_form in clitics_dict.items():
        text = text.replace(clitic, split_form)

    return text


# Example usage:
if __name__ == "__main__":
    # Test examples
    english_text = "I can't believe you won't help us!"
    farsi_text = "میخوام برم خانه‌تان"
    turkish_text = "Napıyorsun? Evimde kitabını okuyor."
    chinese_text = "我们的朋友看见了什么？"

    print("English:", split_contractions(english_text, "english"))
    print("Farsi:", split_contractions(farsi_text, "farsi"))
    print("Turkish:", split_contractions(turkish_text, "turkish"))
    print("Chinese:", split_contractions(chinese_text, "chinese"))
