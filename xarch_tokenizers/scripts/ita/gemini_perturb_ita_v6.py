import argparse
import random
import re
import functools

# Set a fixed seed for reproducibility of random perturbations
random.seed(42)

# --- Punctuation Helper & Decorator ---

def _strip_punctuation(word):
    """Separates leading/trailing punctuation from the core word."""
    # This regex finds a prefix of non-word chars, a core of word chars, and a suffix of non-word chars.
    match = re.match(r'^([\W_]*)(\w*)([\W_]*)$', word)
    if match:
        return match.groups()  # (prefix, core, suffix)
    return ('', word, '') # Fallback for words without a clear core

def preserves_punctuation(func):
    """
    A decorator that strips punctuation from a word, applies the perturbation
    function to the core word, and then reattaches the punctuation.
    """
    @functools.wraps(func)
    def wrapper(word, *args, **kwargs):
        prefix, core_word, suffix = _strip_punctuation(word)
        if not core_word:  # Don't perturb if there's no core word (e.g., just punctuation)
            return (word, "") if "random" in func.__name__ else []
        
        # Call the original perturbation function on the core word
        result = func(core_word, *args, **kwargs)

        # Re-attach punctuation
        if "exhaustive" in func.__name__:
            # For exhaustive funcs returning a list of tuples
            return [(f"{prefix}{p_word}{suffix}", replacement) for p_word, replacement in result]
        else:
            # For random funcs returning a single tuple
            perturbed_word, replacement = result
            return f"{prefix}{perturbed_word}{suffix}", replacement
    return wrapper


# --- Perturbation Configuration ---

# Keyboard layout for proximity errors (simplified Italian layout)
KEYBOARD_NEIGHBORS = {
    'a': 'qwsz', 'b': 'vgn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
    'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'l': 'kopà',
    'm': 'njk,', 'n': 'bhjm', 'o': 'iklp', 'p': 'olà', 'q': 'wa',
    'r': 'edfgt', 's': 'wedxza', 't': 'rfgyh', 'u': 'yihj', 'v': 'cfgb',
    'x': 'zsdc', 'y': 'tghu', 'z': 'asx', 'è': 'é', 'é': 'è',
    'à': 'ò', 'ò': 'à', 'ù': 'ì', 'ì': 'ù',
}

# Common OCR-like errors
OCR_ERRORS = {
    'o': '0', 'l': '1', 'i': '1', 'a': '4', 's': '5', 'g': '9',
    'e': '3', 'b': '8', 'rn': 'm', 'ri': 'n', 'cl': 'd'
}

# Homoglyphs (visually similar characters)
HOMOGLYPHS = {
    'o': 'о',  # Latin 'o' -> Cyrillic 'о'
    'l': 'I',  # Latin 'l' -> Latin capital 'I'
    'a': 'а',  # Latin 'a' -> Cyrillic 'а'
    'e': 'е',  # Latin 'e' -> Cyrillic 'е'
    'c': 'с',  # Latin 'c' -> Cyrillic 'с'
}

# --- Quote Swapping Configuration ---
SWAPPABLE_SINGLE_QUOTES = ["'", "‘", "’", "‹", "›", "′", "‚", "‵", "❛", "❜", "‛"]
SWAPPABLE_DOUBLE_QUOTES = ['"', '“', '”', '«', '»', "″", "‴", "⁗", "「", "」", "『", "』", "„", "‶", "❝", "❞", "‟"]
# For coherent pair swapping
QUOTE_PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ('“', '”'),
    ('‘', '’'),
    ('«', '»'),
    ('「', '」'),
    ('『', '』'),
]


# Italian-specific configurations
ITALIAN_CONTRACTIONS_SPLIT = {
    'del': 'de il', 'dello': 'de lo', 'della': 'de la', 'dei': 'de i',
    'degli': 'de gli', 'delle': 'de le', 'dell\'': 'de l\'',
    'al': 'a il', 'allo': 'a lo', 'alla': 'a la', 'ai': 'a i',
    'agli': 'a gli', 'alle': 'a le', 'all\'': 'a l\'',
    'dal': 'da il', 'dallo': 'da lo', 'dalla': 'da la', 'dai': 'da i',
    'dagli': 'da gli', 'dalle': 'da le', 'dall\'': 'da l\'',
    'nel': 'in il', 'nello': 'in lo', 'nella': 'in la', 'nei': 'in i',
    'negli': 'in gli', 'nelle': 'in le', 'nell\'': 'in l\'',
    'col': 'con il', 'coi': 'con i',
    'sul': 'su il', 'sullo': 'su lo', 'sulla': 'su la', 'sui': 'su i',
    'sugli': 'su gli', 'sulle': 'su le', 'sull\'': 'su l\'',
}

ITALIAN_CLITICS_SPLIT = {
    'glielo': 'glie lo', 'gliela': 'glie la', 'glieli': 'glie li', 'gliele': 'glie le',
    'melo': 'me lo', 'mela': 'me la', 'meli': 'me li', 'mele': 'me le',
    'telo': 'te lo', 'tela': 'te la', 'teli': 'te li', 'tele': 'te le',
    'celo': 'ce lo', 'cela': 'ce la', 'celi': 'ce li', 'cele': 'ce le',
    'velo': 've lo', 'vela': 've la', 'veli': 've li', 'vele': 've le',
    'dammelo': 'da me lo', 'dimmelo': 'di me lo', 'fallo': 'fa lo', 'dallo': 'da lo',
    'farcelo': 'far ce lo', 'portamelo': 'porta me lo'
}

# Accent variation maps
ACCENT_VARIATIONS = {
    'a': ['a', 'à', 'á', 'â', 'ä', 'ã', "a'"],
    'e': ['e', 'è', 'é', 'ê', 'ë', "e'"],
    'i': ['i', 'ì', 'í', 'î', 'ï', "i'"],
    'o': ['o', 'ò', 'ó', 'ô', 'ö', 'õ', "o'"],
    'u': ['u', 'ù', 'ú', 'û', 'ü', "u'"],
}
BASE_VOWEL_MAP = {
    'à': 'a', 'á': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a',
    'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
    'ò': 'o', 'ó': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
    'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
}

# Title and Profession variations
TITLE_VARIATIONS = [
    {'signore', 'signor', 'sig.', 'sig', 'sig.re', 'signora', 'sig.ra'},
    {'dottore', 'dottor', 'dott.', 'dott', 'dr.', 'dr', 'dottoressa', 'dott.ssa', 'dr.ssa'},
    {'professore', 'professor', 'prof.', 'prof', 'professoressa', 'prof.ssa'},
    {'ingegnere', 'ingegner', 'ing.', 'ing'},
    {'avvocato', 'avv.', 'avv', 'avvocatessa'},
]

# Phonetic spelling variations
PHONETIC_SPELLINGS = {
    "che": "ke", "chi": "ki",
    "co": "ko", "ca": "ka", "cu": "ku",
    "ti": "t", "ci": "c",
    "ghe": "ge", "ghi": "gi",
    "scia": "sha", "sce": "she", "sci": "shi", "scio": "sho", "sciu": "shu",
    "gni": "gny", "gna": "gnya",
    "qua": "cua", "que": "cue", "qui": "cui", "quo": "cuo"
}

# 'h' omission for the verb "avere"
AVERE_H_OMISSION = {
    'ho': 'o',
    'ha': 'a',
    'hanno': 'anno',
}

# SMS / text-speak spelling variations
SMS_SPELLINGS = {
    "per": "x",
    "non": "nn",
    "sei": "6",
    "qualche": "qlc",
    "qualcosa": "qls",
    "comunque": "cmq",
    "messaggio": "msg",
    "dove": "dv",
    "domanda": "dom",
    "aspetta": "asp",
    "dopo": "dp",
    "tutto": "tt",
    "di": "d",
    "vi": "v",
    "troppo": "trp",
    "bene": "bn",
    "grazie": "grz",
    "sono": "sn"
}


# --- RANDOM Perturbation Functions (return tuple: (perturbed_word, replacement_str)) ---

@preserves_punctuation
def p_typographical_error_random(word):
    if word.isdigit() or (word.endswith('%') and word[:-1].isdigit()):
        return word, ""
    if len(word) < 2: return word, ""
    pos = random.randint(0, len(word) - 1)
    op = random.choice(['insert', 'delete', 'substitute'])
    if op == 'insert':
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:pos] + char + word[pos:], char
    elif op == 'delete':
        return word[:pos] + word[pos+1:], ""
    else: # substitute
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:pos] + char + word[pos+1:], char

@preserves_punctuation
def p_keyboard_proximity_error_random(word):
    chars = list(word)
    eligible_indices = [i for i, char in enumerate(chars) if char.lower() in KEYBOARD_NEIGHBORS]
    if not eligible_indices: return word, ""
    idx = random.choice(eligible_indices)
    char_to_replace = chars[idx].lower()
    neighbor = random.choice(KEYBOARD_NEIGHBORS[char_to_replace])
    chars[idx] = neighbor.upper() if chars[idx].isupper() else neighbor
    return "".join(chars), chars[idx]

@preserves_punctuation
def p_ocr_error_random(word):
    keys = list(OCR_ERRORS.keys())
    random.shuffle(keys)
    for target in keys:
        if target in word:
            replacement = OCR_ERRORS[target]
            return word.replace(target, replacement, 1), replacement
    return word, ""

@preserves_punctuation
def p_permutation_error_random(word):
    if word.isdigit() or (word.endswith('%') and word[:-1].isdigit()):
        return word, ""
    if len(word) < 2: return word, ""
    pos = random.randint(0, len(word) - 2)
    perturbed_word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
    return perturbed_word, perturbed_word

@preserves_punctuation
def p_change_capitalization_random(word):
    op = random.choice(['lower', 'upper', 'title'])
    if op == 'lower': 
        p_word = word.lower()
        return p_word, p_word
    if op == 'upper':
        p_word = word.upper()
        return p_word, p_word
    p_word = word.title()
    return p_word, p_word

@preserves_punctuation
def p_add_homoglyph_random(word):
    chars = list(word)
    eligible_indices = [i for i, char in enumerate(chars) if char in HOMOGLYPHS]
    if not eligible_indices: return word, ""
    idx = random.choice(eligible_indices)
    replacement = HOMOGLYPHS[chars[idx]]
    chars[idx] = replacement
    return "".join(chars), replacement

@preserves_punctuation
def p_add_zero_width_char_random(word):
    if len(word) < 2: return word, ""
    pos = random.randint(1, len(word) - 1)
    replacement = u'\u200b'
    return word[:pos] + replacement + word[pos:], replacement

@preserves_punctuation
def p_repeat_letters_random(word):
    if len(word) < 3: return word, ""
    vowel_indices = [i for i, char in enumerate(word) if char.lower() in 'aeiou']
    if not vowel_indices: return word, ""
    pos = random.choice(vowel_indices)
    p_word = word[:pos+1] + (word[pos] * random.randint(1, 2)) + word[pos+1:]
    return p_word, p_word

# --- EXHAUSTIVE Perturbation Functions (return list of tuples: [(word, replacement)]) ---

def _num_to_words_it_helper(num):
    """Helper to convert numbers < 1000 to Italian words."""
    if num == 0: return ""
    units = ["", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"]
    teens = ["dieci", "undici", "dodici", "tredici", "quattordici", "quinzici", "sedici", "diciassette", "diciotto", "diciannove"]
    tens = ["", "dieci", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"]
    
    if num < 10: return units[num]
    if num < 20: return teens[num - 10]
    if num < 100:
        ten, unit = divmod(num, 10)
        if unit in [1, 8]: return tens[ten][:-1] + units[unit]
        return tens[ten] + units[unit]
    if num < 1000:
        hundred, rest = divmod(num, 100)
        hundred_str = "cento" if hundred == 1 else units[hundred] + "cento"
        return hundred_str + _num_to_words_it_helper(rest)
    return ""

@preserves_punctuation
def p_number_format_exhaustive(word):
    clean_word = re.sub(r'[.,\s]', '', word)
    if not clean_word.isdigit(): return []
    
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
            if prefix_num >= 100 and prefix_num % 100 == 0:
                hundreds_part = prefix_num // 100
                hundreds_words = _num_to_words_it_helper(hundreds_part)
                if hundreds_words:
                    options.add(f"{hundreds_words} cento mila")

    if 0 <= num <= 100:
        if num == 0:
            options.add("zero")
        else:
            word_form = _num_to_words_it_helper(num)
            if word_form:
                options.add(word_form)

    return sorted([(opt, opt) for opt in options if opt != word])

@preserves_punctuation
def p_percentage_symbol_exhaustive(word):
    if word == '%':
        replacements = ['per cento', 'percento', 'xcento']
        return [(r, r) for r in replacements]
    return []

# This function should NOT use the decorator, as it needs to see the '%' sign
def p_percentage_exhaustive(word):
    match = re.fullmatch(r'(\d+)(%)', word)
    if not match: return []

    num_str, symbol = match.groups()
    results = set()
    
    # Manually call the decorated number formatter on the number part
    num_variations_tuples = p_number_format_exhaustive(num_str) + [(num_str, num_str)]
    symbol_variations_tuples = p_percentage_symbol_exhaustive(symbol) + [(symbol, symbol)]

    for num_var, _ in num_variations_tuples:
        for sym_var, _ in symbol_variations_tuples:
            if sym_var == '%':
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
def p_sms_spelling_exhaustive(word):
    word_lower = word.lower()
    if word_lower in SMS_SPELLINGS:
        replacement = str(SMS_SPELLINGS[word_lower])
        if word and word[0].isupper() and replacement.isalpha():
            p_word = replacement.capitalize()
            return [(p_word, p_word)]
        else:
            return [(replacement, replacement)]
    return []

@preserves_punctuation
def p_phonetic_spelling_exhaustive(word):
    results = set()
    word_lower = word.lower()
    
    for original, phonetic in PHONETIC_SPELLINGS.items():
        if original in word_lower:
            perturbed_word_lower = word_lower.replace(original, phonetic)
            if word and word[0].isupper():
                p_word = perturbed_word_lower.capitalize()
                results.add((p_word, phonetic))
            else:
                results.add((perturbed_word_lower, phonetic))
    return sorted(list(results), key=lambda x: x[0])

@preserves_punctuation
def p_double_letter_error_exhaustive(word):
    results = set()
    for i in range(len(word) - 1):
        if word[i].lower() == word[i+1].lower():
            p_word = word[:i] + word[i+1:]
            replacement = word[i]
            results.add((p_word, replacement))
    return sorted(list(results), key=lambda x: x[0])

@preserves_punctuation
def p_h_omission_exhaustive(word):
    results = set()
    word_lower = word.lower()

    if word_lower in AVERE_H_OMISSION:
        replacement = AVERE_H_OMISSION[word_lower]
        if word and word[0].isupper():
            p_word = replacement.capitalize()
            results.add((p_word, replacement))
        else:
            results.add((replacement, replacement))

    if 'ch' in word_lower:
        p_word = word.replace('ch', 'c').replace('Ch', 'C')
        results.add((p_word, 'c'))
    if 'gh' in word_lower:
        p_word = word.replace('gh', 'g').replace('Gh', 'G')
        results.add((p_word, 'g'))
    return sorted(list(results), key=lambda x: x[0])

@preserves_punctuation
def p_title_variation_exhaustive(word):
    word_lower = word.lower()
    is_capitalized = word[0].isupper() if word else False

    for variation_set in TITLE_VARIATIONS:
        if word_lower in variation_set:
            results = variation_set - {word_lower}
            if is_capitalized:
                p_words = [v[0].upper() + v[1:] for v in results if v]
                return sorted([(p, p) for p in p_words], key=lambda x: x[0])
            else:
                return sorted([(p, p) for p in results], key=lambda x: x[0])
    return []

@preserves_punctuation
def p_accent_variation_exhaustive(word):
    results = set()
    for i, char in enumerate(word):
        char_lower = char.lower()
        base_vowel = BASE_VOWEL_MAP.get(char_lower, char_lower if char_lower in ACCENT_VARIATIONS else None)
        
        if base_vowel:
            variations = ACCENT_VARIATIONS[base_vowel]
            for var in variations:
                if var != char_lower:
                    new_char = var.upper() if char.isupper() else var
                    p_word = word[:i] + new_char + word[i+1:]
                    results.add((p_word, new_char))
    return sorted(list(results), key=lambda x: x[0])

@preserves_punctuation
def p_currency_format_exhaustive(word):
    currency_map = {'€': 'euro', 'euro': '€', 'eur': '€', '$': 'dollaro', 'dollaro': '$', 'usd': '$'}
    if word.lower() in currency_map:
        replacement = currency_map[word.lower()]
        return [(replacement, replacement)]
    return []

@preserves_punctuation
def p_date_format_exhaustive(word):
    if re.fullmatch(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', word):
        current_sep = '/' if '/' in word else '-'
        new_seps = ['.', '/', '-']
        new_seps.remove(current_sep)
        p_words = [word.replace(current_sep, sep) for sep in new_seps]
        return [(p, p) for p in p_words]
    return []

@preserves_punctuation
def p_split_contraction_exhaustive(word):
    replacement = ITALIAN_CONTRACTIONS_SPLIT.get(word.lower())
    if replacement:
        return [(replacement, replacement)]
    return []

@preserves_punctuation
def p_split_clitic_exhaustive(word):
    results = []
    sorted_clitics = sorted(ITALIAN_CLITICS_SPLIT.keys(), key=len, reverse=True)
    for clitic in sorted_clitics:
        if word.lower().endswith(clitic):
            base = word[:-len(clitic)]
            replacement = ITALIAN_CLITICS_SPLIT[clitic]
            p_word = base + replacement
            results.append((p_word, replacement))
    return results

def p_internal_space_exhaustive(word):
    prefix, core, suffix = _strip_punctuation(word)
    results = set()
    if not core or (not prefix and not suffix):
        return []

    # Case 1: Space after prefix only
    if prefix:
        for i in range(1, 11):
            space = " " * i
            new_word = f"{prefix}{space}{core}{suffix}"
            results.add((new_word, new_word))

    # Case 2: Space before suffix only
    if suffix:
        for i in range(1, 11):
            space = " " * i
            new_word = f"{prefix}{core}{space}{suffix}"
            results.add((new_word, new_word))

    # Case 3: Space on both sides (if both prefix and suffix exist)
    if prefix and suffix:
        for i in range(1, 11):
            space = " " * i
            new_word = f"{prefix}{space}{core}{space}{suffix}"
            results.add((new_word, new_word))

    return sorted(list(results), key=lambda x: x[0])

def p_swap_quote_exhaustive(word):
    results = set()
    for i, char in enumerate(word):
        # Swap single quotes/apostrophes
        if char in SWAPPABLE_SINGLE_QUOTES:
            for quote_char in SWAPPABLE_SINGLE_QUOTES:
                if quote_char != char:
                    p_word = word[:i] + quote_char + word[i+1:]
                    results.add((p_word, quote_char))
        # Swap double quotes
        elif char in SWAPPABLE_DOUBLE_QUOTES:
            for quote_char in SWAPPABLE_DOUBLE_QUOTES:
                if quote_char != char:
                    p_word = word[:i] + quote_char + word[i+1:]
                    results.add((p_word, quote_char))

    return sorted(list(results), key=lambda x: x[0])

def p_swap_quote_pair_exhaustive(word):
    if len(word) < 2:
        return []

    original_left, original_right = None, None
    
    # Iterate through possible pairs to find a match for the outer characters
    for left, right in QUOTE_PAIRS:
        if word.startswith(left) and word.endswith(right):
            original_left, original_right = left, right
            break
    
    if original_left is None:
        return []
    
    core_content = word[len(original_left):-len(original_right)]
    original_pair_found = (original_left, original_right)
        
    results = set()
    for new_left, new_right in QUOTE_PAIRS:
        if (new_left, new_right) != original_pair_found:
            p_word = f"{new_left}{core_content}{new_right}"
            replacement = f"{new_left}...{new_right}"
            results.add((p_word, replacement))
            
    return sorted(list(results), key=lambda x: x[0])


# --- Context-Aware Perturbation Functions ---
def p_context_detach_punctuation(words, index):
    word = words[index]
    prefix, core, suffix = _strip_punctuation(word)
    results = set()
    if not core or (not prefix and not suffix):
        return []

    # Detach only prefix
    if prefix and not suffix:
        new_words = words[:index] + [prefix, core] + words[index+1:]
        results.add((" ".join(new_words), f"{prefix} {core}"))

    # Detach only suffix
    if suffix and not prefix:
        new_words = words[:index] + [core, suffix] + words[index+1:]
        results.add((" ".join(new_words), f"{core} {suffix}"))

    # Detach from a word that has both prefix and suffix
    if prefix and suffix:
        # Detach prefix only
        p_words1 = list(words)
        p_words1[index] = f"{core}{suffix}"
        p_words1.insert(index, prefix)
        results.add((" ".join(p_words1), f"{prefix} {core}{suffix}"))

        # Detach suffix only
        p_words2 = list(words)
        p_words2[index] = f"{prefix}{core}"
        p_words2.insert(index + 1, suffix)
        results.add((" ".join(p_words2), f"{prefix}{core} {suffix}"))
        
        # Detach both
        p_words3 = list(words)
        p_words3[index] = core
        p_words3.insert(index + 1, suffix)
        p_words3.insert(index, prefix)
        results.add((" ".join(p_words3), f"{prefix} {core} {suffix}"))
        
    return sorted(list(results), key=lambda x: x[0])

def p_context_euphonic_d(words, index):
    word = words[index].lower()
    next_word = words[index+1].lower() if index + 1 < len(words) else ''
    if not next_word: return []
    
    results = []
    if word in ['e', 'a'] and next_word and next_word[0] not in 'aeiou':
        p_words = words[:]
        replacement = word + 'd'
        p_words[index] = replacement
        results.append((" ".join(p_words), replacement))
    if word in ['ed', 'ad'] and next_word and next_word[0] == word[-2]:
        p_words = words[:]
        replacement = word[:-1]
        p_words[index] = replacement
        results.append((" ".join(p_words), replacement))
    return results

def p_context_merge_contraction(words, index):
    if index + 1 < len(words):
        combo = f"{words[index].lower()} {words[index+1].lower()}"
        merge_map = {v: k for k, v in ITALIAN_CONTRACTIONS_SPLIT.items()}
        if combo in merge_map:
            replacement = merge_map[combo]
            new_words = words[:index] + [replacement] + words[index+2:]
            return [(" ".join(new_words), replacement)]
    return []

def p_context_concatenate(words, index):
    if index >= len(words) - 1: return []
    replacement = words[index] + words[index+1]
    new_words = words[:index] + [replacement] + words[index+2:]
    return [(" ".join(new_words), replacement)]

def p_context_add_space(words, index):
    if index >= len(words) - 1: return []
    prefix = " ".join(words[:index+1])
    suffix = " ".join(words[index+1:])
    results = []
    for i in range(1, 11): # Add 1 to 10 *extra* spaces
        space = " " * i
        perturbed_sentence = prefix + space + suffix
        results.append((perturbed_sentence, perturbed_sentence))
    return results

# --- Main Logic ---

RANDOM_WORD_PERTURBATIONS = {
    "typo": p_typographical_error_random,
    "keyboard": p_keyboard_proximity_error_random,
    "ocr": p_ocr_error_random,
    "permutation": p_permutation_error_random,
    "capitalization": p_change_capitalization_random,
    "homoglyph": p_add_homoglyph_random,
    "zerowidth": p_add_zero_width_char_random,
    "repeat": p_repeat_letters_random,
}

EXHAUSTIVE_WORD_PERTURBATIONS = {
    "sms_spelling": p_sms_spelling_exhaustive,
    "phonetic_spelling": p_phonetic_spelling_exhaustive,
    "double_letter_error": p_double_letter_error_exhaustive,
    "h_omission": p_h_omission_exhaustive,
    "title_variation": p_title_variation_exhaustive,
    "accent_variation": p_accent_variation_exhaustive,
    "number": p_number_format_exhaustive,
    "currency": p_currency_format_exhaustive,
    "date": p_date_format_exhaustive,
    "split_contraction": p_split_contraction_exhaustive,
    "split_clitic": p_split_clitic_exhaustive,
    "percentage": p_percentage_exhaustive,
    "percentage_symbol": p_percentage_symbol_exhaustive,
    "internal_space": p_internal_space_exhaustive,
    "swap_quote": p_swap_quote_exhaustive,
    "swap_quote_pair": p_swap_quote_pair_exhaustive,
}

CONTEXT_AWARE_PERTURBATIONS = {
    "euphonic_d": p_context_euphonic_d,
    "merge_contraction": p_context_merge_contraction,
    "concatenate": p_context_concatenate,
    "remove_space": p_context_concatenate,
    "add_space": p_context_add_space,
    "detach_punctuation": p_context_detach_punctuation,
}

ALL_PERTURBATIONS = {**RANDOM_WORD_PERTURBATIONS, **EXHAUSTIVE_WORD_PERTURBATIONS, **CONTEXT_AWARE_PERTURBATIONS}

def generate_perturbations(input_file, questions_output, num_versions, specific_types, target_words_file):
    
    if '.' in questions_output:
        name, ext = questions_output.rsplit('.', 1)
        metadata_output = f"{name}_metadata.{ext}"
    else:
        metadata_output = f"{questions_output}_metadata"

    target_word_lines = []
    if target_words_file:
        try:
            with open(target_words_file, 'r', encoding='utf-8') as f:
                target_word_lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Error: Target words file not found at '{target_words_file}'")
            return

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(questions_output, 'w', encoding='utf-8', newline='') as q_outfile, \
             open(metadata_output, 'w', encoding='utf-8', newline='') as m_outfile:
            
            total_generated = 0
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) < 2: continue
                
                original_question = parts[0]
                words = original_question.split()
                if not words: continue

                indices_to_perturb = []
                if target_word_lines and line_num <= len(target_word_lines) and target_word_lines[line_num - 1]:
                    target_words_set = {w.lower() for w in target_word_lines[line_num - 1].split()}
                    indices_to_perturb = [i for i, w in enumerate(words) if _strip_punctuation(w)[1].lower() in target_words_set]
                else:
                    target_word_str = parts[-1] if len(parts) > 5 and not target_word_lines else None
                    if target_word_str:
                        indices_to_perturb = [i for i, w in enumerate(words) if _strip_punctuation(w)[1].lower() == target_word_str.lower()]
                    else:
                        indices_to_perturb = list(range(len(words)))

                answers = parts[1:-1] if (len(parts) > 5 and not target_word_lines) else parts[1:]
                types_to_apply = specific_types if specific_types else ALL_PERTURBATIONS.keys()

                for pert_type in types_to_apply:
                    if pert_type not in ALL_PERTURBATIONS:
                        print(f"Warning: Perturbation type '{pert_type}' not recognized. Skipping.")
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
                                perturbed_word, replacement = func(original_word_affected)
                                
                                if perturbed_word != original_word_affected:
                                    generated_results.add((perturbed_word, replacement))

                            for perturbed_word, replacement in generated_results:
                                temp_words = words[:]
                                temp_words[index] = perturbed_word
                                perturbed_question = " ".join(temp_words)
                                
                                question_parts = [perturbed_question] + answers
                                q_outfile.write('\t'.join(map(str, question_parts)) + '\n')
                                
                                metadata_parts = [original_word_affected, str(index), pert_type, replacement, str(line_num)]
                                
                                # Sanitize only the metadata parts for the metadata file
                                sanitized_metadata_parts = [str(p).replace('"', "'") for p in metadata_parts]
                                m_outfile.write('\t'.join(map(str, question_parts + sanitized_metadata_parts)) + '\n')
                                total_generated += 1
                        
                        elif pert_type in EXHAUSTIVE_WORD_PERTURBATIONS:
                            func = EXHAUSTIVE_WORD_PERTURBATIONS[pert_type]
                            
                            perturbed_results = func(original_word_affected)

                            for p_word, replacement in perturbed_results:
                                if p_word != original_word_affected:
                                    temp_words = words[:]
                                    temp_words[index] = p_word
                                    perturbed_question = " ".join(temp_words)

                                    question_parts = [perturbed_question] + answers
                                    q_outfile.write('\t'.join(map(str, question_parts)) + '\n')

                                    metadata_parts = [original_word_affected, str(index), pert_type, replacement, str(line_num)]
                                    sanitized_metadata_parts = [str(p).replace('"', "'") for p in metadata_parts]
                                    m_outfile.write('\t'.join(map(str, question_parts + sanitized_metadata_parts)) + '\n')
                                    total_generated += 1

                        elif pert_type in CONTEXT_AWARE_PERTURBATIONS:
                            func = CONTEXT_AWARE_PERTURBATIONS[pert_type]
                            perturbed_results = func(words, index)
                            for p_sentence, replacement in perturbed_results:
                                if p_sentence != " ".join(words):
                                    question_parts = [p_sentence] + answers
                                    q_outfile.write('\t'.join(map(str, question_parts)) + '\n')
                                    
                                    metadata_parts = [original_word_affected, str(index), pert_type, replacement, str(line_num)]
                                    sanitized_metadata_parts = [str(p).replace('"', "'") for p in metadata_parts]
                                    m_outfile.write('\t'.join(map(str, question_parts + sanitized_metadata_parts)) + '\n')
                                    total_generated += 1
        
        print(f"Successfully generated {total_generated} perturbed examples.")
        print(f"Output saved to: {questions_output} and {metadata_output}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tokenization perturbations into two files: a questions file and a metadata file.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Input file format:
  - Standard: Question\\tCorrectAnswer\\tOpt1\\tOpt2\\tOpt3
  - With Target Word: Question\\tCorrectAnswer\\tOpt1\\tOpt2\\tOpt3\\tTargetWord

Output files:
  - questions_output: A clean file with `PerturbedQuestion\\t...Answers...`
  - [questions_output_name]_metadata.[ext]: An extended file with all info:
    `PerturbedQuestion\\t...Answers...\\tAffectedWord\\tWordIndex\\tPerturbationType\\tReplacement\\tQuestionNumber`
"""
    )
    parser.add_argument("input_file", help="Path to the input TSV file.")
    parser.add_argument("questions_output", help="Path for the output TSV file with questions. The metadata file will be named based on this.")
    parser.add_argument("-n", "--num_versions", type=int, default=1, help="Number of versions for RANDOM perturbation types (e.g., typo, ocr). Ignored for exhaustive types (e.g., number, date).")
    parser.add_argument("--types", nargs='+', choices=sorted(ALL_PERTURBATIONS.keys()), help="Generate only specific perturbation types. If not set, all types are generated.")
    parser.add_argument("--target-words-file", help="Path to a file with space-separated target words, one line per question.")
    args = parser.parse_args()
    generate_perturbations(args.input_file, args.questions_output, args.num_versions, args.types, args.target_words_file)


