import argparse
import random
import re
import unicodedata

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

# --- RANDOM Perturbation Functions (return one string) ---

def p_typographical_error(word):
    if len(word) < 2: return word
    pos = random.randint(0, len(word) - 1)
    op = random.choice(['insert', 'delete', 'substitute'])
    if op == 'insert':
        return word[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[pos:]
    elif op == 'delete':
        return word[:pos] + word[pos+1:]
    else: # substitute
        return word[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[pos+1:]

def p_keyboard_proximity_error(word):
    chars = list(word)
    eligible_indices = [i for i, char in enumerate(chars) if char.lower() in KEYBOARD_NEIGHBORS]
    if not eligible_indices: return word
    idx = random.choice(eligible_indices)
    char_to_replace = chars[idx].lower()
    neighbor = random.choice(KEYBOARD_NEIGHBORS[char_to_replace])
    chars[idx] = neighbor.upper() if chars[idx].isupper() else neighbor
    return "".join(chars)

def p_ocr_error(word):
    keys = list(OCR_ERRORS.keys())
    random.shuffle(keys)
    for target in keys:
        if target in word:
            return word.replace(target, OCR_ERRORS[target], 1)
    return word

def p_permutation_error(word):
    if len(word) < 2: return word
    pos = random.randint(0, len(word) - 2)
    return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]

def p_change_capitalization(word):
    op = random.choice(['lower', 'upper', 'title'])
    if op == 'lower': return word.lower()
    if op == 'upper': return word.upper()
    return word.title()

def p_add_homoglyph(word):
    chars = list(word)
    eligible_indices = [i for i, char in enumerate(chars) if char in HOMOGLYPHS]
    if not eligible_indices: return word
    idx = random.choice(eligible_indices)
    chars[idx] = HOMOGLYPHS[chars[idx]]
    return "".join(chars)

def p_add_zero_width_char(word):
    if len(word) < 2: return word
    pos = random.randint(1, len(word) - 1)
    return word[:pos] + u'\u200b' + word[pos:]

def p_repeat_letters(word):
    if len(word) < 3: return word
    vowel_indices = [i for i, char in enumerate(word) if char.lower() in 'aeiou']
    if not vowel_indices: return word
    pos = random.choice(vowel_indices)
    return word[:pos+1] + (word[pos] * random.randint(1, 2)) + word[pos+1:]

# --- EXHAUSTIVE Perturbation Functions (return list of strings) ---

def p_title_variation_exhaustive(word):
    word_lower = word.lower()
    # Also clean the word by removing a trailing period for matching
    clean_word_no_period = word_lower.rstrip('.')
    is_capitalized = word[0].isupper()
    
    original_form_found = None

    for variation_set in TITLE_VARIATIONS:
        # Check for a match with both the original (lowercased) word and the period-cleaned version
        if word_lower in variation_set:
            original_form_found = word_lower
        elif clean_word_no_period in variation_set:
            original_form_found = clean_word_no_period
        
        if original_form_found:
            # Found the set, return all other members
            results = variation_set - {original_form_found}
            if is_capitalized:
                # Return with first letter capitalized, handling cases like "Dott.ssa"
                return sorted([v[0].upper() + v[1:] for v in results])
            else:
                return sorted(list(results))
    return []

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
                    results.add(word[:i] + new_char + word[i+1:])
    return sorted(list(results))

def p_number_format_exhaustive(word):
    if not re.fullmatch(r'\d{1,3}([,.]\d{3})*', word): return []
    options = set()
    if ',' in word: options.add(word.replace(',', '.'))
    if '.' in word: options.add(word.replace('.', ','))
    clean_num_str = re.sub(r'[,.]', '', word)
    try:
        num = int(clean_num_str)
        if num % 1000 == 0 and num > 0:
            if num % 1000000 == 0: options.add(f"{num // 1000000}M")
            else:
                k_val = f"{num // 1000}K"
                options.add(k_val)
                options.add(k_val.lower())
    except (ValueError, TypeError): pass
    return sorted(list(options))

def p_currency_format_exhaustive(word):
    word_lower = word.lower().strip(".,;:?!'\"")
    currency_map = {'€': 'euro', 'euro': '€', 'eur': '€', '$': 'dollaro', 'dollaro': '$', 'usd': '$'}
    if word_lower in currency_map:
        return [currency_map[word_lower]]
    return []

def p_date_format_exhaustive(word):
    if re.fullmatch(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', word):
        current_sep = '/' if '/' in word else '-'
        new_seps = ['.', '/', '-']
        new_seps.remove(current_sep)
        return [word.replace(current_sep, sep) for sep in new_seps]
    return []

def p_split_contraction_exhaustive(word):
    res = ITALIAN_CONTRACTIONS_SPLIT.get(word.lower())
    return [res] if res else []

def p_split_clitic_exhaustive(word):
    results = []
    sorted_clitics = sorted(ITALIAN_CLITICS_SPLIT.keys(), key=len, reverse=True)
    for clitic in sorted_clitics:
        if word.lower().endswith(clitic):
            base = word[:-len(clitic)]
            results.append(base + ITALIAN_CLITICS_SPLIT[clitic])
    return results

# --- Context-Aware Perturbation Functions ---

def p_context_euphonic_d(words, index):
    word = words[index].lower()
    next_word = words[index+1].lower() if index + 1 < len(words) else ''
    if not next_word: return []
    
    results = []
    if word in ['e', 'a'] and next_word[0] not in 'aeiou':
        p_words = words[:]
        p_words[index] = word + 'd'
        results.append(" ".join(p_words))
    if word in ['ed', 'ad'] and next_word[0] == word[-2]:
        p_words = words[:]
        p_words[index] = word[:-1]
        results.append(" ".join(p_words))
    return results

def p_context_merge_contraction(words, index):
    if index + 1 < len(words):
        combo = f"{words[index].lower()} {words[index+1].lower()}"
        merge_map = {v: k for k, v in ITALIAN_CONTRACTIONS_SPLIT.items()}
        if combo in merge_map:
            new_words = words[:index] + [merge_map[combo]] + words[index+2:]
            return [" ".join(new_words)]
    return []

def p_context_concatenate(words, index):
    if index >= len(words) - 1: return []
    new_words = words[:index] + [words[index] + words[index+1]] + words[index+2:]
    return [" ".join(new_words)]

def p_context_add_space(words, index):
    if index >= len(words) - 1: return []
    prefix = " ".join(words[:index+1])
    suffix = " ".join(words[index+1:])
    return [prefix + "  " + suffix, prefix + "   " + suffix]

# --- Main Logic ---

RANDOM_WORD_PERTURBATIONS = {
    'typo': p_typographical_error, 'keyboard': p_keyboard_proximity_error,
    'ocr': p_ocr_error, 'permutation': p_permutation_error,
    'capitalization': p_change_capitalization, 'homoglyph': p_add_homoglyph,
    'zerowidth': p_add_zero_width_char, 'repeat': p_repeat_letters,
}

EXHAUSTIVE_WORD_PERTURBATIONS = {
    'title_variation': p_title_variation_exhaustive,
    'accent_variation': p_accent_variation_exhaustive,
    'number': p_number_format_exhaustive, 'currency': p_currency_format_exhaustive,
    'date': p_date_format_exhaustive,
    'split_contraction': p_split_contraction_exhaustive,
    'split_clitic': p_split_clitic_exhaustive,
}

CONTEXT_AWARE_PERTURBATIONS = {
    'euphonic_d': p_context_euphonic_d, 'merge_contraction': p_context_merge_contraction,
    'concatenate': p_context_concatenate, 'remove_space': p_context_concatenate,
    'add_space': p_context_add_space,
}

ALL_PERTURBATIONS = {**RANDOM_WORD_PERTURBATIONS, **EXHAUSTIVE_WORD_PERTURBATIONS, **CONTEXT_AWARE_PERTURBATIONS}

def generate_perturbations(input_file, output_file, num_versions, specific_types):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            total_generated = 0
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) < 2: continue
                
                original_question = parts[0]
                target_word_str = parts[-1] if len(parts) > 5 else None
                answers = parts[1:-1] if target_word_str else parts[1:]
                words = original_question.split()
                if not words: continue

                indices_to_perturb = [i for i, w in enumerate(words) if target_word_str and w.strip(".,;:?!'\"").lower() == target_word_str.lower()] if target_word_str else list(range(len(words)))
                types_to_apply = specific_types if specific_types else ALL_PERTURBATIONS.keys()

                for pert_type in types_to_apply:
                    if pert_type not in ALL_PERTURBATIONS:
                        print(f"Warning: Perturbation type '{pert_type}' not recognized. Skipping.")
                        continue
                    
                    for index in indices_to_perturb:
                        # --- RANDOM PERTURBATION LOGIC ---
                        if pert_type in RANDOM_WORD_PERTURBATIONS:
                            for _ in range(num_versions):
                                func = RANDOM_WORD_PERTURBATIONS[pert_type]
                                original_word = words[index]
                                perturbed_word = func(original_word)
                                if perturbed_word != original_word:
                                    temp_words = words[:]
                                    temp_words[index] = perturbed_word
                                    output_line = '\t'.join([" ".join(temp_words)] + answers + [pert_type])
                                    outfile.write(output_line + '\n')
                                    total_generated += 1
                        
                        # --- EXHAUSTIVE PERTURBATION LOGIC ---
                        elif pert_type in EXHAUSTIVE_WORD_PERTURBATIONS:
                            func = EXHAUSTIVE_WORD_PERTURBATIONS[pert_type]
                            original_word = words[index]
                            perturbed_words = func(original_word)
                            for p_word in perturbed_words:
                                temp_words = words[:]
                                temp_words[index] = p_word
                                output_line = '\t'.join([" ".join(temp_words)] + answers + [pert_type])
                                outfile.write(output_line + '\n')
                                total_generated += 1

                        # --- CONTEXT-AWARE PERTURBATION LOGIC ---
                        elif pert_type in CONTEXT_AWARE_PERTURBATIONS:
                            func = CONTEXT_AWARE_PERTURBATIONS[pert_type]
                            perturbed_sentences = func(words, index)
                            for p_sentence in perturbed_sentences:
                                output_line = '\t'.join([p_sentence] + answers + [pert_type])
                                outfile.write(output_line + '\n')
                                total_generated += 1
        
        print(f"Successfully generated {total_generated} perturbed examples.")
        print(f"Output saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tokenization perturbations for a TSV dataset on a word-by-word basis.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Input file format:
  - Standard: Question\\tCorrectAnswer\\tOpt1\\tOpt2\\tOpt3
  - With Target Word: Question\\tCorrectAnswer\\tOpt1\\tOpt2\\tOpt3\\tTargetWord
"""
    )
    parser.add_argument("input_file", help="Path to the input TSV file.")
    parser.add_argument("output_file", help="Path for the output TSV file.")
    parser.add_argument("-n", "--num_versions", type=int, default=1, help="Number of versions for RANDOM perturbation types (e.g., typo, ocr). Ignored for exhaustive types (e.g., number, date).")
    parser.add_argument("--types", nargs='+', choices=sorted(ALL_PERTURBATIONS.keys()), help="Generate only specific perturbation types. If not set, all types are generated.")
    args = parser.parse_args()
    generate_perturbations(args.input_file, args.output_file, args.num_versions, args.types)

