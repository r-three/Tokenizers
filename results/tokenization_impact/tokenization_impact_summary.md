# Tokenization Impact Analysis

## Token Count Analysis

Correlation between token ratio and performance: 0.008

### Tokenization Types with Highest Token Counts (Relative to Baseline)

- arabic_with_diacritics: 1.64x
- japanese_word_split: 1.19x
- english_with_russian_case: 1.11x
- subscript_notation: 1.10x
- pinyin_with_tones: 1.10x

## Embedding Similarity Analysis

Correlation between embedding similarity and performance difference: -0.208

### Tokenization Pairs with Lowest Embedding Similarities

- telugu_standard vs english_translation: 0.226
- telugu_partial_split vs english_translation: 0.229
- bengali_standard vs english_translation: 0.253
- bengali_split vs english_translation: 0.255
- telugu_standard vs telugu_transliterated: 0.273

## Information Theory Analysis

Correlation between information density and performance: -0.021
Correlation between entropy and performance: 0.034

### Tokenization Types with Highest Information Density

- standard_capitalization: 6.52 chars/token
- alternative_capitalization: 6.38 chars/token
- split_words: 6.30 chars/token
- single_word: 6.10 chars/token
- two_words: 5.93 chars/token

## Key Findings

The analysis suggests that tokenization differences have minimal impact on model performance. The observed performance variations are likely due to other factors such as model architecture or training data rather than tokenization.
