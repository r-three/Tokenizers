"""initial explorations around tokenizers"""

import os
from typing import Any, Dict, List, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer


def load_tokenizers() -> Dict[str, AutoTokenizer]:
    """Load all tokenizers"""
    return {
        "mistral": AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1"),
        "qwen": AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True),
        "olmo": AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B"),
        "llama": AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf"),
    }


def safe_decode(token: Union[str, bytes]) -> str:
    """Safely decode token if it's bytes, keep as is if string"""
    if isinstance(token, bytes):
        try:
            return token.decode("utf-8", errors="replace")
        except:
            return repr(token)
    return str(token)


def get_vocab_sets(tokenizers: Dict[str, AutoTokenizer]) -> Dict[str, Set[str]]:
    """Get vocabulary sets for all tokenizers, handling byte tokens"""
    vocab_sets = {}
    for name, tokenizer in tokenizers.items():
        vocab = tokenizer.get_vocab().keys()
        # Convert all tokens to strings, handling bytes
        vocab_sets[name] = {safe_decode(token) for token in vocab}
    return vocab_sets


def get_vocab_sizes(vocab_sets: Dict[str, Set[str]]) -> Dict[str, int]:
    """Get vocabulary sizes for each tokenizer"""
    return {name: len(vocab) for name, vocab in vocab_sets.items()}


def analyze_vocab_overlap(vocab_sets: Dict[str, Set[str]]) -> pd.DataFrame:
    """Analyze vocabulary overlap between tokenizers"""
    results = []
    names = list(vocab_sets.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            set1, set2 = vocab_sets[name1], vocab_sets[name2]

            overlap = set1.intersection(set2)
            unique1 = set1 - set2
            unique2 = set2 - set1

            # Filter out non-printable or very long tokens for display
            def filter_displayable(tokens, limit=10):
                displayable = [t for t in tokens if len(t) < 50 and t.isprintable()]
                return sorted(displayable)[:limit]

            results.append(
                {
                    "Pair": f"{name1}-{name2}",
                    "Overlap": len(overlap),
                    f"Unique to {name1}": len(unique1),
                    f"Unique to {name2}": len(unique2),
                    "Overlap_tokens": filter_displayable(overlap),
                    f"Unique_{name1}_tokens": filter_displayable(unique1),
                    f"Unique_{name2}_tokens": filter_displayable(unique2),
                }
            )

    return pd.DataFrame(results)


def find_common_tokens(
    vocab_sets: Dict[str, Set[str]], n_samples: int = 20
) -> List[str]:
    """Find tokens that are common across all tokenizers"""
    common_tokens = set.intersection(*vocab_sets.values())
    # Filter out non-printable or very long tokens
    displayable = [t for t in common_tokens if len(t) < 50 and t.isprintable()]
    return sorted(displayable)[:n_samples]


def analyze_token_lengths(
    vocab_sets: Dict[str, Set[str]],
    save_plot: bool = False,
    save_path: str = "token_length_distribution.png",
) -> Dict[str, List[int]]:
    """Analyze token length distributions"""
    length_distributions = {}

    for name, vocab in vocab_sets.items():
        # Filter out extremely long tokens that might be binary
        lengths = [len(token) for token in vocab if len(token) < 100]
        length_distributions[name] = lengths

    plt.figure(figsize=(12, 6))
    for name, lengths in length_distributions.items():
        sns.kdeplot(lengths, label=name)

    plt.title("Token Length Distribution")
    plt.xlabel("Token Length")
    plt.ylabel("Density")
    plt.legend()

    if save_plot:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return length_distributions


def compare_tokenization(
    text: str, tokenizers: Dict[str, AutoTokenizer]
) -> pd.DataFrame:
    """Compare how different tokenizers split the same text"""
    results = {}

    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        # Convert tokens to strings if they're bytes
        tokens = [safe_decode(t) for t in tokens]
        results[name] = tokens

    max_tokens = max(len(tokens) for tokens in results.values())
    padded_results = {
        name: tokens + [""] * (max_tokens - len(tokens))
        for name, tokens in results.items()
    }

    return pd.DataFrame(padded_results)


def get_token_statistics(vocab_sets: Dict[str, Set[str]]) -> Dict[str, Dict[str, Any]]:
    """Get detailed statistics about each tokenizer's vocabulary"""
    stats = {}

    for name, vocab in vocab_sets.items():
        # Filter out extremely long tokens
        filtered_vocab = {t for t in vocab if len(t) < 100}
        lengths = [len(token) for token in filtered_vocab]

        # Filter displayable tokens for samples
        displayable = [t for t in vocab if t.isprintable() and len(t) < 50]

        stats[name] = {
            "vocab_size": len(vocab),
            "avg_token_length": np.mean(lengths),
            "median_token_length": np.median(lengths),
            "min_token_length": min(lengths),
            "max_token_length": max(lengths),
            "special_tokens": [
                token
                for token in displayable
                if token.startswith("<") and token.endswith(">")
            ],
            "sample_tokens": sorted(displayable)[:10],
        }

    return stats


def analyze_subword_patterns(
    vocab_sets: Dict[str, Set[str]],
) -> Dict[str, Dict[str, int]]:
    """Analyze common subword patterns in vocabularies"""
    pattern_stats = {}

    for name, vocab in vocab_sets.items():
        # Filter out binary/non-displayable tokens
        filtered_vocab = {t for t in vocab if t.isprintable() and len(t) < 100}

        patterns = {
            "starts_with_##": sum(1 for t in filtered_vocab if t.startswith("##")),
            "starts_with_Ġ": sum(1 for t in filtered_vocab if t.startswith("Ġ")),
            "starts_with_underscore": sum(
                1 for t in filtered_vocab if t.startswith("_")
            ),
            "special_tokens": sum(
                1 for t in filtered_vocab if t.startswith("<") and t.endswith(">")
            ),
            "numbers": sum(1 for t in filtered_vocab if any(c.isdigit() for c in t)),
            "uppercase": sum(1 for t in filtered_vocab if any(c.isupper() for c in t)),
        }
        pattern_stats[name] = patterns

    return pattern_stats


def main():
    """Example usage of tokenizer analysis functions"""
    tokenizers = load_tokenizers()
    vocab_sets = get_vocab_sets(tokenizers)

    # Print vocabulary sizes
    print("\nVocabulary Sizes:")
    for name, size in get_vocab_sizes(vocab_sets).items():
        print(f"{name.capitalize()}: {size}")

    # Analyze overlap
    print("\nVocabulary Overlap Analysis:")
    overlap_df = analyze_vocab_overlap(vocab_sets)
    print(overlap_df[["Pair", "Overlap"]])

    # Find common tokens
    print("\nCommon Tokens Sample:")
    print(find_common_tokens(vocab_sets, 10))

    # Compare tokenization
    sample_text = "The quick brown fox jumps over the lazy dog. It's a common pangram!"
    print("\nTokenization Comparison:")
    print(compare_tokenization(sample_text, tokenizers))

    # Get detailed statistics
    print("\nDetailed Statistics:")
    stats = get_token_statistics(vocab_sets)
    for name, stat in stats.items():
        print(f"\n{name.capitalize()}:")
        print(f"Vocabulary Size: {stat['vocab_size']}")
        print(f"Average Token Length: {stat['avg_token_length']:.2f}")
        print(f"Special Tokens: {len(stat['special_tokens'])}")

    # Analyze subword patterns
    print("\nSubword Pattern Analysis:")
    patterns = analyze_subword_patterns(vocab_sets)
    print(pd.DataFrame(patterns))


if __name__ == "__main__":
    main()
