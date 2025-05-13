import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# For tokenizer loading
from transformers import AutoTokenizer

from xarch_tokenizers.config import Config
from xarch_tokenizers.experiment_config import load_config
from xarch_tokenizers.logging.logger import setup_logger
from xarch_tokenizers.logging.plot_utils import setup_styles
from xarch_tokenizers.models import load_tokenizer
from xarch_tokenizers.utils.system import VECTOR_HF_MAPPING, get_host, Hosts

# Usage:
#     python tokenizer_comparison.py --tokenizer_names yourmodule.BertTokenizer yourmodule.GPT2Tokenizer
# python xarch_tokenizers/scripts/compare_tokenizers.py --tokenizer_names meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct Qwen/Qwen2.5-3B-Instruct CohereLabs/aya-expanse-8b --experiment_name=llama-qwen-aya

## TODO: add long text
SAMPLE_TEXTS = {
    "english": "The quick brown fox jumps over the lazy dog. It's 1234.56 and costs $789.",
    "french": "Le renard brun rapide saute par-dessus le chien paresseux. C'est 1234,56 et coûte 789€.",
    "german": "Der schnelle braune Fuchs springt über den faulen Hund. Es ist 1234,56 und kostet 789€.",
    "turkish": "Hızlı kahverengi tilki tembel köpeğin üstunden atlar. 1234.56'dır ve 789$ tutar.",
    "chinese": "快速的棕色狐狸跳过懒狗。它是1234.56，价格为789美元。",
    "arabic": "الثعلب البني السريع يقفز فوق الكلب الكسول. إنه 1234.56 ويكلف 789 دولارًا.",
    "hindi": "तेज भूरी लोमड़ी आलसी कुत्ते पर कूदती है। यह 1234.56 है और 789 डॉलर की कीमत है।",
    "code": "def calculate_sum(a, b):\n    return a + b\n\nresult = calculate_sum(123, 456)",
    "mixed": "English text with numbers 12345 and special chars !@#$%, plus some code: x = f(y)",
    "numbers": "1 12 123 1234 12345 123456 1234567 12345678 123456789",
    "emoji": "😀 👍 🚀 🌍 🎉 💡 🔥 🎵 🏆 🌈",
}


@dataclass
class TokenizerComparisonConfig(Config):
    tokenizer_names: List[str] = field(
        default_factory=list,
        metadata={"help": "Pass the names of the tokenizers to analyze.", "nargs": "+"},
    )


def setup_tokenizer_environment(config: TokenizerComparisonConfig):
    """Set up environment, models, and tokenizers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(config, "tokenizer_diff")
    logger.info("Starting tokenizer analysis with config:")
    config.display()

    tokenizers = dict()
    models = dict()
    # Load tokenizer and model
    for tokenizer_name in config.tokenizer_names:
        if get_host() == Hosts.vector:
            _tokenizer_name = VECTOR_HF_MAPPING.get(tokenizer_name, tokenizer_name)
        else:
            _tokenizer_name = tokenizer_name
        tokenizer = load_tokenizer(_tokenizer_name)
        tokenizer_name = tokenizer_name.split("/")[-1]
        tokenizers[tokenizer_name] = tokenizer
        # _model_name = VECTOR_HF_MAPPING.get(config.model_name, config.model_name)
        # logger.info("Loading model from %s", _model_name)
        model = None
        models[tokenizer_name] = model
    # TODO: maybe return models
    return models, tokenizers, logger, device


def get_tokenizer_info(tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Extract basic information about the tokenizer"""
    info = {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "all_special_tokens": tokenizer.all_special_tokens,
        "special_tokens_map": tokenizer.special_tokens_map,
        # Try to determine the tokenizer type
        "is_fast": getattr(tokenizer, "is_fast", False),
        "is_bpe": hasattr(tokenizer, "bpe_ranks") or "BPE" in type(tokenizer).__name__,
        "is_wordpiece": "WordPiece" in type(tokenizer).__name__,
        "is_unigram": "Unigram" in type(tokenizer).__name__
        or "SentencePiece" in type(tokenizer).__name__,
        "has_merges": hasattr(tokenizer, "merges"),
    }
    return info


def analyze_number_tokenization(tokens: List[str], original_text: str) -> str:
    """Analyze how numbers are tokenized"""
    if len(tokens) == 1:
        return "whole_number"
    elif len(tokens) == len(original_text):
        return "digit_by_digit"
    elif len(tokens) < len(original_text):
        # Check for 3-digit grouping (common in some tokenizers)
        lengths = [len(t.replace("##", "").strip()) for t in tokens]
        if all(l <= 3 for l in lengths):
            # TODO: left to right, right-to-left
            return "3_digit"
        else:
            return "mixed_grouping"
    else:
        return "complex"


def analyze_vocabulary_overlap(
    tokenizers: Dict[str, AutoTokenizer], logger
) -> Dict[str, Any]:
    """
    Analyze vocabulary overlap between tokenizers:
    - Shared vocabulary tokens
    - Unique tokens per tokenizer
    - Overlap percentages
    """
    logger.info("Analyzing vocabulary overlap...")
    vocabularies = {}

    # Extract vocabulary from each tokenizer
    for name, tokenizer in tokenizers.items():
        if hasattr(tokenizer, "get_vocab"):
            vocabularies[name] = set(tokenizer.get_vocab().keys())
        elif hasattr(tokenizer, "vocab"):
            vocabularies[name] = set(tokenizer.vocab.keys())
        else:
            vocabularies[name] = set()
            logger.info(f"Warning: Could not extract vocabulary for {name}")

    # Compute pairwise overlaps
    overlaps = {}
    for name1 in vocabularies:
        for name2 in vocabularies:
            if name1 != name2:
                key = f"{name1}_vs_{name2}"
                shared = vocabularies[name1].intersection(vocabularies[name2])
                overlaps[key] = {
                    "shared_tokens": len(shared),
                    "overlap_percentage_of_first": len(shared)
                    / len(vocabularies[name1])
                    * 100
                    if vocabularies[name1]
                    else 0,
                    "overlap_percentage_of_second": len(shared)
                    / len(vocabularies[name2])
                    * 100
                    if vocabularies[name2]
                    else 0,
                }

    # Calculate unique tokens per tokenizer
    unique_tokens = {}
    for name, vocab in vocabularies.items():
        other_vocabs = set().union(*[v for n, v in vocabularies.items() if n != name])
        unique = vocab - other_vocabs
        unique_tokens[name] = {
            "unique_tokens": len(unique),
            "unique_percentage": len(unique) / len(vocab) * 100 if vocab else 0,
        }

    return {
        "vocab_sizes": {name: len(vocab) for name, vocab in vocabularies.items()},
        "pairwise_overlaps": overlaps,
        "unique_tokens": unique_tokens,
    }


def analyze_categorical_differences(
    tokenizers: Dict[str, AutoTokenizer], tokenizer_info: Dict[str, Any], logger
) -> pd.DataFrame:
    """
    Analyze categorical differences between tokenizers:
    - Word/subword/character based
    - Algorithm type (BPE, WordPiece, Unigram)
    - Special token handling for numbers, punctuation, etc.
    """
    logger.info("Analyzing categorical differences...")
    results = []

    for name, tokenizer in tokenizers.items():
        # Determine tokenizer type
        if tokenizer_info[name]["is_bpe"]:
            algorithm_type = "BPE"
        elif tokenizer_info[name]["is_wordpiece"]:
            algorithm_type = "WordPiece"
        elif tokenizer_info[name]["is_unigram"]:
            algorithm_type = "Unigram/SentencePiece"
        else:
            algorithm_type = "Unknown"

        # Determine token granularity
        # This is a heuristic approach - could be refined
        word_based = False
        subword_based = True  # Most modern tokenizers are subword-based
        char_based = False

        # Check how numbers are tokenized
        number_text = "12345"
        number_tokens = tokenizer.tokenize(number_text)
        number_handling = analyze_number_tokenization(number_tokens, number_text)

        results.append(
            {
                "tokenizer": name,
                "algorithm": algorithm_type,
                "word_based": word_based,
                "subword_based": subword_based,
                "character_based": char_based,
                "number_handling": number_handling,
                "vocab_size": tokenizer_info[name]["vocab_size"],
                "special_tokens": len(tokenizer_info[name]["all_special_tokens"]),
            }
        )

    return pd.DataFrame(results)


def analyze_language_differences(
    tokenizers: Dict[str, AutoTokenizer], logger
) -> pd.DataFrame:
    """
    Analyze how tokenizers perform across different languages:
    - Tokens per character ratio
    - Token count comparison
    - Special handling detection
    """
    logger.info("Analyzing language differences...")
    results = []

    for lang, text in SAMPLE_TEXTS.items():
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(text)

            results.append(
                {
                    "tokenizer": name,
                    "language": lang,
                    "text_length": len(text),
                    "token_count": len(tokens),
                    "tokens_per_char": len(tokens) / len(text) if text else 0,
                    "avg_token_length": np.mean(
                        [len(t.replace("##", "")) for t in tokens]
                    )
                    if tokens
                    else 0,
                    "tokens": tokens,
                }
            )

    return pd.DataFrame(results)


def analyze_token_length_distribution(
    tokenizers: Dict[str, AutoTokenizer], logger
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze token length distribution for each tokenizer:
    - Calculate statistics on token lengths
    - Create histograms of token lengths
    """
    logger.info("Analyzing token length distributions...")
    distributions = {}

    # Combine all sample texts
    combined_text = " ".join(SAMPLE_TEXTS.values())

    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(combined_text)
        # Clean token lengths (remove special chars like ##, Ġ, etc.)
        token_lengths = [len(clean_token(t)) for t in tokens]

        distributions[name] = {
            "mean_length": np.mean(token_lengths),
            "median_length": np.median(token_lengths),
            "std_dev": np.std(token_lengths),
            "min_length": min(token_lengths),
            "max_length": max(token_lengths),
            "length_counts": Counter(token_lengths),
        }

    return distributions


def clean_token(token: str) -> str:
    """Clean a token by removing tokenizer-specific prefixes/suffixes"""
    # Remove common prefixes/suffixes that don't represent the actual token content
    token = token.replace("##", "").replace("Ġ", "").replace("▁", "")
    # Add more cleaning rules as needed
    return token


def analyze_merge_rules(tokenizers: Dict[str, AutoTokenizer], logger) -> Dict[str, Any]:
    """
    Analyze merge rules for BPE-based tokenizers:
    - Extract merge priorities
    - Compare merge strategies
    TODO: check the problem
    """
    logger.info("Analyzing merge rules...")
    merge_analysis = {}

    for name, tokenizer in tokenizers.items():
        if hasattr(tokenizer, "merges") and tokenizer.merges:
            # BPE tokenizers typically have a merges list
            merges = tokenizer.merges

            # Analyze first 1000 merges for efficiency
            sample_merges = merges[:1000] if len(merges) > 1000 else merges

            # Extract components from sample merges
            components = []
            for m in sample_merges:
                if isinstance(m, str):
                    parts = m.split()
                    if len(parts) == 2:
                        components.extend(parts)

            merge_analysis[name] = {
                "total_merges": len(merges),
                "component_frequency": Counter(components).most_common(20),
                "sample_merges": sample_merges[:20],  # First 20 merges for illustration
            }
        else:
            merge_analysis[name] = {
                "total_merges": 0,
                "component_frequency": [],
                "sample_merges": [],
                "note": "No merge rules found or not a BPE-based tokenizer",
            }

    return merge_analysis


def plot_categorical_diff(df: pd.DataFrame, output_dir: str):
    """Plot categorical differences"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    cols = ["algorithm", "number_handling", "vocab_size"]
    for i, col in enumerate(cols):
        # plt.subplot(2, 2, i + 1)
        ax = axs[i]
        if col == "vocab_size":
            sns.barplot(x="tokenizer", y=col, data=df, ax=ax)
            ax.set_yscale("log")
            ax.set_title(f"Vocab. Size")
        else:
            counts = df.groupby(["tokenizer", col]).size().unstack()
            counts.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(output_dir / "categorical_differences.png", bbox_inches="tight")


def plot_vocab_overlap(overlap_data: Dict[str, Any], output_dir: str):
    """Plot vocabulary overlap information"""
    # Create overlap matrix
    tokenizers = list(overlap_data["vocab_sizes"].keys())
    n = len(tokenizers)
    overlap_matrix = np.zeros((n, n))

    for i, t1 in enumerate(tokenizers):
        for j, t2 in enumerate(tokenizers):
            if i == j:
                overlap_matrix[i, j] = 100  # Self-overlap is 100%
            else:
                key = f"{t1}_vs_{t2}"
                if key in overlap_data["pairwise_overlaps"]:
                    overlap_matrix[i, j] = overlap_data["pairwise_overlaps"][key][
                        "overlap_percentage_of_first"
                    ]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt=".1f",
        xticklabels=tokenizers,
        yticklabels=tokenizers,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
    )
    plt.title("Vocabulary Overlap Percentage")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vocabulary_overlap.png"))

    # Plot unique tokens
    plt.figure(figsize=(10, 6))
    unique_data = {
        t: d["unique_percentage"] for t, d in overlap_data["unique_tokens"].items()
    }
    plt.bar(unique_data.keys(), unique_data.values())
    plt.title("Percentage of Unique Tokens per Tokenizer")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unique_tokens.png"))


def plot_language_diff(df: pd.DataFrame, output_dir: str):
    """Plot language-level differences"""
    plt.figure(figsize=(14, 10))

    # Plot tokens per character across languages
    plt.subplot(2, 1, 1)
    pivot_df = df.pivot(index="language", columns="tokenizer", values="tokens_per_char")
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Tokens per Character Across Languages")

    # Plot token count comparison
    plt.subplot(2, 1, 2)
    sns.barplot(x="language", y="token_count", hue="tokenizer", data=df)
    plt.title("Token Count by Language and Tokenizer")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "language_differences.png"))


def plot_token_length_dist(distributions: Dict[str, Dict[str, Any]], output_dir: str):
    """Plot token length distributions"""
    plt.figure(figsize=(12, 8))

    # Plot histograms of token lengths
    max_length = max([dist["max_length"] for dist in distributions.values()])

    for i, (name, dist) in enumerate(distributions.items()):
        plt.subplot(len(distributions), 1, i + 1)

        # Get counts for each length
        lengths = list(range(1, max_length + 1))
        counts = [dist["length_counts"].get(l, 0) for l in lengths]

        plt.bar(lengths, counts)
        plt.title(f"{name} - Token Length Distribution")
        plt.axvline(
            x=dist["mean_length"],
            color="r",
            linestyle="--",
            label=f"Mean: {dist['mean_length']:.2f}",
        )
        plt.axvline(
            x=dist["median_length"],
            color="g",
            linestyle="--",
            label=f"Median: {dist['median_length']:.2f}",
        )
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_length_distribution.png"))


def save_all_data(
    cat_diff, vocab_overlap, lang_diff, token_lengths, merge_rules, output_dir
):
    """Save all raw data for further analysis"""
    # Save categorical differences
    cat_diff.to_csv(
        os.path.join(output_dir, "categorical_differences.csv"), index=False
    )

    # Save vocabulary overlap data
    with open(os.path.join(output_dir, "vocabulary_overlap.json"), "w") as f:
        # Convert sets to lists for JSON serialization
        serializable_overlap = vocab_overlap.copy()
        # Just save the metrics, not the actual tokens
        json.dump(serializable_overlap, f, indent=2)

    # Save language differences
    # Remove the actual tokens to keep the file size manageable
    lang_diff_save = lang_diff.copy()
    lang_diff_save = lang_diff_save.drop(columns=["tokens"])
    lang_diff_save.to_csv(
        os.path.join(output_dir, "language_differences.csv"), index=False
    )

    # Save token length distributions
    with open(os.path.join(output_dir, "token_length_distributions.json"), "w") as f:
        # Convert Counter objects to dictionaries for JSON serialization
        serializable_dists = {}
        for name, dist in token_lengths.items():
            serializable_dist = dist.copy()
            serializable_dist["length_counts"] = {
                str(k): v for k, v in dist["length_counts"].items()
            }
            serializable_dists[name] = serializable_dist
        json.dump(serializable_dists, f, indent=2)

    # Save merge rules
    with open(os.path.join(output_dir, "merge_rules.json"), "w") as f:
        # Convert sets to lists for JSON serialization
        serializable_overlap = merge_rules.copy()
        # Just save the metrics, not the actual tokens
        json.dump(serializable_overlap, f, indent=2)


def generate_report(output_dir: str = "./tokenizer_analysis"):
    """Generate a comprehensive report of the analysis"""
    # Implement report generation logic here
    pass


def analyze_all(output_dir: str = "./tokenizer_analysis"):
    """Run all analyses and generate reports and visualizations"""
    output_dir.mkdir(parents=True, exist_ok=True)


def run_analysis(config, models, tokenizers, logger):
    ## 1. static analysis
    tokenizer_info = dict()
    for tokenizer_name, tokenizer in tokenizers.items():
        tokenizer_info[tokenizer_name] = get_tokenizer_info(tokenizer)
    # Run all analyses
    cat_diff = analyze_categorical_differences(tokenizers, tokenizer_info, logger)
    vocab_overlap = analyze_vocabulary_overlap(tokenizers, logger)
    lang_diff = analyze_language_differences(tokenizers, logger)
    token_lengths = analyze_token_length_distribution(tokenizers, logger)
    merge_rules = analyze_merge_rules(tokenizers, logger)

    logger.info("Generating visualizations...")
    # Categorical differences visualization
    plot_categorical_diff(cat_diff, config.experiment_dir)
    # Vocabulary overlap visualization
    plot_vocab_overlap(vocab_overlap, config.experiment_dir)
    # Language differences visualization
    plot_language_diff(lang_diff, config.experiment_dir)
    # Token length distribution visualization
    plot_token_length_dist(token_lengths, config.experiment_dir)

    # Save all raw data
    save_all_data(
        cat_diff,
        vocab_overlap,
        lang_diff,
        token_lengths,
        merge_rules,
        config.experiment_dir,
    )
    # Generate report
    generate_report(config.experiment_dir)

    logger.info(f"Analysis complete. Results saved to {config.experiment_dir}")

    # Return summary of results
    return {
        "categorical": cat_diff,
        "vocabulary": vocab_overlap,
        "language": lang_diff,
        "token_lengths": token_lengths,
        "merge_rules": merge_rules,
    }


def main():
    """Entry point for translate command."""
    setup_styles()
    config = load_config(TokenizerComparisonConfig)
    models, tokenizers, logger, device = setup_tokenizer_environment(config)
    run_analysis(config, models, tokenizers, logger)


if __name__ == "__main__":
    main()
