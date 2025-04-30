import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import viridis
from matplotlib.colors import to_hex

TASK_TO_PLOT_MAPPING = {
    "turkishmmlu_mathematics": "TUR-MMLU-Math",
    "turkishmmlu_geography": "TUR-MMLU-Geo",
    "turkishmmlu_philosophy": "TUR-MMLU-Phil",
    "turkishmmlu_turkish_language_and_literature": "TUR-MMLU-Lit",
    "cmmlu_modern_chinese": "CH-MMLU-ModernCH",
    "hellaswag": "HellaSwag",
    "gsm8k": "GSM8K",
    "translated_turkishmmlu_mathematics": "(Trans.) TUR-MMLU-Math",
    "translated_turkishmmlu_geography": "(Trans.) TUR-MMLU-Geo",
    "translated_turkishmmlu_philosophy": "(Trans.) TUR-MMLU-Phil",
    "translated_turkishmmlu_turkish_language_and_literature": "(Trans.) TUR-MMLU-Lit",
    "translated_cmmlu_modern_chinese": "(Trans.) CH-MMLU-ModernCH",
    "mgsm_bn": "MGSM-Bn",
    "mgsm_en": "MGSM-En",
    "mgsm_es": "MGSM-Es",
    "mgsm_fr": "MGSM-Fr",
    "mgsm_ja": "MGSM-Ja",
    "mgsm_de": "MGSM-De",
    "mgsm_ru": "MGSM-Ru",
    "mgsm_sw": "MGSM-Sw",
    "mgsm_te": "MGSM-Te",
    "mgsm_th": "MGSM-Th",
    "mgsm_zh": "MGSM-Zh",
}


def setup_styles():
    import matplotlib.font_manager as fm

    font_path = Path(
        os.environ.get("HOME"), "Library/Fonts/AtkinsonHyperlegible-Regular.ttf"
    )
    # Find all Atkinson Hyperlegible font files you have
    font_files = [
        font_path.with_stem(f"AtkinsonHyperlegible-{setting}")
        for setting in ["Regular", "Bold", "Italic", "BoldItalic"]
    ]

    # Register each font file
    for font_file in font_files:
        if os.path.exists(font_file):
            fm.fontManager.addfont(font_file)
    # Add the font file to matplotlib's font manager
    custom_font = fm.FontProperties(fname=font_path)

    sns.set_style("whitegrid")
    # Set global font sizes
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [custom_font.get_name()]
            + plt.rcParams["font.sans-serif"],
            "font.size": 14,  # Default font size
            "axes.titlesize": 16,  # Title font size
            "axes.labelsize": 14,  # Axis label font size
            "xtick.labelsize": 14,  # X-axis tick label font size
            "ytick.labelsize": 14,  # Y-axis tick label font size
            "legend.fontsize": 14,  # Legend font size
            "figure.titlesize": 16,  # Figure title font size
        }
    )


# Define base colors for each model family
MODEL_FAMILY_COLORS = {
    "llama": "#5B21FF",  # Brighter purple
    "qwen": "#FF3E30",  # Brighter red
    "gpt": "#4285F4",  # Google Blue
    "mistral": "#00C853",  # Brighter green
    "gemma": "#FFD600",  # Brighter yellow
    "claude": "#FF9100",  # Brighter orange
    "phi": "#00E5FF",  # Brighter cyan
    "aya": "#8D6E63",  # Brighter brown
    "mpt": "#78909C",  # Brighter blue-grey
    "pythia": "#FF6E40",  # Brighter deep orange
}


def get_model_color(model_name, param_size=None):
    """
    Get color for a specific model based on family and parameter size.

    Args:
        model_name (str): Name of the model
        param_size (float, optional): Parameter size in billions. If None, uses base color.

    Returns:
        str: Hex color code
    """
    # Identify model family
    model_name = model_name.lower()
    family = None

    for family_name in MODEL_FAMILY_COLORS:
        if family_name in model_name:
            family = family_name
            break

    if family is None:
        # Default color if family not found
        return "#999999"

    base_color = MODEL_FAMILY_COLORS[family]

    # If no parameter size provided, return base color
    if param_size is None:
        return base_color

    # Adjust darkness based on parameter size
    # Larger models get darker colors
    base_rgb = plt.matplotlib.colors.to_rgb(base_color)

    # Map parameter size to darkness factor (0.5 to 1.2)
    # Small models (1-8B) will be lighter
    # Large models (70B+) will be darker
    if param_size < 5:
        darkness = 0.5 + (param_size / 10)  # Lighter for smaller models
    elif param_size < 20:
        darkness = 0.7 + (param_size / 100)  # Medium for medium models
    else:
        darkness = 0.9 + (min(param_size, 100) / 500)  # Darker for larger models

    # Adjust RGB values (make darker by reducing values)
    adjusted_rgb = tuple(min(1.0, c * darkness) for c in base_rgb)

    return to_hex(adjusted_rgb)


LLAMA_SIZES = [3, 8, 13, 34, 70, 405]
QWEN_SIZES = [0.5, 1.5, 3, 7, 14, 32, 72]

MODEL_TO_COLOR = {
    # Llama models - using viridis colors
    **{
        f"meta-llama/Meta-Llama-3-{size}B-Instruct": get_model_color("llama", size)
        for size in LLAMA_SIZES
    },
    **{
        f"meta-llama/Meta-Llama-3.1-{size}B-Instruct": get_model_color("llama", size)
        for size in LLAMA_SIZES
    },
    # Qwen models - using viridis colors
    **{
        f"Qwen/Qwen1.5-{size}B-Instruct": get_model_color("qwen", size)
        for size in QWEN_SIZES
    },
    **{
        f"Qwen/Qwen2-{size}B-Instruct": get_model_color("qwen", size)
        for size in QWEN_SIZES
    },
    **{
        f"Qwen/Qwen2.5-{size}B-Instruct": get_model_color("qwen", size)
        for size in QWEN_SIZES
    },
    # Other model families can be added here
    "mistralai/Mistral-7B-Instruct-v0.3": get_model_color("mistral", 7),
    "mistralai/Mixtral-8x7B-v0.1": get_model_color("mistral", 56),
    "mistralai/Mixtral-8x7B-v0.1": get_model_color("mistral", 56),
    "CohereLabs/aya-expanse-8B": get_model_color("aya", 8),
    "google/gemma-7b-it": get_model_color("gemma", 7),
}


# Function to get color for any model, even if not in the predefined map
def get_color_for_model(model_name):
    """Get color for any model, extracting family and size if possible"""
    if model_name in MODEL_TO_COLOR:
        return MODEL_TO_COLOR[model_name]

    # Try to extract parameter size from name
    size = None
    name_lower = model_name.lower()

    # Look for patterns like "7b", "13B", "70-b", etc.
    import re

    size_match = re.search(r"[-_/\s](\d+(\.\d+)?)[Bb]", name_lower)
    if size_match:
        try:
            size = float(size_match.group(1))
        except ValueError:
            pass

    return get_model_color(model_name, size)
