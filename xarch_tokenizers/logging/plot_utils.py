import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import seaborn as sns
from matplotlib.cm import viridis
from matplotlib.colors import to_hex

NORMAL_FONT = 18
TITLE_FONT = 24
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


def get_new_6_fig(title, sharey=True):
    # Create figure with subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=sharey)
    fig.suptitle(title, y=0.98)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    return fig, axes_flat


def get_new_fig(title, sharey=True, nrows=2, ncols=3):
    # Create figure with subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=sharey
    )
    fig.suptitle(title, y=0.98)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    return fig, axes_flat


def download_font():
    """Download the Atkinson Hyperlegible font if it doesn't exist"""
    # Define the font URL
    font_url = "https://github.com/googlefonts/atkinson-hyperlegible/archive/main.zip"
    # Create fonts directory in user home if it doesn't exist
    fonts_dir = Path(os.environ.get("HOME"), ".fonts")
    fonts_dir.mkdir(exist_ok=True)
    font_path = fonts_dir / "AtkinsonHyperlegible-Regular.ttf"
    # Only download if the font doesn't already exist
    if not font_path.exists():
        import tempfile
        import zipfile

        print(f"Downloading Atkinson Hyperlegible font...")
        # Download the zip file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
            response = requests.get(font_url, stream=True)
            response.raise_for_status()  # Ensure we got a successful response
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name
        # Extract the font file from the zip
        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            # Find the regular TTF file in the zip
            for zip_info in zip_ref.infolist():
                if zip_info.filename.endswith("Regular.ttf"):
                    # Extract with the new name
                    zip_info.filename = os.path.basename(font_path)
                    zip_ref.extract(zip_info, fonts_dir)
                    print(f"Font extracted to {font_path}")
                    break
        # Clean up the temporary file
        os.unlink(temp_path)
        # Update matplotlib font cache
        fm.fontManager.addfont(str(font_path))
    return font_path


def setup_styles_matplotlib(custom_font):
    """Setup global styles for matplotlib"""
    sns.set_style("whitegrid")
    # Set global font sizes
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [custom_font.get_name()]
            + plt.rcParams["font.sans-serif"],
            "font.size": NORMAL_FONT,  # Default font size
            "axes.titlesize": TITLE_FONT,  # Title font size
            "axes.labelsize": NORMAL_FONT,  # Axis label font size
            "xtick.labelsize": NORMAL_FONT,  # X-axis tick label font size
            "ytick.labelsize": NORMAL_FONT,  # Y-axis tick label font size
            "legend.fontsize": NORMAL_FONT,  # Legend font size
            "figure.titlesize": TITLE_FONT,  # Figure title font size
        }
    )


def setup_styles_plotly(custom_font):
    """Setup global styles for plotly"""
    # Set up Plotly template
    template = go.layout.Template()

    # Define custom font settings for Plotly
    font_settings = {
        "family": "Atkinson Hyperlegible, Arial, sans-serif",
        "size": NORMAL_FONT,
        "color": "#333333",
    }

    # Set the global font
    template.layout.font = font_settings

    # Set title font
    template.layout.title = {
        "font": {
            "family": font_settings["family"],
            "size": TITLE_FONT,
            "color": font_settings["color"],
        },
        "x": 0.5,  # Center title
        "xanchor": "center",
    }

    # Set axis styling
    axis_settings = {
        "titlefont": {
            "family": font_settings["family"],
            "size": TITLE_FONT,
            "color": font_settings["color"],
        },
        "tickfont": {
            "family": font_settings["family"],
            "size": NORMAL_FONT,
            "color": font_settings["color"],
        },
        "gridcolor": "#E5E5E5",  # Similar to whitegrid in seaborn
        "linecolor": "#E5E5E5",
    }

    # Apply axis settings to template
    template.layout.xaxis = axis_settings
    template.layout.yaxis = axis_settings

    # Set legend styling
    template.layout.legend = {
        "font": {
            "family": font_settings["family"],
            "size": NORMAL_FONT,
            "color": font_settings["color"],
        },
        "bgcolor": "rgba(255, 255, 255, 0.5)",
        "bordercolor": "#E5E5E5",
        "borderwidth": 1,
    }

    # Set plot background
    template.layout.plot_bgcolor = "white"
    template.layout.paper_bgcolor = "white"

    # Apply margin
    template.layout.margin = {"l": 60, "r": 40, "t": 60, "b": 60}

    # Set the template as default
    pio.templates["custom"] = template
    pio.templates.default = "custom"


def setup_styles():
    font_path = download_font()
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
    setup_styles_matplotlib(custom_font)
    setup_styles_plotly(custom_font)


# Define base colors for each model family
MODEL_FAMILY_COLORS = {
    "mpt": "#78909C",  # Brighter blue-grey
    "llama": "#5B21FF",  # Brighter purple
    "qwen": "#FF3E30",  # Brighter red
    "gpt": "#4285F4",  # Google Blue
    "xglm": "#0081FB",  # Meta Blue
    "mistral": "#00C853",  # Brighter green
    "gemma": "#FFD600",  # Brighter yellow
    "claude": "#FF9100",  # Brighter orange
    "phi": "#00E5FF",  # Brighter cyan
    "aya": "#8D6E63",  # Brighter brown
    "pythia": "#FF6E40",  # Brighter deep orange
    "comma": "#FFDD00",  # orange
    "llama": "#8B5CF6",  # Violet
    "qwen": "#EF4444",  # Red
    "gpt-4o": "#0C4F3A",  # Emerald
    "gpt2": "#4B937CFF",  # Emerald
    "xglm": "#3B82F6",  # Blue
    "mistral": "#F59E0B",  # Amber
    "gemma": "#06B6D4",  # Cyan
    "claude": "#F97316",  # Orange
    "phi": "#37197E",  # Purple
    "aya": "#351E0D",  # Pink
    "pythia": "#7476EC",  # Indigo
    "byt5": "#18AA2B",  # Emerald-600
    "mbert": "#2563EB",  # Blue-600
    "bloom": "#5031DD",  # Violet-600
    "tokenmonster": "#ED6AAC",
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

MODEL_TO_COLOR = {
    "Comma": MODEL_FAMILY_COLORS["comma"],
    "LLaMA": MODEL_FAMILY_COLORS["llama"],
    "Phi-3": MODEL_FAMILY_COLORS["phi"],
    "GPT-2": MODEL_FAMILY_COLORS["gpt2"],
    "GPT-4o": MODEL_FAMILY_COLORS["gpt-4o"],
    "Bloom": MODEL_FAMILY_COLORS["bloom"],
    "XGLM": MODEL_FAMILY_COLORS["xglm"],
    "Tekken": MODEL_FAMILY_COLORS["mistral"],
    "ByT5": MODEL_FAMILY_COLORS["byt5"],
    "MBERT": MODEL_FAMILY_COLORS["mbert"],
    "Qwen-3": MODEL_FAMILY_COLORS["qwen"],
    "TokenMonster": MODEL_FAMILY_COLORS["tokenmonster"],
    "Gemma-2": MODEL_FAMILY_COLORS["gemma"],
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
