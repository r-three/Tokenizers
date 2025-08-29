import hashlib
import json
import multiprocessing as mp
import unicodedata
from curses import meta
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tkinter import ALL
from typing import Any, Dict, List, Literal, Optional, Set

import numpy as np
import pandas as pd
from click import Option
from datasets import Dataset, load_dataset
from regex import D

from xarch_tokenizers.config import Config
from xarch_tokenizers.experiment_config import load_config
from xarch_tokenizers.logging.logger import setup_logger
from xarch_tokenizers.perturbations import PERTURBATION_MAPPINGS
from xarch_tokenizers.scripts.convert_dataset_to_hf_format import (
    cleanup_excel,
    read_data,
)


@dataclass
class PerturbationArgs(Config):
    dataset_path: Path = field(
        default=Path("input.jsonl"), metadata={"help": "Path to the input dataset."}
    )
    output_dir: Path = field(
        default=Path("output"), metadata={"help": "Output directory."}
    )
    output_format: Optional[Literal["jsonl", "xlsx"]] = field(
        default=None,
        metadata={"help": "Output format, leave empty to match input format.."},
    )

    perturbations: List[
        Literal[
            "cultural_references",
            "domain_specific_punctuation",
            "quotes_and_parentheses",
            "rare_n_grams",
            "sentence_boundaries",
            "wiki_jargon",
            "character_substitution",
            "deliberate_misspellings",
            "emoji_substitution",
            "word_reordering",
            "colloquial",
            "phonetic_spelling",
            "letter_repetition_for_emphasis",
            "word_concatenation",
            "borrowing",
            "calques",
            "transliteration",
            "chemical_formulas",
            "equations_with_mixed_notation",
            "numerical_formats",
            "scientific_notation",
            "unit_combinations",
            "clitics",
            "compounds",
            "contractions",
            "affixation_edge_cases",
            "derivations",
            "inflections",
            "academic_citations",
            "place_names_with_apostrophes",
            "special_names_across_cultures",
            "technical_product_names",
            "keyboard_proximity_errors",
            "noise_injection",
            "ocr_errors",
            "permutations",
            "typographical_errors",
            "abbreviations_with_periods",
            "brand_names_with_punctuation",
            "code_language_script_switching",
            "diacritics_presence_absence",
            "equivalent_expressions",
            "historical_spelling",
            "homoglyphs",
            "proper_nouns_with_unusual_capitalization",
            "regional_spelling_variations",
            "word_spacing_zero_width_characters_extra_space",
            "romanization",
            "capitalization",
            "email_addresses",
            "headers_and_section_titles",
            "list_markers",
            "text_decorations",
            "unicode_formatting",
            "unusual_formatting",
            "urls_and_file_paths",
        ]
    ] = field(default_factory=lambda: ["keyboard_proximity_errors"])

    sheet_name: str = field(
        default="English-Text-Completion",
        metadata={
            "help": "If the provided dataset_path is an excel file, pass the relevant sheet name."
        },
    )
    question_field: str = field(
        default="question",
        metadata={"help": "Name of the field containing the question."},
    )
    target_field: str = field(
        default="target",
        metadata={"help": "Name of the field containing the target."},
    )
    option_fields: List[str] = field(
        default_factory=lambda: ["A", "B", "C"],
        metadata={"help": "Names of the field containing the options."},
    )
    set_id_field: Optional[str] = field(
        default="Set Id", metadata={"help": "Name of the field containing the set id."}
    )
    variation_id_field: Optional[str] = field(
        default="Variation Id",
        metadata={"help": "Name of the field containing the variation id."},
    )
    perturbation_subcategory_field: Optional[str] = field(
        default="Subcategory",
        metadata={
            "help": "Name of the field containing the perturbation (subcategory)."
        },
    )
    perturbation_category_field: Optional[str] = field(
        default="Category",
        metadata={
            "help": "Name of the field containing the perturbation's general category."
        },
    )

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path).resolve().absolute()
        self.output_dir = Path(self.output_dir).resolve().absolute()
        return super().__post_init__()


def apply_perturbation(df, perturbation):
    if perturbation.func is not None:
        df = perturbation.func(df)
    return df


def read_data(config, logger):
    logger.info(f"Loading data from {config.dataset_path}")
    if config.dataset_path.suffix == ".xlsx":
        df = pd.read_excel(
            config.dataset_path,
            sheet_name=config.sheet_name,
            na_values=["", "null", "NULL"],
        )
        df = cleanup_excel(df, config)
    else:
        df = pd.read_csv(
            config.dataset_path,
            sep=config.separator,
            engine="python",
            na_values=["", "null", "NULL"],
        )

    logger.info(f"Loaded {len(df)} rows")

    # Create output directory
    output_dir_ = Path(config.output_dir)
    output_dir_.mkdir(parents=True, exist_ok=True)
    return df


def perturb(config, logger):
    df = read_data(config, logger)
    logger.info("Data loaded successfully.")
    # Apply perturbations
    # get perturbation
    resulting_df = df.copy()
    cannonical_ids = (
        df.groupby(config.set_id_field)[config.variation_id_field].min().reset_index()
    )
    cannonical_df = df[
        df.apply(
            lambda x: x[config.variation_id_field]
            == cannonical_ids.loc[
                cannonical_ids[config.set_id_field] == x[config.set_id_field],
                config.variation_id_field,
            ].values[0],
            axis=1,
        )
    ]
    for perturbation_name in config.perturbations:
        if perturbation_name not in PERTURBATION_MAPPINGS:
            logger.warning(f"Unknown perturbation: {perturbation_name}")
            continue
        logger.info(f"Applying perturbation: {perturbation_name}")
        # Apply the specific perturbation to the dataframe
        perturbation = PERTURBATION_MAPPINGS[perturbation_name]
        if perturbation.func is None:
            logger.warning(
                f"Skipping non-automatable perturbation: {perturbation_name}"
            )
            continue
        from code import interact

        # interact(local=locals() | globals())
        # df = apply_perturbation(df, perturbation)
        perturbed_df = cannonical_df.copy()
        perturbed_df[config.question_field] = cannonical_df[
            config.question_field
        ].apply(perturbation.func)
        perturbed_df[config.perturbation_subcategory_field] = perturbation_name
        perturbed_df[config.perturbation_category_field] = perturbation.category
        resulting_df = pd.concat([resulting_df, perturbed_df], ignore_index=True)
    logger.info("All perturbations applied successfully.")

    if (
        config.output_format is None and config.dataset_path.suffix == ".xlsx"
    ) or config.output_format == "xlsx":
        output_path = config.output_dir / "perturbed_data.xlsx"
        resulting_df.to_excel(output_path, index=False)
    elif (
        config.output_format is None and config.dataset_path.suffix == ".jsonl"
    ) or config.output_format == "jsonl":
        output_path = config.output_dir / "perturbed_data.jsonl"
        resulting_df.to_json(output_path, orient="records", lines=True)
    logger.info(f"Perturbed data saved to {output_path}")
    return df


def main():
    config: PerturbationArgs = load_config(PerturbationArgs)
    logger = setup_logger(config, "perturber")

    # output_dir = config.output_dir.resolve().absolute()
    perturb(config, logger)


if __name__ == "__main__":
    main()
