#!/usr/bin/env python3
"""
Tokenization Variant Test Converter

This script converts the structured tokenization variant test set into a format
that is directly compatible with lm-evaluation-harness.
"""

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from xarch_tokenizers.utils.system import VECTOR_HF_MAPPING

# Usage:
# python xarch_tokenizers/scripts/convert_dataset_to_hf_format.py --dataset_path data/custom_dataset.json --output_path data/custom_dataset_hf.json


@dataclass
class DatasetConverterConfig(Config):
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset .json", "required": True},
    )
    output_path: Optional[str] = field(
        default=None, metadata={"help": "Out path for writing the pivoted datasets"}
    )
    _create_experiment_dir: bool = False

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        if self.output_path is None:
            self.output_path = self.dataset_path.with_name(
                f"{self.dataset_path.name}_converted"
            )
        self.output_path = Path(self.output_path)
        super().__post_init__()


def transform_to_lm_eval_format(input_json_path, output_json_path):
    """
    Transform the structured tokenization test set into lm-eval-harness format.

    Args:
        input_json_path: Path to the original structured test set
        output_json_path: Path where the transformed data will be saved
    """
    # Load the structured test set
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The transformed dataset will have a list of examples
    transformed_data = []

    # Process each test set
    for test_set in data.get("test_sets", []):
        set_id = test_set.get("set_id")
        category = test_set.get("category", "Unknown")
        title = test_set.get("title", "Untitled")

        # Primary target is the first one, with fallbacks for different structures
        if "targets" in test_set and test_set["targets"]:
            target = test_set["targets"][0]
            all_targets = test_set["targets"]
        elif "answer" in test_set:
            target = test_set["answer"]
            all_targets = [target]
            if "acceptable_answers" in test_set:
                all_targets.extend(test_set["acceptable_answers"])
        else:
            print(f"Warning: No target found for set {set_id}, skipping")
            continue

        # Process each variant
        for variant in test_set.get("variants", []):
            variant_id = variant.get("variant_id", "unknown")
            tokenization_type = variant.get("tokenization_type", "unknown")
            question = variant.get("question", "")

            # Create an example in lm-eval format
            example = {
                "id": f"{set_id}_{variant_id}",
                "question": question,
                "answer": target,
                "all_targets": all_targets,
                "tokenization_type": tokenization_type,
                "category": category,
                "title": title,
                "set_id": str(set_id),
            }

            transformed_data.append(example)

    # Save the transformed data
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)

    print(f"Transformed {len(transformed_data)} examples")
    print(f"Saved to {output_json_path}")

    # Return some statistics about the data
    stats = {
        "num_examples": len(transformed_data),
        "categories": list(set(ex["category"] for ex in transformed_data)),
        "tokenization_types": list(
            set(ex["tokenization_type"] for ex in transformed_data)
        ),
        "num_sets": len(set(ex["set_id"] for ex in transformed_data)),
    }

    return transformed_data, stats


def main():
    config = load_config(DatasetConverterConfig)
    transform_to_lm_eval_format(config.dataset_path, config.output_path)


if __name__ == "__main__":
    main()
