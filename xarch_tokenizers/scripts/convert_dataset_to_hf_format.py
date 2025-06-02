#!/usr/bin/env python3
"""
Tokenization Robustness Dataset Test Converter

This script converts the structured tokenization robustness dataset test set into a format
that is directly compatible with lm-evaluation-harness.
It creates parquet files and creates corresponding yaml files in the lm_eval directory
## TODO: upload to hf
"""

import argparse
import json
import os
import random
import traceback
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import lm_eval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
import torch
import yaml
from datasets import Dataset

# For tokenizer loading
from transformers import AutoTokenizer

from xarch_tokenizers.config import Config
from xarch_tokenizers.experiment_config import load_config
from xarch_tokenizers.logging.logger import setup_basic_logger, setup_logger
from xarch_tokenizers.logging.plot_utils import setup_styles
from xarch_tokenizers.models import load_tokenizer
from xarch_tokenizers.utils.system import VECTOR_HF_MAPPING
from xarch_tokenizers.utils.utils import find_package_dir

LM_EVAL_PKG_DIR = Path(find_package_dir("lm_eval"))
# Usage:
# python xarch_tokenizers/scripts/convert_dataset_to_hf_format.py --dataset_path data/custom_dataset.json --output_dir data/custom_dataset_hf.json


@dataclass
class LmEvalTaskArgs:
    # Dataset configuration
    dataset_path: str = field(
        metadata={"help": "the name of the dataset on the HF Hub."}
    )
    dataset_name: str = field(
        metadata={
            "help": "the dataset configuration to use. Leave `null` if your dataset does not require a config to be passed. See https://huggingface.co/docs/datasets/load_hub#configurations for more info."
        }
    )
    dataset_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "any extra keyword arguments that should be passed to the dataset constructor, e.g. `data_dir`."
        },
    )
    data_files: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Path to dataset_files Or with files already split into separate directories: train: .../train/data.arrow "
        },
    )

    # Data splits
    training_split: str = field(
        default="null", metadata={"help": "split name of training set, or `null`"}
    )
    validation_split: str = field(
        default="null", metadata={"help": "split name of val. set, or `null`"}
    )
    test_split: str = field(
        default="null", metadata={"help": "split name of test set, or `null`"}
    )

    # Data processing
    process_docs: Optional[str] = field(
        default=None,
        metadata={
            "help": "function to process documents, e.g. '!function utils.process_docs'"
        },
    )

    # Prompt configuration
    doc_to_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "defines the input string a model will be given, can use Jinja2 templates or function references"
        },
    )
    doc_to_target: Optional[Union[str, int]] = field(
        default=None,
        metadata={"help": "defines the target text, can be a string or integer index"},
    )
    doc_to_choice: Optional[Union[str, List[str]]] = field(
        default=None,
        metadata={
            "help": "list of possible choice strings for multiple choice tasks, e.g. ['No', 'Yesy'], ['A', 'B', 'C', 'D'], dataset_feature e.g. choices"
        },
    )
    target_delimiter: str = field(
        default=" ", metadata={"help": "delimiter between input and target"}
    )

    # Prompt source integration
    # use_prompt: Optional[str] = field(default=None, metadata={"help": "use promptsource template, e.g. 'promptsource:GPT-3 Style' or 'promptsource:*'"})

    # Few-shot configuration
    fewshot_split: str = field(
        default="dev",
        metadata={"help": "split name to draw fewshot examples from, or `null`"},
    )
    fewshot_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "few-shot configuration including sampler and samples"},
    )
    num_fewshot: Optional[int] = field(
        default=None, metadata={"help": "number of few-shot examples to use"}
    )

    # Metrics configuration
    metric_list: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help": """list of metrics to compute with their aggregation functions 
    ```yaml
    metric_list:
    - metric: <name of the metric here>
        aggregation: <name of the aggregation fn here>
        higher_is_better: <true or false>
    - metric: !function script.function
        aggregation: ...
        higher_is_better: ...
    ```"""
        },
    )

    # Task registration
    task: str = field(default=None, metadata={"help": "name of the task (mandatory)"})
    task_alias: Optional[str] = field(
        default=None, metadata={"help": "alternative task name for display purposes"}
    )
    tag: Optional[List[str]] = field(
        default=None, metadata={"help": "list of tags to categorize the task"}
    )

    # Advanced configuration
    class_: Optional[str] = field(
        default=None,
        metadata={
            "help": "custom Python class for task implementation, e.g. '!function task.SQuAD2'"
        },
    )
    include: Optional[str] = field(
        default=None, metadata={"help": "include other YAML configuration files"}
    )
    description: Optional[str] = field(
        default=None, metadata={"help": "description of the task"}
    )

    # Task output type
    output_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "task output type, e.g. 'multiple_choice' or 'generate_until'"
        },
    )

    # Versioning
    metadata: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "metadata including version information"}
    )

    # Group configuration (for group configs)
    group: Optional[str] = field(
        default=None, metadata={"help": "group name for task groupings"}
    )
    group_alias: Optional[str] = field(
        default=None, metadata={"help": "alternative group name for display purposes"}
    )
    aggregate_metric_list: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "metrics to aggregate across subtasks in groups"},
    )

    # You can also set `dataset_path` as a directory path in your local system. This will assume that there is a loading script with the same name as the directory. [See datasets docs](https://huggingface.co/docs/datasets/loading#local-loading-script).
    _must_include: tuple = (
        "task",
        "dataset_path",
        "dataset_kwargs",
        "test_split",
        "doc_to_text",
        "doc_to_target",
        "doc_to_choice",
        "output_type",
        "metric_list",
    )

    def export_dict(self, aggregate: bool = False):
        dct = dict()
        for _field in fields(self):
            if not aggregate and _field.name == "aggregate_metric_list":
                continue
            if not _field.name.startswith("_") and (
                (v := getattr(self, _field.name)) or (_field.name in self._must_include)
            ):
                dct[_field.name] = v
        return dct


@dataclass
class DatasetConverterConfig(Config):
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset .json | or csv", "required": True},
    )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Out path for writing the pivoted datasets"}
    )
    separator: str = field(
        default="|||",
        metadata={
            "help": "If the provided dataset_path is a csv pass the relevant separator."
        },
    )
    version: Literal["v01", "v1"] = field(
        default="v1",
        metadata={
            "help": "v01 for the cannonical form with perturbations, v1 for csv formatted"
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
    metadata_fields: Dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Mapping for metadata, e.g. subcategories: Subcategory will include a subcategory field for each example and place the raw_format['Subcategory'] into that field."
        },
    )
    create_subset_dirs: bool = False
    subset_by: str = None
    lm_eval_task: Optional[LmEvalTaskArgs] = field(
        default=None, metadata={"help": "LM Evaluation task configuration"}
    )
    _create_experiment_dir: bool = False

    upload_to_hf: bool = False

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        if not self.dataset_path.exists():
            raise ValueError(
                f"Provided path ({self.dataset_path.absolute().as_posix()}) doesn't exist."
            )
        if self.output_dir is None:
            self.output_dir = self.dataset_path.with_name(
                f"{self.dataset_path.name}_converted"
            )
        self.output_dir = Path(self.output_dir)
        if self.create_subset_dirs:
            assert self.subset_by, (
                "Provide a valid feature|column name to create subsets for."
            )
        # Convert nested dictionaries to dataclasses
        if isinstance(self.lm_eval_task, dict):
            self.lm_eval_task = LmEvalTaskArgs(**self.lm_eval_task)

        super().__post_init__()


def transform_v01_to_lm_eval_format(input_json_path, output_json_path):
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


def create_task_config(
    config: DatasetConverterConfig,
    logger,
    task_name: str = None,
    config_name: Union[Path, str] = None,
    base_dir: bool = False,
    update_args: Dict[str, Any] = {},
):
    """Create LM Evaluation Harness task configuration."""
    dataset_path = LM_EVAL_PKG_DIR / "tasks" / config.lm_eval_task.dataset_name
    dataset_path.mkdir(exist_ok=True, parents=True)
    if base_dir:
        dataset_path.joinpath("utils.py").touch()
        dataset_path.joinpath("README.md").touch()
    if task_name is None:
        task_name = config.lm_eval_task.dataset_name
    if config_name is None:
        config_name = task_name
    config_path = (dataset_path / config_name).with_suffix(".yaml")

    # Write YAML config
    with open(config_path, "w") as f:
        yaml.dump(
            {"task": task_name} | update_args,
            f,
            default_flow_style=False,
        )

    logger.info(f"Created task config: {config_path}")
    return config_path


def convert_to_lm_eval_format(
    config: DatasetConverterConfig,
    logger,
    df: pd.DataFrame,
    output_dir: Path,
):
    """Convert dataset to LM Evaluation Harness format."""

    samples = []

    for idx, row in df.iterrows():
        # Extract question and choices
        question = str(row.get(config.question_field, "")).strip()
        if not question:
            continue

        choices = []
        choice_labels = ["A", "B", "C", "D"]

        for i, opt_col in enumerate(config.option_fields):
            opt_text = str(row.get(opt_col, "")).strip()
            if opt_text and opt_text != "nan":
                choices.append(opt_text)

        correct_idx = random.randint(0, len(config.option_fields))
        correct_answer = str(row.get("Correct", "")).strip()
        choices.insert(correct_idx, correct_answer)

        # Create sample in LM Eval format
        sample = {
            "question": question,
            "choices": choices,
            "answer": correct_idx,
            "answer_label": choice_labels[correct_idx],
            "metadata": {
                opt_field: opt_val
                for opt_field, opt_val in config.metadata_fields.items()
            },
        }

        samples.append(sample)
    # TODO: save dev, test, train
    data_files = {}
    output_path = output_dir / "test"
    output_dir.mkdir(exist_ok=True, parents=True)
    # output_path.mkdir(exist_ok=True, parents=True)
    # output_path = output_path.with_suffix(".arrow")

    dataset = Dataset.from_list(samples)
    # Save as Arrow format
    output_path = output_path.with_suffix("")  # Remove any extension
    # dataset.save_to_disk(str(output_path))
    # data_files["test"] = [p.absolute().as_posix() for p in output_path.rglob("*.arrow")]

    output_path = output_path.with_suffix(".parquet")  # Remove any extension
    dataset.to_parquet(str(output_path))
    data_files["test"] = [
        p.absolute().as_posix() for p in output_dir.rglob("*.parquet")
    ]

    logger.info(f"Converted {len(samples)} samples to {output_path}")
    return samples, data_files


def transform_v1(config: DatasetConverterConfig, logger):
    logger.info(f"Loading data from {config.dataset_path}")
    df = pd.read_csv(config.dataset_path, sep=config.separator, engine="python")
    logger.info(f"Loaded {len(df)} rows")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_conf_path = create_task_config(
        config,
        logger,
        task_name=config.lm_eval_task.dataset_name,
        # relative_path="",
        config_name=f"{config.lm_eval_task.dataset_name}_base",
        base_dir=True,
        update_args=config.lm_eval_task.export_dict()
        | {
            "dataset_name": None,
            # "group": f"{config.lm_eval_task.dataset_name}",
        },
    )
    if config.create_subset_dirs:
        if config.subset_by not in df.columns:
            raise ValueError(f"subset_by column ({config.subset_by}) not in the data.")
        # handle rows with multiple subsets -> #TODO: maybe add primary and secondary subsets
        df["_subset"] = df[config.subset_by].apply(
            lambda x: x.split(",")[0] if isinstance(x, str) else x
        )
        all_subsets = df["_subset"].unique()
        added_subsets = set()
        for subset in all_subsets:
            logger.info(f"Processing subset: {subset}.")
            subset_df = df[df["_subset"] == subset]
            subset = str(subset)
            subset = (
                subset.replace(" & ", "_")
                .replace(" ", "_")
                .replace("/", "_")
                .replace(".", "_")
                .replace(",", "")
                .lower()
            )
            if subset in [None, "null", "na", "nan", "not_found", "none"]:
                continue
            task_name = f"{config.lm_eval_task.dataset_name}_{subset}"
            data_path = output_dir / subset / task_name
            # todo upload to hf
            try:
                samples, data_file_paths = convert_to_lm_eval_format(
                    config, logger, subset_df, data_path
                )
                create_task_config(
                    config,
                    logger,
                    task_name=f"{config.lm_eval_task.dataset_name}_{subset}",
                    update_args=OrderedDict(
                        {
                            "task": task_name,
                            "include": base_conf_path.absolute().as_posix(),
                            "dataset_kwargs": {"data_files": data_file_paths},
                            "dataset_path": "parquet",
                        }
                    ),
                )
                added_subsets.add(task_name)
            except:
                logger.error(f"Error saving subset: {subset}")
                logger.error(traceback.format_exc())
        group_kwargs = {
            "group": config.lm_eval_task.dataset_name,
            "task": list(added_subsets),
            "aggregate_metric_list": config.lm_eval_task.aggregate_metric_list,
            "metadata": config.lm_eval_task.metadata,
        }
        group_conf_path = create_task_config(
            config,
            logger,
            task_name=config.lm_eval_task.dataset_name,
            # relative_path="",
            config_name=f"_{config.lm_eval_task.dataset_name}",
            base_dir=False,
            update_args=group_kwargs,
        )
    else:
        data_path = output_dir / f"{config.lm_eval_task.dataset_name}_lm_eval"
        samples, data_file_paths = convert_to_lm_eval_format(
            config,
            logger,
            df,
            data_path,
        )

    logger.info(f"\\nConversion complete! Files created in {output_dir}")
    logger.info(f"To run evaluation:")
    logger.info(
        f"lm_eval --model hf --model_args pretrained=<model_name> --tasks {config.lm_eval_task.dataset_name} --device cuda"
    )


def main():
    config: DatasetConverterConfig = load_config(DatasetConverterConfig)
    logger = setup_logger(config, "dataset_converter")

    if config.version == "v01":
        transform_v01_to_lm_eval_format(config.dataset_path, config.output_dir)
    elif config.version == "v1":
        transform_v1(config, logger)


if __name__ == "__main__":
    main()

# python xarch_tokenizers/scripts/convert_dataset_to_hf_format.py xarch_tokenizers/configs/tokenization_robustness/v101/convert_v101_to_lm_eval.yaml
# lm_eval
