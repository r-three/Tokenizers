#!/usr/bin/env python3
"""
Tokenization Robustness Dataset Test Converter

This script converts the structured tokenization robustness dataset test set into a format
that is directly compatible with lm-evaluation-harness.
It creates parquet files and creates corresponding yaml files in the lm_eval directory
"""

## TODO: do i need to add the subset config to daaset card
## TODO: language collections
# TODO: add lang fitlering??
import enum
import glob
import json
import math
import random
import traceback
import warnings
from codecs import ignore_errors
from dataclasses import dataclass, field, fields
from pathlib import Path
from re import L
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from cv2 import add
from datasets import Dataset
from git import Tree
from transformers import AutoModel, AutoTokenizer

# For tokenizer loading
from xarch_tokenizers.experiment_config import load_config
from xarch_tokenizers.logging.logger import setup_logger
from xarch_tokenizers.scripts.lingua_tokenizers import Tokenizer, build_tokenizer
from xarch_tokenizers.scripts.upload_dataset_to_hf import (
    HFUploadConfig,
    get_dataset_name,
    get_repo_id,
    upload,
    upload_dataset,
)
from xarch_tokenizers.utils.utils import find_package_dir

LM_EVAL_PKG_DIR = Path(find_package_dir("lm_eval"))
# Usage:
# python xarch_tokenizers/scripts/convert_dataset_to_hf_format.py xarch_tokenizers/configs/tokenization_robustness/v101/convert_v103_to_lm_eval.yaml
# python xarch_tokenizers/scripts/convert_dataset_to_hf_format.py xarch_tokenizers/configs/tokenization_robustness/v101/convert_v102_to_lm_eval.yaml
# python xarch_tokenizers/scripts/convert_dataset_to_hf_format.py xarch_tokenizers/configs/tokenization_robustness/v101/convert_v101_to_lm_eval.yaml
# python xarch_tokenizers/scripts/upload_dataset_to_hf.py --input_dir=data/v101 --upload_all=true --private=false --is_translation=false  --upload_individually=false


@dataclass
class LmEvalTaskArgs:
    # Dataset configuration
    dataset_path: str = field(
        metadata={"help": "the name of the dataset on the HF Hub."}
    )
    dataset_name: str = field(
        default="null",
        metadata={
            "help": "the dataset configuration to use. Leave `null` if your dataset does not require a config to be passed. See https://huggingface.co/docs/datasets/load_hub#configurations for more info."
        },
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
        default=None, metadata={"help": "split name of test set, or `null`"}
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
    fewshot_split: Optional[str] = field(
        default=None,
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
    task: Optional[str] = field(
        default=None, metadata={"help": "name of the task (mandatory)"}
    )
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
class DatasetConverterConfig(HFUploadConfig):
    dataset_path: Optional[Path] = field(
        # dataset_path: Optional[Union[str, Path]] = field(
        default=None,
        metadata={"help": "Path to the dataset .json | or csv", "required": True},
    )
    dataset_format: Literal["json", "jsonl", "arrow", "parquet"] = field(
        default="parquet",
        metadata={"help": "Format of the dataset"},
    )
    output_dir: Optional[Path] = field(
        # output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Out path for writing the pivoted datasets"},
    )
    separator: str = field(
        default="|||",
        metadata={
            "help": "If the provided dataset_path is a csv pass the relevant separator."
        },
    )
    sheet_name: str = field(
        default="Examples-cleaned",
        metadata={
            "help": "If the provided dataset_path is an excel file, pass the relevant sheet name."
        },
    )
    version: Literal["v01", "v1", "collection"] = field(
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
    split_field: str = field(
        default="split",
        metadata={"help": "Name of the field containing the split."},
    )
    combine_all_splits: bool = field(
        default=False,
        metadata={
            "help": "Whether to melt dataset into just test and flatten the split field into the split column."
        },
    )
    flatten_metadata: bool = False
    create_subset_dirs: bool = False
    subset_by: Optional[str] = None
    dataset_by: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the field corresponding to the dataset, only relevant when --collections is passed."
        },
    )
    lm_eval_task: LmEvalTaskArgs = field(
        # lm_eval_task: Optional[LmEvalTaskArgs] = field(
        default_factory=LmEvalTaskArgs,
        metadata={"help": "LM Evaluation task configuration"},
    )
    record_tokenizer_stats: bool = False
    _create_experiment_dir: bool = False
    set_id_field: str = "Set Id"
    variation_id_field: str = "Variation Id"

    upload_to_hf: bool = False
    lm_eval_local_or_hf: Literal["local", "hf"] = field(
        default="local",
        metadata={
            "help": "Whether to use local files created or HF for LM Evaluation Harness."
        },
    )
    split_by_lang: Optional[bool] = field(
        default=False, metadata={"help": "Whether to split the dataset by language."}
    )

    def __post_init__(self):
        self.dataset_path = Path(str(self.dataset_path))
        if not self.dataset_path.resolve().absolute().exists():
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
        if self.dataset_by and self.collections is None:
            warnings.warn(
                "Dataset by argument will be ignored since no collection is passed"
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
    task_name: Optional[str] = None,
    config_name: Optional[Union[Path, str]] = None,
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
    config_path.parent.mkdir(exist_ok=True, parents=True)
    print(config_path)
    # Write YAML config
    with open(config_path, "w") as f:
        yaml.dump(
            {"task": task_name} | update_args,
            f,
            default_flow_style=False,
        )

    logger.debug(f"Created task config: {config_path}")
    return config_path


def convert_to_lm_eval_format(
    config: DatasetConverterConfig,
    logger,
    df: pd.DataFrame,
    output_dir: Path,
):
    """Convert dataset to LM Evaluation Harness format."""

    samples = []
    df = df.replace(np.nan, "")

    for idx, row in df.iterrows():
        # Extract question and choices
        question = str(row.get(config.question_field, "")).strip()
        if not question:
            print(f"Warning: Question is empty for row {idx}, skipping")
            continue

        choices = []
        choice_labels = ["A", "B", "C", "D"]

        for i, opt_col in enumerate(config.option_fields):
            opt_text = str(row.get(opt_col, "")).strip()
            if opt_text and opt_text != "nan":
                choices.append(opt_text)

        if len(choices) == 0:
            logger.debug(
                f"Warning: No choices found for row {idx}, question: {question}, skipping"
            )
            continue
        correct_idx = random.randint(0, len(config.option_fields))
        correct_answer = str(row.get("Correct", "")).strip()
        choices.insert(correct_idx, correct_answer)
        metadata = {
            opt_field: row[opt_val]
            # if isinstance(row[opt_val], str)
            # or (isinstance(row[opt_val], float) and not math.isnan(row[opt_val]))
            # else ""
            for opt_field, opt_val in config.metadata_fields.items()
        }
        if config.record_tokenizer_stats:
            metadata["vanilla_cos_sim_to_canonical"] = row[
                "vanilla_cos_sim_to_canonical"
            ]
            metadata["trimmed_cos_sim_to_canonical"] = row[
                "trimmed_cos_sim_to_canonical"
            ]
            metadata["token_counts"] = row["token_counts"]
        split = str(row.get(config.split_field, "")).strip()
        if not split or split.lower() in ["", "nan", "none", "null"]:
            split = "test"
            logger.debug(f"Warning: Split is empty for row {idx}, setting to test")
        sample = {
            "question": question,
            "choices": choices,
            "answer": correct_idx,
            "answer_label": choice_labels[correct_idx],
            "split": split,
        }
        if config.flatten_metadata:
            sample.update(metadata)
        else:
            sample["metadata"] = metadata
        samples.append(sample)
    if len(samples) == 0:
        logger.warning(f"No valid samples found for {output_dir}, skipping")
        return None, None
    samples = pd.DataFrame(samples)
    data_files = {}
    samples["set_id"] = samples["set_id"].astype("string")
    samples["variation_id"] = samples["variation_id"].astype("string")
    samples["lang"] = samples["lang"].astype("string")
    for split in samples["split"].unique():
        try:
            # output_path.mkdir(exist_ok=True, parents=True)
            if config.combine_all_splits:
                output_path = output_dir / "test"
                dataset = Dataset.from_pandas(samples)
                split = "test"
            else:
                output_path = output_dir / split
                dataset = Dataset.from_pandas(
                    samples[samples["split"] == split], split=split
                )
            ## TODO: save as other formats
            output_path = output_dir / f"{split}.parquet"
            dataset.to_parquet(str(output_path))
            data_files[split] = [
                p.absolute().as_posix() for p in output_dir.rglob(f"{split}*.parquet")
            ]
            if config.combine_all_splits:
                break
        except Exception as e:
            logger.error(e)
            import code

            code.interact(local=locals() | globals())

    logger.info(f"Converted {len(samples)} samples to {config.output_dir}")
    return samples, data_files


def cleanup_excel(df: pd.DataFrame, config: DatasetConverterConfig):
    """Cleans up the Excel DataFrame by removing rows with empty questions and certain versions (e.g. depreceated rows)."""
    df = df.dropna(subset=[config.question_field])
    df = df[
        ~(
            df["Version"]
            .fillna("")
            .str.lower()
            .str.startswith(tuple(["depreceate", "ignore", "maybe", "no"]))
        )
    ]
    return df


def cleanup_str(s):
    """Clean up a string to be used as a valid identifier (e.g., for filenames or task names)."""
    s = (
        s.replace(" & ", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
        .lower()
    )
    return s


def is_valid(s):
    if s is None:
        return False
    s = str(s).strip().lower()
    return s not in ["null", "na", "nan", "not_found", "none", "#value!"]


def create_subsets(
    df,
    subset,
    logger,
    task_name,
    output_dir,
    config,
    base_conf_path,
    config_name: Optional[str] = None,
):
    logger.info(f"Processing subset: {subset}.")
    subset_df = df[df["_subset"] == subset]
    subset = cleanup_str(str(subset))
    if not is_valid(subset):
        return
    data_path = output_dir / task_name
    data_path.mkdir(exist_ok=True, parents=True)
    try:
        samples, data_file_paths = convert_to_lm_eval_format(
            config,
            logger,
            subset_df,
            data_path,
        )
        if samples is None or len(samples) == 0:
            return None
        update_args = {
            "task": task_name,
            "include": base_conf_path.name,
            "dataset_name": task_name,
        }
        if config.lm_eval_local_or_hf == "local":
            update_args["dataset_kwargs"] = {"data_files": data_file_paths}
            update_args["dataset_path"] = "parquet"
        else:
            # subset id
            update_args["dataset_name"] = subset
            update_args["dataset_name"] = task_name
        create_task_config(
            config,
            logger,
            task_name=task_name,
            update_args=update_args,
            config_name=config_name,
        )
    except:
        logger.error(f"Error saving subset: {subset}")
        logger.error(traceback.format_exc())

    return subset


def transform_v1(config: DatasetConverterConfig, logger):
    df = read_data(config, logger)
    update_args = dict(
        dataset_name=None,
        dataset_path="parquet",
    )
    if config.lm_eval_local_or_hf == "hf":
        update_args["dataset_path"] = get_repo_id(
            get_dataset_name(config.dataset_path, config), config
        )
    base_conf_path = create_task_config(
        config,
        logger,
        task_name=config.lm_eval_task.dataset_name,
        config_name=f"{config.lm_eval_task.dataset_name}_base",
        base_dir=True,
        update_args=config.lm_eval_task.export_dict() | update_args,
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
            task_name = f"{config.lm_eval_task.dataset_name}_{subset}"
            subset = create_subsets(
                df, subset, logger, task_name, output_dir, config, base_conf_path
            )
            if subset is not None:
                added_subsets.add(subset)
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

    logger.info(f"\\nConversion complete! Files created in {config.output_dir}")
    logger.info(f"To run evaluation:")
    logger.info(
        f"lm_eval --model hf --model_args pretrained=<model_name> --tasks {config.lm_eval_task.dataset_name} --device cuda"
    )


def transform_w_collection(
    df,
    config: DatasetConverterConfig,
    logger,
    prefix: Optional[str] = "",
):
    if config.dataset_by is not None:
        df["_dataset"] = df[config.dataset_by].apply(
            lambda x: x.split(",")[0] if isinstance(x, str) else x
        )
    else:
        # df["_dataset"] = config.lm_eval_task.dataset_name
        df["_dataset"] = config.dataset_name
    update_args = dict(
        # dataset_name=None,
        dataset_path="parquet",
    )

    def process_collection(df, collection_prefix: str = ""):
        all_datasets = df["_dataset"].unique()
        added_datasets = set()
        for dataset_name in all_datasets:
            filtered_df = df[df["_dataset"] == dataset_name]
            dataset_name = cleanup_str(str(dataset_name))
            # dataset_name = f"{collection_prefix}{cleanup_str(str(dataset_name))}"
            if not is_valid(dataset_name):
                continue
            output_dir = config.output_dir / dataset_name
            logger.info(f"Processing dataset: {dataset_name}.")
            base_conf_path = create_task_config(
                config,
                logger,
                task_name=dataset_name,
                config_name=f"{collection_prefix}{dataset_name}/{dataset_name}_base",
                base_dir=False,
                update_args=config.lm_eval_task.export_dict()
                | update_args
                | {
                    "dataset_path": f"{config.hf_organization}/{dataset_name}",
                    "dataset_name": dataset_name,
                },
            )
            if config.lm_eval_local_or_hf == "hf":
                update_args["dataset_path"] = get_repo_id(
                    get_dataset_name(dataset_name, config), config
                )

            if config.create_subset_dirs:
                if config.subset_by not in df.columns:
                    raise ValueError(
                        f"subset_by column ({config.subset_by}) not in the data."
                    )
                # TODO: add canonical subset
                filtered_df["_subset"] = filtered_df[config.subset_by].apply(
                    lambda x: x.split(",")[0] if isinstance(x, str) else x
                )
                mask = (
                    filtered_df[config.variation_id_field]
                    .astype(str)
                    .str.split(".")
                    .str[-1]
                    == "0"
                )
                filtered_df.loc[mask, "_subset"] = "cannonical"

                all_subsets = filtered_df["_subset"].unique()
                added_subsets = set()
                for subset in all_subsets:
                    task_name = cleanup_str(f"{dataset_name}_{subset}")
                    # task_name = cleanup_str(f"{subset}")
                    subset = create_subsets(
                        filtered_df,
                        subset,
                        logger,
                        task_name,
                        output_dir,
                        config,
                        base_conf_path,
                        config_name=f"{prefix}{dataset_name}/{task_name}",
                    )
                    if subset is not None:
                        added_subsets.add(task_name)
                if len(added_subsets) == 0:
                    logger.warning(f"No subsets created for {dataset_name}.")
                    continue
                # group for dataset
                group_kwargs = {
                    "group": dataset_name,
                    "task": list(added_subsets),
                    "aggregate_metric_list": config.lm_eval_task.aggregate_metric_list,
                    "metadata": config.lm_eval_task.metadata,
                }
                group_conf_path = create_task_config(
                    config,
                    logger,
                    task_name=cleanup_str(f"{prefix}{dataset_name}"),
                    # task_name=dataset_name,
                    # relative_path="",
                    config_name=f"{prefix}{dataset_name}/_{dataset_name}",
                    # config_name=f"_{dataset_name}",
                    base_dir=False,
                    update_args=group_kwargs,
                )
                added_datasets.add(dataset_name)

            for collection in config.collections:
                collection = cleanup_str(f"{collection_prefix}{collection}")
                # group for collection
                # tODO: check if this is the best way
                group_kwargs = {
                    "group": collection,
                    "task": list(added_datasets),
                    "aggregate_metric_list": config.lm_eval_task.aggregate_metric_list,
                    "metadata": config.lm_eval_task.metadata,
                }
                group_conf_path = create_task_config(
                    config,
                    logger,
                    task_name=collection,
                    config_name=f"{collection_prefix}_{collection}",
                    base_dir=False,
                    update_args=group_kwargs,
                )

    process_collection(df, collection_prefix=prefix)

    logger.info(f"\\nConversion complete! Files created in {config.output_dir}")
    logger.info(f"To run evaluation:")
    logger.info(
        f"lm_eval --model hf --model_args pretrained=<model_name> --tasks {config.lm_eval_task.dataset_name} --device cuda"
    )


def record_tokenizer_stats(df, config: DatasetConverterConfig):
    tokenizers = [
        "google/gemma-2-2b",
        "common-pile/comma-v0.1-1t",
        "meta-llama/Llama-3.2-1B",
        "microsoft/Phi-3-mini-4k-instruct",
        "gpt2",
        "bigscience/bloom",
        "facebook/xglm-564M",
        "mistralai/tekken",
        "google/byt5-small",
        "google-bert/bert-base-multilingual-cased",
        "Qwen/Qwen3-8B",
        "tokenmonster/englishcode-32000-consistent-v1",
        "tiktoken/gpt-4o",
        "CohereLabs/aya-expanse-8b",
    ]
    models = [
        "google-gemma-2-2b",
        "common-pile-comma-v0.1",
        "meta-llama-Llama-3.2-1B",
        "microsoft-Phi-3-mini-4k-instruct",
        "gpt2",
        "bigscience-bloom",
        "facebook-xglm-564M",
        "mistralai-tekken",
        "google-byt5-small",
        "google-bert-bert-base-multilingual-cased",
        "Qwen-Qwen3-8B",
        "tokenmonster-englishcode-32000-consistent-v1",
        "tiktoken-gpt-4o",
        "cohereLabs-aya-expanse-8b",
    ]

    ## cosine sim. against canonical
    canonical_mask = df[config.variation_id_field].apply(
        lambda x: str(x).split(".")[1] == "0"
    )
    import torch

    vanilla_cos_sims = pd.DataFrame()
    trimmed_cos_sims = pd.DataFrame()

    for i, (tok_name, model_name) in enumerate(zip(tokenizers, models)):
        tok = build_tokenizer(tok_name)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        import gc

        gc.collect()
        model = AutoModel.from_pretrained(
            f"r-three/supertoken_models-llama_{model_name}"
        )
        # model = model.to(device)
        with torch.no_grad():
            for i, row in df.iterrows():
                import code

                set_id = row[config.set_id_field]

                if canonical_mask[i]:
                    print(f"Skipping canonical example set_id {set_id} in row {i}")
                    vanilla_cos_sims.loc[i, tok_name] = 1.0
                    trimmed_cos_sims.loc[i, tok_name] = 1.0
                    continue

                var_id = row[config.variation_id_field]
                try:
                    canonical_text = df[
                        canonical_mask & (df[config.set_id_field] == set_id)
                    ].iloc[0][config.question_field]
                except:
                    import code

                    code.interact(local=locals() | globals())
                # code.interact(local=locals() | globals())
                if not canonical_text:
                    print(
                        f"Something wrong with the canonical text for set id {set_id}"
                    )
                canonical_tokens = tok.encode(
                    canonical_text, add_bos=False, add_eos=False
                )
                perturbed_text = row[config.question_field]
                perturbed_tokens = tok.encode(
                    perturbed_text, add_bos=False, add_eos=False
                )
                canonical_embed = (
                    model.embed_tokens(torch.tensor(canonical_tokens))
                    .cpu()
                    .mean(axis=0)
                )
                perturbed_embed = (
                    model.embed_tokens(torch.tensor(perturbed_tokens))
                    .cpu()
                    .mean(axis=0)
                )
                vanilla_cos_sim = torch.nn.functional.cosine_similarity(
                    canonical_embed, perturbed_embed, dim=0
                ).item()

                ## brute force trimming
                start_ind_can, start_ind_pert = 0, 0
                end_ind_can, end_ind_pert = (
                    len(canonical_tokens) - 1,
                    len(perturbed_tokens) - 1,
                )
                while start_ind_can < end_ind_can and start_ind_pert < end_ind_pert:
                    if (
                        canonical_tokens[start_ind_can]
                        == perturbed_tokens[start_ind_pert]
                    ):
                        start_ind_can += 1
                        start_ind_pert += 1
                    else:
                        break
                while end_ind_can > start_ind_can and end_ind_pert > start_ind_pert:
                    if canonical_tokens[end_ind_can] == perturbed_tokens[end_ind_pert]:
                        end_ind_can -= 1
                        end_ind_pert -= 1
                    else:
                        break
                # print(
                #     f"Row {i}, start_ind_can: {start_ind_can} - start_ind_pert: {start_ind_pert};\nend_ind_can: {end_ind_can} - end_ind_pert: {end_ind_pert}"
                # )
                canonical_tokens = canonical_tokens[start_ind_can : end_ind_can + 1]
                perturbed_tokens = perturbed_tokens[start_ind_pert : end_ind_pert + 1]
                canonical_embed = (
                    model.embed_tokens(torch.tensor(canonical_tokens))
                    .cpu()
                    .mean(axis=0)
                )
                perturbed_embed = (
                    model.embed_tokens(torch.tensor(perturbed_tokens))
                    .cpu()
                    .mean(axis=0)
                )
                trimmed_cos_sim = torch.nn.functional.cosine_similarity(
                    canonical_embed, perturbed_embed, dim=0
                ).item()
                vanilla_cos_sims.loc[i, tok_name] = vanilla_cos_sim
                trimmed_cos_sims.loc[i, tok_name] = trimmed_cos_sim
                # df.loc[i, f"{tok_name}_vanilla_cos_sim"] = vanilla_cos_sim
                # df.loc[i, f"{tok_name}_trimmed_cos_sim"] = trimmed_cos_sim
    import code

    df["vanilla_cos_sim_to_canonical"] = vanilla_cos_sims.to_dict(orient="records")
    df["trimmed_cos_sim_to_canonical"] = trimmed_cos_sims.to_dict(orient="records")

    token_counts = pd.DataFrame()
    for tok_name in tokenizers:
        tok = build_tokenizer(tok_name)
        token_counts[tok_name] = (
            df[config.question_field]
            .apply(lambda x: len(tok.encode(x, add_bos=False, add_eos=False)))
            .tolist()
        )
    df["token_counts"] = token_counts.to_dict(orient="records")

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
    # cleanup empty cells
    df = df.map(lambda x: x if is_valid(x) else "")

    if config.dataset_by:
        assert config.dataset_by in df.columns, ValueError(
            f"dataset_by column '{config.dataset_by}' not found in dataset."
        )
        df["_dataset"] = df[config.dataset_by].apply(
            lambda x: x.split(",")[0] if isinstance(x, str) else x
        )
        df = df[df[config.dataset_by].apply(is_valid)]
    if config.subset_by is not None:
        assert config.subset_by in df.columns, ValueError(
            f"subset_by column '{config.subset_by}' not found in dataset."
        )
        df["_subset"] = df[config.subset_by].apply(
            lambda x: x.split(",")[0] if isinstance(x, str) else x
        )
        df = df[df[config.subset_by].apply(is_valid)]
        # df = df[df["_subset"].apply(is_valid)]
    if config.record_tokenizer_stats:
        record_tokenizer_stats(df, config)
    logger.info(f"Loaded {len(df)} rows")

    # Create output directory
    output_dir_ = Path(config.output_dir)
    output_dir_.mkdir(parents=True, exist_ok=True)
    return df


def main():
    config: DatasetConverterConfig = load_config(DatasetConverterConfig)
    logger = setup_logger(config, "dataset_converter")
    output_dir = config.output_dir.resolve().absolute()

    if config.version == "v01":
        transform_v01_to_lm_eval_format(config.dataset_path, output_dir)
    elif config.version == "v1":
        transform_v1(config, logger)
    elif config.version == "collection":
        df = read_data(config, logger)
        output_dir = config.output_dir

        if config.split_by_lang:
            all_languages = df["Lang"].unique()
            for lang in all_languages:
                tmp_df = df[df["Lang"] == lang]
                if "eng" in lang:
                    continue
                config.output_dir = output_dir / lang
                if config.dataset_by:
                    tmp_df[config.dataset_by] = tmp_df[config.dataset_by].apply(
                        lambda x: f"{lang}_{x}"
                    )
                process_df(tmp_df)
                transform_w_collection(tmp_df, config, logger, prefix=f"{lang}/")
        else:
            transform_w_collection(df, config, logger, prefix=f"")

        if config.upload_to_hf:
            for dataset_name in config.output_dir.glob("*"):
                dataset_name = dataset_name.resolve().absolute()
                upload_dataset(dataset_name, config, logger)
        exit(0)
    if config.upload_to_hf:
        upload_dataset(output_dir, config, logger)


if __name__ == "__main__":
    main()

# lm_eval
