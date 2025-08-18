"""
Script to upload datasets to Hugging Face Hub.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import wrap
from time import sleep
from typing import Any, Dict, List, Optional

import huggingface_hub
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, create_repo, login
from transformers import (
    HfArgumentParser,
)

from xarch_tokenizers.config import Config
from xarch_tokenizers.logging.logger import setup_basic_logger, setup_logger

logger = setup_basic_logger()


def wrap_huggingface_hub_op(func, logger, success_message=None, error_message=None):
    """Decorator to wrap Hugging Face Hub operations with error handling."""

    if error_message is None:
        error_message = "Error occurred in Hugging Face Hub operation"

    def wrapper(*args, **kwargs):
        sleep_time = 5
        num_errs = 0
        while True:
            try:
                res = func(*args, **kwargs)
                if res is None:
                    return True
                return res
            except huggingface_hub.utils.HfHubHTTPError as e:
                logger.error(f"Error occurred in Hugging Face Hub operation: {e}")
                sleep(sleep_time)
                num_errs += 1
                if num_errs % 3 == 0:
                    sleep_time *= 2
                logger.info(f"Sleeping {sleep_time}")

            except Exception as e:
                logger.error(f"{error_message}: {e}")
                return False

    return wrapper


@dataclass
class HFUploadConfig(Config):
    """Configuration for Hugging Face dataset upload."""

    # Translated dataset directory
    input_dir: str = field(
        default="translated_tasks",
        metadata={"help": "Directory containing translated datasets"},
    )

    # HF Hub settings
    hf_token: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face API token (or set HF_TOKEN env var)"},
    )

    hf_organization: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face organization (omit for personal account)"},
    )
    dataset_name: str = field(
        default=None, metadata={"help": "Name of the dataset on HF Hub"}
    )

    # Dataset metadata
    tag: str = field(default=None, metadata={"help": "Tag for the dataset on HF Hub"})

    license_name: str = field(
        default="cc-by-sa-4.0", metadata={"help": "License name for the dataset"}
    )

    # Processing options
    upload_individually: bool = field(
        default=False, metadata={"help": "Upload datasets individually"}
    )
    private: bool = field(
        default=True, metadata={"help": "Pass false to make the dataset public."}
    )
    is_translation: bool = field(
        default=True,
        metadata={"help": "Whether the dataset is a translated version of a benchmark"},
    )

    dataset_card: dict = field(
        default_factory=lambda: {},
        metadata={"help": "Dataset card info to upload to Hugging Face Hub"},
    )
    collections: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of collections to which the dataset belongs"},
    )

    def __post_init__(self):
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN", None)
        if self.hf_organization is None:
            self.hf_organization = wrap_huggingface_hub_op(
                huggingface_hub.whoami, logger
            )().get("name", None)
        return super().__post_init__()


def load_jsonl_file(filepath: Path) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def create_dataset_dict(
    config: HFUploadConfig, dataset_dir: Path, logger
) -> DatasetDict:
    """Create a DatasetDict from parquet|arrow|jsonl|json files in the dataset directory."""
    dataset_dict = DatasetDict()

    for suffix in ["parquet", "arrow", "jsonl", "json"]:
        files = list(dataset_dir.rglob(f"*.{suffix}"))
        logger.info(f"Found {len(files)} files with suffix {suffix}")
        for filepath in files:
            logger.info(f"Processing file: {filepath}")
            split_name = filepath.stem
            # Load the data
            if suffix == "parquet" or suffix == "arrow":
                data = pd.read_parquet(filepath)
            elif suffix == "jsonl":
                data = load_jsonl_file(filepath)
            elif suffix == "json":
                data = pd.read_json(filepath)

            df = pd.DataFrame(data)
            if config.flatten_metadata and "metadata" in df.columns:
                metadata = data.pop("metadata")
                for key, value in metadata.items():
                    data[key] = value
            import numpy as np

            df.replace(np.nan, "", inplace=True)
            # Create dataset
            dataset = Dataset.from_pandas(df)
            dataset_dict[split_name] = dataset
        if len(files) > 0:
            logger.info(
                f"Found {len(files)} files with suffix {suffix}, continuing with next directory..."
            )
            break
    return dataset_dict


def get_dataset_name(dataset_dir: Path, config: HFUploadConfig) -> str:
    """Extract metadata for non-translation datasets."""
    if config.is_translation:
        return extract_dataset_metadata_translation(dataset_dir, config)[
            "hf_dataset_name"
        ]
    dataset_name = (
        Path(dataset_dir).name if config.dataset_name is None else config.dataset_name
    )
    return dataset_name


def extract_dataset_metadata_translation(
    dataset_dir: Path, config: HFUploadConfig
) -> Dict[str, Any]:
    """Extract metadata from the dataset directory."""
    # Get dataset name components from directory name
    dir_name = dataset_dir.name

    # Expected format: {dataset_name}_{source_lang}_to_{target_lang}
    parts = dir_name.split("_")
    if len(parts) >= 4 and "to" in parts:
        to_index = parts.index("to")
        dataset_name = "_".join(parts[: to_index - 1])
        source_lang = parts[to_index - 1]
        target_lang = parts[to_index + 1]
    else:
        # Fallback if directory name doesn't match expected pattern
        dataset_name = dir_name
        source_lang = "unknown"
        target_lang = "unknown"

    # Check if a translation config file exists
    config_file = dataset_dir / "translation_config.json"
    config_data = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

    # Build metadata
    metadata = {
        "original_dataset": dataset_name,
        "source_language": source_lang,
        "target_language": target_lang,
        "translation_method": config_data.get(
            "translation_model_template", "Helsinki-NLP/opus-mt"
        ),
        "translation_date": config_data.get("experiment_name", "").split("_")[-1]
        if config_data.get("experiment_name")
        else None,
    }

    # Create human-readable name and description
    hf_dataset_name = f"{dataset_name}-{source_lang}-to-{target_lang}"

    # Build tags list
    tags = config.tags.copy()
    tags.extend([f"source-language-{source_lang}", f"target-language-{target_lang}"])
    tags.append(f"original-{dataset_name}")

    return {
        "hf_dataset_name": hf_dataset_name,
        "metadata": metadata,
        "tags": tags,
        "description": f"This dataset contains the {dataset_name} benchmark translated from {source_lang} to {target_lang}. It can be used for cross-lingual evaluation of language models.",
    }


def get_repo_id(dataset_name: str, config: HFUploadConfig) -> str:
    """Generate repository ID based on configuration."""
    if config.hf_organization:
        return f"{config.hf_organization}/{dataset_name}"
    else:
        username = wrap_huggingface_hub_op(huggingface_hub.whoami)(
            token=config.hf_token
        )["name"]
        return f"{username}/{dataset_name}"


def create_dataset_card(
    config: HFUploadConfig, config_names: List[str] = None
) -> DatasetCard:
    """Create a dataset card for the dataset."""
    data = DatasetCardData(
        language=config.dataset_card.get("language"),
        license=config.dataset_card.get("license"),
        annotations_creators=config.dataset_card.get("annotations_creators"),
        language_creators=config.dataset_card.get("language_creators"),
        multilinguality=config.dataset_card.get("multilinguality"),
        size_categories=config.dataset_card.get("size_categories"),
        source_datasets=config.dataset_card.get("source_datasets"),
        task_categories=config.dataset_card.get("task_categories"),
        task_ids=config.dataset_card.get("task_ids"),
        paperswithcode_id=config.dataset_card.get("paperswithcode_id"),
        pretty_name=config.dataset_card.get("pretty_name"),
        train_eval_index=config.dataset_card.get("train_eval_index"),
        config_names=config_names or config.dataset_card.get("config_names"),
        ignore_metadata_errors=config.dataset_card.get("ignore_metadata_errors"),
        tags=config.tags,
    )
    return DatasetCard.from_template(data, **config.dataset_card)


def upload_to_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    config: HFUploadConfig,
    logger,
    config_name: str = "default",
) -> None:
    """Upload dataset dictionary to Hugging Face Hub."""

    print(f"  Preparing to upload to: {repo_id}")

    # Upload dataset to Hub
    wrap_huggingface_hub_op(
        dataset_dict.push_to_hub,
        logger,
        success_message=f"  Successfully uploaded dataset to {repo_id}",
        error_message=f"  Failed to upload dataset to {repo_id}",
    )(
        repo_id=repo_id,
        token=config.hf_token,
        private=config.private,
        config_name=config_name,
        commit_message=f"Uploading {config_name} subset",
    )


def upload_dataset(dataset_dir: Path, config: HFUploadConfig, logger) -> None:
    """Upload dataset with multiple subsets from subdirectories."""
    # Extract base metadata
    base_dataset_name = get_dataset_name(dataset_dir, config)
    repo_id = get_repo_id(base_dataset_name, config)
    api = HfApi()
    try:
        # Check if repo exists
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception:
        # Repo doesn't exist, create it
        wrap_huggingface_hub_op(create_repo, logger)(
            repo_id=repo_id,
            repo_type="dataset",
            private=config.private,
            token=config.hf_token,
            exist_ok=True,
        )
    config_names = []

    card = create_dataset_card(config, config_names)
    card.push_to_hub(repo_id, token=config.hf_token)
    for subset_dir in dataset_dir.iterdir():
        subset_name = subset_dir.name
        print(f"  Processing subset: {subset_name}")

        subset_dataset_dict = create_dataset_dict(config, subset_dir, logger)
        if not subset_dataset_dict:
            print(f"    No valid data files found in subset {subset_name}")
            continue
        # todo: depreceate upload_individually
        if config.upload_individually:
            success = wrap_huggingface_hub_op(upload_to_hub, logger)(
                subset_dataset_dict, repo_id, config, logger, "default"
            )
        else:
            success = wrap_huggingface_hub_op(upload_to_hub, logger)(
                subset_dataset_dict, repo_id, config, logger, subset_name
            )
        if success:
            config_names.append(subset_name)

    if config.tag:
        wrap_huggingface_hub_op(huggingface_hub.create_tag, logger)(
            repo_id, tag=config.tag, repo_type="dataset", exist_ok=True
        )

    if config.collections:
        from huggingface_hub import add_collection_item, create_collection

        for collection_ in config.collections:
            collection = wrap_huggingface_hub_op(create_collection, logger)(
                title=collection_,
                namespace=config.hf_organization,
                exists_ok=True,
            )
            wrap_huggingface_hub_op(add_collection_item, logger)(
                collection.slug,
                item_id=repo_id,
                item_type="dataset",
                exists_ok=True,
            )


def upload():
    parser = HfArgumentParser((HFUploadConfig,))
    (config,) = parser.parse_args_into_dataclasses()

    # Login to Hugging Face
    if not config.hf_token:
        print(
            "No HF_TOKEN provided. Please set the HF_TOKEN environment variable or use --hf_token"
        )
    login(token=config.hf_token)

    # Setup logger
    logger = setup_logger(config)
    logger.info(f"Uploading to huggingface with config: {config}")

    # Get list of dataset directories to process
    input_dir_path = Path(config.input_dir)
    if not input_dir_path.exists():
        print(f"Input directory {input_dir_path} does not exist")
        return

    if config.upload_individually:
        # Process all subdirectories and upload them as individual datasets
        dataset_dirs = [d for d in input_dir_path.iterdir() if d.is_dir()]
        for dataset_dir in dataset_dirs:
            upload_dataset(dataset_dir, config, logger)
    else:
        upload_dataset(input_dir_path, config, logger)


if __name__ == "__main__":
    upload()
