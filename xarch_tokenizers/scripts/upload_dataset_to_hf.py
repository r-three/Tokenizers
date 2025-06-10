"""
Script to upload translated [multilingual] datasets to Hugging Face Hub.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo, login
from transformers import (
    HfArgumentParser,
)

from xarch_tokenizers.config import Config
from xarch_tokenizers.logging.logger import setup_logger


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

    dataset_name_prefix: str = field(
        default="translated-", metadata={"help": "Prefix for dataset names on HF Hub"}
    )

    # Dataset metadata
    dataset_tags: List[str] = field(
        default_factory=lambda: [
            "multilingual",
            "translated",
            "evaluation",
            "benchmark",
        ],
        metadata={"help": "Tags for the dataset on HF Hub"},
    )

    license_name: str = field(
        default="cc-by-sa-4.0", metadata={"help": "License name for the dataset"}
    )

    add_original_dataset_tag: bool = field(
        default=True, metadata={"help": "Add original dataset as a tag"}
    )
    # Processing options
    force_upload: bool = field(
        default=False, metadata={"help": "Force upload even if dataset already exists"}
    )
    upload_all: bool = field(
        default=False, metadata={"help": "Upload all datasets in the directory"}
    )
    private: bool = field(
        default=True, metadata={"help": "Pass false to make the dataset public."}
    )

    def __post_init__(self):
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN", None)
        return super().__post_init__()


def load_jsonl_file(filepath: Path) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def create_dataset_dict(dataset_dir: Path) -> DatasetDict:
    """Create a DatasetDict from JSONL files in the dataset directory."""
    dataset_dict = DatasetDict()

    # Map file names to dataset splits
    split_mapping = {
        "train.jsonl": "train",
        "validation.jsonl": "validation",
        "dev.jsonl": "validation",  # Map dev to validation
        "test.jsonl": "test",
    }

    # Load each split
    for filename, split_name in split_mapping.items():
        filepath = dataset_dir / filename
        if filepath.exists():
            # Load the data
            data = load_jsonl_file(filepath)

            # Convert to DataFrame first for easier processing
            df = pd.DataFrame(data)

            # Create dataset
            dataset = Dataset.from_pandas(df)

            # Add to dataset dict
            if split_name not in dataset_dict:
                dataset_dict[split_name] = dataset
            elif split_name == "validation" and "dev.jsonl" in filename:
                # If we already have a validation set from validation.jsonl,
                # and this is from dev.jsonl, rename to dev
                dataset_dict["dev"] = dataset
    return dataset_dict


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
    hf_dataset_name = (
        f"{config.dataset_name_prefix}{dataset_name}-{source_lang}-to-{target_lang}"
    )

    # Build tags list
    tags = config.dataset_tags.copy()
    tags.extend([f"source-language-{source_lang}", f"target-language-{target_lang}"])
    if config.add_original_dataset_tag:
        tags.append(f"original-{dataset_name}")

    return {
        "hf_dataset_name": hf_dataset_name,
        "metadata": metadata,
        "tags": tags,
        "description": f"This dataset contains the {dataset_name} benchmark translated from {source_lang} to {target_lang}. It can be used for cross-lingual evaluation of language models.",
    }


def upload_dataset(dataset_dir: Path, config: HFUploadConfig) -> None:
    """Upload a single dataset to Hugging Face Hub."""
    print(f"Processing dataset directory: {dataset_dir}")

    # Create Dataset object
    try:
        dataset_dict = create_dataset_dict(dataset_dir)
        if not dataset_dict:
            print(f"  No valid data files found in {dataset_dir}")
            return

        print(f"  Created dataset with splits: {list(dataset_dict.keys())}")
    except Exception as e:
        print(f"  Failed to create dataset: {e}")
        return
    # TODO: not uploading the readme
    # Extract metadata
    metadata_info = extract_dataset_metadata_translation(dataset_dir, config)
    hf_dataset_name = metadata_info["hf_dataset_name"]

    # Determine repo_id based on whether an organization is specified
    if config.hf_organization:
        repo_id = f"{config.hf_organization}/{hf_dataset_name}"
    else:
        repo_id = hf_dataset_name

    print(f"  Preparing to upload to: {repo_id}")

    # Create repo if it doesn't exist or force_upload is True
    api = HfApi()
    try:
        if config.force_upload:
            print(f"  Force upload enabled - creating/overwriting repo")
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                token=config.hf_token,
                private=config.private,
                exist_ok=True,
            )
        else:
            try:
                # Check if repo exists
                api.repo_info(repo_id=repo_id, repo_type="dataset")
                print(
                    f"  Repository {repo_id} already exists. Use --force_upload to overwrite."
                )
                return
            except Exception:
                # Repo doesn't exist, create it
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=config.private,
                    token=config.hf_token,
                    exist_ok=True,
                )
    except Exception as e:
        print(f"  Failed to create repository: {e}")
        return

    # Upload each split
    try:
        # Push dataset to Hub
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            token=config.hf_token,
            private=config.private,
            # tags=metadata_info["tags"],
            # license=config.license_name,
        )

    except Exception as e:
        print(f"  Failed to upload dataset: {e}")


def upload():
    parser = HfArgumentParser((HFUploadConfig,))
    (config,) = parser.parse_args_into_dataclasses()

    # Login to Hugging Face
    if not config.hf_token:
        print(
            "No HF_TOKEN provided. Please set the HF_TOKEN environment variable or use --hf_token"
        )
        # return
    login(token=config.hf_token)

    # Setup logger
    logger = setup_logger(config)
    logger.info(f"Uploading to huggingface with config: {config}")

    # Get list of dataset directories to process
    input_dir_path = Path(config.input_dir)
    if not input_dir_path.exists():
        print(f"Input directory {input_dir_path} does not exist")
        return

    if config.upload_all:
        # Process all subdirectories
        dataset_dirs = [d for d in input_dir_path.iterdir() if d.is_dir()]
    elif config.datasets:
        # Process only specified directories
        dataset_dirs = [input_dir_path / d for d in config.datasets]
    else:
        print(
            "No datasets specified. Use --upload_all or --datasets to specify which datasets to upload"
        )
        return

    # Upload each dataset
    print(f"Found {len(dataset_dirs)} datasets to process")
    for dataset_dir in dataset_dirs:
        upload_dataset(dataset_dir, config)


if __name__ == "__main__":
    upload()
