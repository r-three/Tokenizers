"""
Script to upload translated [multilingual] datasets to Hugging Face Hub.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import huggingface_hub
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
    dataset_name: str = field(
        default=None, metadata={"help": "Name of the dataset on HF Hub"}
    )

    # Dataset metadata
    dataset_tags: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Tags for the dataset on HF Hub"},
    )
    tag: str = field(default=None, metadata={"help": "Tag for the dataset on HF Hub"})

    license_name: str = field(
        default="cc-by-sa-4.0", metadata={"help": "License name for the dataset"}
    )

    # Processing options
    force_upload: bool = field(
        default=False, metadata={"help": "Force upload even if dataset already exists"}
    )
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


def extract_dataset_metadata(
    dataset_dir: Path, config: HFUploadConfig
) -> Dict[str, Any]:
    """Extract metadata for non-translation datasets."""
    if config.is_translation:
        return extract_dataset_metadata_translation(dataset_dir, config)
    dataset_name = (
        dataset_dir.name if config.dataset_name is None else config.dataset_name
    )
    hf_dataset_name = f"{dataset_name}"
    return {
        "hf_dataset_name": hf_dataset_name,
        "metadata": {"original_dataset": dataset_name},
        "tags": config.dataset_tags,
        "description": f"This dataset contains the {dataset_name} benchmark.",
    }


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
    tags = config.dataset_tags.copy()
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
        username = huggingface_hub.whoami(token=config.hf_token)["name"]
        return f"{username}/{dataset_name}"


def upload_to_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    config: HFUploadConfig,
    logger,
    config_name: str = "default",
) -> None:
    """Upload dataset dictionary to Hugging Face Hub."""

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
        logger.error(f"Repository creation failed: {e}")
        return

    # Upload dataset to Hub
    try:
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            token=config.hf_token,
            private=config.private,
            config_name=config_name,
            # tags=metadata_info["tags"],
            # license=config.license_name,
        )
        print(f"  Successfully uploaded dataset to {repo_id}")

    except Exception as e:
        print(f"  Failed to upload dataset: {e}")
        logger.error(f"Dataset upload failed: {e}")


def upload_dataset(dataset_dir: Path, config: HFUploadConfig, logger) -> None:
    """Upload dataset with multiple subsets from subdirectories."""

    # Extract base metadata
    metadata_info = extract_dataset_metadata(dataset_dir, config)
    base_dataset_name = metadata_info["hf_dataset_name"]
    repo_id = get_repo_id(base_dataset_name, config)

    for subset_dir in dataset_dir.iterdir():
        subset_name = subset_dir.name
        print(f"  Processing subset: {subset_name}")

        try:
            # Create dataset for this subset
            subset_dataset_dict = create_dataset_dict(config, subset_dir, logger)
            if not subset_dataset_dict:
                print(f"    No valid data files found in subset {subset_name}")
                continue
            if config.upload_individually:
                upload_to_hub(subset_dataset_dict, repo_id, config, logger, "default")
            else:
                upload_to_hub(subset_dataset_dict, repo_id, config, logger, subset_name)

        except Exception as e:
            print(f"    Failed to process subset {subset_name}: {e}")
            logger.error(f"Error processing subset {subset_name}: {e}")
            continue
    if config.tag:
        huggingface_hub.card
        huggingface_hub.create_tag(repo_id, tag=config.tag, repo_type="dataset")


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
