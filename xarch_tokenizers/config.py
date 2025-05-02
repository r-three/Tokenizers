"""Defines configureation class Config, and TaskConfig"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
import yaml
from rich import box
from rich.console import Console
from rich.table import Table

from xarch_tokenizers.data.data_config import TaskConfig

logger = logging.getLogger(__file__)


@dataclass
class Config:
    """Defines base config class for all experiments"""

    # Basic settings
    model_name: str = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "Base model used."},
    )
    # datasets: List[Union[str, Dict[str, Any], TaskConfig]] = field(
    datasets: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Datasets to use for evaluation/training",
            "nargs": "+",
        },
    )
    use_wandb: bool = field(
        default=False, metadata={"help": "Whether to use Weights & Biases for logging"}
    )
    wandb_project: str = field(default="xarch", metadata={"help": "W&B project name"})
    wandb_entity: Optional[str] = field(
        default=os.environ.get("WANDB_ENTITY"), metadata={"help": "W&B entity name"}
    )
    tags: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": "Tags passed to wandb.",
            "nargs": "*",
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    save_dir: str = field(
        default=Path("./results"), metadata={"help": "Directory to save results"}
    )
    resume_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to resume training from"}
    )
    experiment_name: str = field(
        default="xarch", metadata={"help": "Name for this experiment"}
    )
    debug: bool = field(default=False, metadata={"help": "Enable debug mode"})
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = field(
        default="INFO", metadata={"help": "Logging level"}
    )

    # Training settings
    batch_size: int = field(default=32, metadata={"help": "Training batch size"})
    tensor_parallel_size: int = field(
        default=None, metadata={"help": "Number of devices for tensor parallelism"}
    )

    # Evaluation settings
    evaluation_method: Literal["lm_eval", "custom"] = field(
        default="lm_eval", metadata={"help": "Evaluation method to use"}
    )
    num_fewshot: int = field(
        default=0, metadata={"help": "Number of examples to use in few-shot learning"}
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size for evaluation"}
    )
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="float32", metadata={"help": "Precision for model "}
    )

    # Resource management
    max_gpu_memory: Optional[str] = field(
        default=None, metadata={"help": "Maximum GPU memory to use (e.g., '24GiB')"}
    )
    load_in_8bit: bool = field(
        default=False, metadata={"help": "Whether to load model in 8-bit quantization"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether to load model in 4-bit quantization"}
    )

    # Advanced settings
    cache_dir: Optional[str] = field(
        default=f"{os.environ.get('SCRATCH')}/.cache",
        metadata={"help": "Directory to cache models and datasets"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer to use (defaults to model_name if None)"},
    )
    chat_template: Optional[str] = field(
        default=None, metadata={"help": "Name of the chat template to be used."}
    )
    apply_chat_template: bool = field(
        default=True,
        metadata={
            "help": "Pass true to apply chattemplate, recommended with Instruct models."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading models"},
    )
    revision: Optional[str] = field(
        default=None, metadata={"help": "Specific model revision to use"}
    )

    # Output and visualization
    save_results: bool = field(
        default=True, metadata={"help": "Whether to save evaluation results"}
    )
    dump_predictions: bool = field(
        default=True,
        metadata={"help": "If true, dumps model's predictions under predictions.json"},
    )
    results_format: Literal["json", "csv", "both"] = field(
        default="json", metadata={"help": "Format for saved results"}
    )
    generate_plots: bool = field(
        default=True, metadata={"help": "Whether to generate plots from results"}
    )
    plot_format: Literal["png", "pdf", "svg"] = field(
        default="png", metadata={"help": "Format for output plots"}
    )
    slurm_job_id: Optional[str] = field(
        default=None, metadata={"help": "Slurm job id, automatically parsed."}
    )

    _dataset_configs: Optional[List[str]] = field(default_factory=list)
    _create_experiment_dir: bool = True

    def __post_init__(self):
        if self.wandb_entity is None:
            self.wandb_entity = os.environ.get("WANDB_ENTITY")
        self.slurm_job_id = os.environ.get("SLURM_JOB_ID")
        # Convert string paths to Path objects
        self.save_dir = Path(self.save_dir)
        if self.resume_dir:
            self.resume_dir = Path(self.resume_dir)

        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique experiment path including timestamp
        if self.experiment_name == "xarch":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = "xarch_%s_%s" % (
                self.model_name.rsplit("/", maxsplit=1)[-1],
                timestamp,
            )
        self.experiment_dir = self.save_dir / self.experiment_name
        if self._create_experiment_dir:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        # Auto-detect GPU count for tensor parallelism if not specified
        if self.tensor_parallel_size is None:
            self.tensor_parallel_size = torch.cuda.device_count()
            if self.tensor_parallel_size == 0:
                self.tensor_parallel_size = 1  # Default to 1 if no GPUs are available

        #  Parse dataset configurations
        self._parse_datasets()

        # Set tokenizer to model name if not specified
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

        if self.cache_dir is None or self.cache_dir == "/checkpoint":
            self.cache_dir = (
                "/checkpoint/{os.environ.get('USER')}/{self.slurm_job_id}/.cache"
            )
        os.environ["$HUGGINGFACE_HUB_CACHE"] = self.cache_dir

    def _parse_datasets(self):
        """Parse the datasets field into TaskConfig objects."""
        self._dataset_configs = []

        for dataset in self.datasets:
            if isinstance(dataset, str):
                # Handle string format: either plain name or colon-separated with params
                self._dataset_configs.append(TaskConfig.from_string(dataset))
            elif isinstance(dataset, dict):
                # Handle dictionary format from YAML
                self._dataset_configs.append(TaskConfig.from_dict(dataset))
            elif isinstance(dataset, TaskConfig):
                # Already a TaskConfig object
                self._dataset_configs.append(dataset)
            else:
                raise ValueError(f"Unsupported dataset specification: {dataset}")

        # Update the original datasets field with names for compatibility
        self.datasets = [ds.name for ds in self._dataset_configs]

    def get_dataset_configs(self) -> List[TaskConfig]:
        """Get the parsed dataset configurations."""
        return self._dataset_configs

    def get_dataset_names(self) -> List[str]:
        """Get just the dataset names."""
        return self.datasets

    def get_dataset_tasks(self) -> List[str]:
        """
        Get dataset task specifications in lm-eval format.
        This handles any subset or split specifications.
        """
        tasks = []
        for ds_config in self._dataset_configs:
            task = ds_config.name
            if ds_config.subset:
                task = f"{task}_{ds_config.subset}"
            # Some datasets might use different formatting
            # Add custom formatting logic here as needed
            tasks.append(task)
        return tasks

    def save_config(self):
        """Save configuration to file in both YAML formats."""
        # Convert Path objects to strings for serialization
        config_dict = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
            if not k.startswith("_")  # Skip internal fields
        }

        # Add serialized dataset configs
        config_dict["dataset_configs"] = [ds.to_dict() for ds in self._dataset_configs]

        # Save as YAML
        yaml_path = self.experiment_dir / "config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        return yaml_path

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from file (JSON or YAML)."""
        config_path = Path(config_path)
        if config_path.suffix not in [".json", ".yaml", ".yml"]:
            raise ValueError(
                f"File format {config_path.suffix} not supported. "
                + "Please pass a json, yaml or yml file.",
            )
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} not found. Please pass an existing file."
            )

        if config_path.suffix.lower() in (".yaml", ".yml"):
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        else:  # Default to JSON
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

        # Handle dataset configs if present
        dataset_configs = config_dict.pop("dataset_configs", None)

        # Create base config
        config = cls(**{k: v for k, v in config_dict.items() if not k.startswith("_")})

        # Override datasets if dataset_configs was present
        if dataset_configs:
            config._dataset_configs = [
                TaskConfig.from_dict(ds) for ds in dataset_configs
            ]
            config.datasets = [ds.name for ds in config._dataset_configs]

        return config

    def get_override_config(self):
        """returns config to override lm_eval task definition
        WIP, could be depreceated in the future"""
        # TODO: write tests
        override_config = dict()
        for conf in self._dataset_configs:
            if conf.override_config:
                override_config[conf.lm_eval_alias] = conf.override_config
        return override_config

    def display(self):
        """Return a Rich-formatted string representation of the Config."""
        console = Console(record=True)  # , width=120)

        # Define sections and their fields
        sections = {
            "Basic Settings": [
                "model_name",
                "datasets",
                "experiment_name",
                "seed",
                "save_dir",
                "resume_dir",
                "debug",
                "log_level",
            ],
            "W&B Settings": ["use_wandb", "wandb_project", "wandb_entity", "tags"],
            "Training Settings": ["batch_size", "tensor_parallel_size"],
            "Evaluation Settings": [
                "evaluation_method",
                "num_fewshot",
                "eval_batch_size",
                "dtype",
            ],
            "Resource Management": ["max_gpu_memory", "load_in_8bit", "load_in_4bit"],
            "Advanced Settings": [
                "cache_dir",
                "tokenizer_name",
                "chat_template",
                "apply_chat_template",
                "trust_remote_code",
                "revision",
            ],
            "Output Settings": [
                "save_results",
                "dump_predictions",
                "results_format",
                "generate_plots",
                "plot_format",
                "slurm_job_id",
            ],
        }

        # Print title
        console.print(f"[bold blue]Config: {self.experiment_name}[/bold blue]")

        # Create and print a separate table for each section
        for section, fields in sections.items():
            table = Table(
                title=section, box=box.ROUNDED, title_style="bold green", expand=True
            )
            table.add_column("Field", style="cyan")
            table.add_column("Value")

            for field in fields:
                value = getattr(self, field)
                # Format various types appropriately
                if isinstance(value, list) and value:
                    if len(value) > 5:
                        formatted_value = f"[{', '.join(str(item) for item in value[:5])}... +{len(value) - 5} more]"
                    else:
                        formatted_value = f"[{', '.join(str(item) for item in value)}]"
                elif isinstance(value, Path):
                    formatted_value = str(value)
                elif value is None:
                    formatted_value = "[dim]None[/dim]"
                elif isinstance(value, bool):
                    formatted_value = f"[{'green' if value else 'red'}]{value}[/]"
                else:
                    formatted_value = str(value)

                table.add_row(field, formatted_value)

            # Print the table
            console.print(table)

        # Create a special table for dataset configurations
        if self._dataset_configs:
            ds_table = Table(
                title="Dataset Configurations",
                box=box.ROUNDED,
                title_style="bold magenta",
                expand=True,
            )
            ds_table.add_column("Index", style="dim")
            ds_table.add_column("Name", style="cyan")
            ds_table.add_column("Subset", style="yellow")
            ds_table.add_column("Override Config", style="green")

            for i, ds_config in enumerate(self._dataset_configs, 1):
                overrides = (
                    str(ds_config.override_config)
                    if hasattr(ds_config, "override_config")
                    and ds_config.override_config
                    else "-"
                )
                ds_table.add_row(
                    str(i), ds_config.name, ds_config.subset or "-", overrides
                )
            console.print(ds_table)

        # return console.export_text()
        # TODO; add experiment specific configs

    def __repr__(self) -> str:
        """Return a simple string representation for debugging."""
        return f"Config(model={self.model_name}, datasets={len(self.datasets)})"
