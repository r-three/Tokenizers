import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from transformers import HfArgumentParser

from tokenizers.config import Config


@dataclass
class EvaluationConfig(Config):
    num_samples: int = field(
        default=None,
        metadata={
            "help": "Number of samples to evaluate on, pass -1 to evaluate on all samples"
        },
    )
    load_existing: bool = field(
        default=True,
        metadata={"help": "Whether to load existing data configs from lm_eval library"},
    )


def load_config(config_class: Config):
    """Load configuration from file or command line arguments."""
    if len(sys.argv) > 1 and (
        sys.argv[1].endswith(".yaml")
        or sys.argv[1].endswith(".yml")
        or sys.argv[1].endswith(".json")
    ):
        # Load from YAML/JSON config
        config = config_class.from_file(sys.argv[1])
        # Remove the config file from args so HfArgumentParser doesn't see it
        sys.argv.pop(1)
    else:
        # Fall back to argument parsing
        parser = HfArgumentParser((config_class,))
        (config,) = parser.parse_args_into_dataclasses()
    return config
