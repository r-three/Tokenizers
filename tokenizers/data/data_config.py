"""Dataset statistics and configurations for both vision and language tasks."""

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class Domain(Enum):
    LANGUAGE = "language"
    VISION = "vision"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    QUESTION_ANSWERING = "question_answering"
    SEQUENCE_PAIR = "sequence_pair"
    NATURAL_LANGUAGE_INFERENCE = "natural_language_inference"
    SEQUENCE_LABELING = "sequence_labeling"
    REGRESSION = "regression"
    SUMMARIZATION = "summarization"


@dataclass
class BaseConfig:
    samples: Union[int, Dict[str, int]]


# Vision Registry
@dataclass
class VisionConfig(BaseConfig):
    classes: int
    channels: int
    resolution: int
    mean: np.ndarray
    std: np.ndarray
    can_cache: bool
    torch_dataset: Optional[Union[str, type]] = None
    hf_path: str = None
    hf_config: Optional[str] = None
    task_type: TaskType = TaskType.CLASSIFICATION


@dataclass
class InstructionFormat:
    instruction: str = "You are a helpful assistant"
    user_format: str = 'Answer the following question: {input}\n\nProvide a step-by-step solution. Put your final answer after "####"'
    user_input: str = "input"
    assistant_output: str = "target"
    answer_len: Literal["short", "medium", "long"] = "short"
    system_instruction: Optional[str] = "You are a helpful assistant."

    def format_example(self, example):
        # TODO
        raise NotImplementedError
        return self.instruction.format(**example)

    def update(self, **kwargs):
        if "instruction" in kwargs:
            self.instruction = kwargs["instruction"]
        if "user_format" in kwargs:
            self.user_format = kwargs["user_format"]
        if "user_input" in kwargs:
            self.user_input = kwargs["user_input"]
        if "assistant_output" in kwargs:
            self.assistant_output = kwargs["assistant_output"]
        if "answer_len" in kwargs:
            self.answer_len = kwargs["answer_len"]
        if "system_instruction" in kwargs:
            self.system_instruction = kwargs["system_instruction"]


# Language Registry
@dataclass
class LanguageConfig(BaseConfig):
    classes: Optional[int]
    task_type: TaskType
    max_seq_length: int
    hf_path: str
    max_gen_seq_length: int = None
    # TODO: template format
    # Field mapping attributes
    input_field: str = "text"  # For classification
    label_field: str = "label"  # For classification
    # QA fields
    context_field: str = "context"
    additional_fields: Optional[List[str]] = None
    hf_config: Optional[str] = None
    splits: Dict[str, str] = None
    vocab_size: Optional[int] = None
    metrics: List[str] = field(default_factory=list)  # Add this field
    trust_remote_code: bool = False
    language: str = "en"
    languages: List[str] = None
    is_translation: bool = False
    source_language: Optional[str] = None

    def is_generation(self):
        return self.task_type == TaskType.GENERATION

    def __post_init__(self):
        if self.max_gen_seq_length is None:
            self.max_gen_seq_length = self.max_seq_length
        if self.languages is None:
            self.languages = [self.language]


@dataclass
class TaskConfig:
    """Configuration for a dataset including optional subset specifications."""

    name: str
    subset: Optional[str] = None
    split: Optional[str] = None
    num_samples: Optional[int] = None

    task_type: TaskType = None
    max_seq_length: int = None
    hf_path: str = None
    max_gen_seq_length: int = None
    # TODO: template format
    # Field mapping attributes
    input_field: str = "text"  # For classification
    label_field: str = "label"  # For classification
    # QA fields
    context_field: str = "context"
    additional_fields: Optional[List[str]] = None
    hf_config: Optional[str] = None
    splits: Dict[str, str] = None
    vocab_size: Optional[int] = None
    metrics: List[str] = field(default_factory=list)  # Add this field
    trust_remote_code: bool = False
    language: str = "en"
    languages: List[str] = None
    is_translation: bool = False
    source_language: Optional[str] = None
    instruction_config: Optional[InstructionFormat] = None
    # TODO: find a better way
    override_config: Optional[Dict] = None
    lm_eval_alias: Optional[str] = None

    def __post_init__(self):
        path = Path(__file__).parent.joinpath("datasets.yaml")
        with open(path, "r", encoding="utf-8") as f:
            dct = yaml.load(f, Loader=yaml.Loader)
        if self.name not in dct:
            logger.warning("%s not in the predefined dict", self.name)
            dct = {self.name: {}}
        dct = dct[self.name]
        self.task_type = TaskType(dct.get("task_type", "generation"))
        self.max_gen_seq_length = dct.get("max_gen_seq_length")
        self.max_seq_length = dct.get("max_seq_length")
        self.hf_path = dct.get("hf_path")
        self.input_field = dct.get("input_field")
        self.label_field = dct.get("label_field")
        self.context_field = dct.get("context_field")
        self.additional_fields = dct.get("additional_fields")
        self.hf_config = dct.get("hf_config")
        self.splits = dct.get("splits")
        self.vocab_size = dct.get("vocab_size")
        self.metrics = dct.get("metrics")
        self.trust_remote_code = dct.get("trust_remote_code")
        self.language = dct.get("language")
        self.languages = dct.get("languages")
        self.is_translation = dct.get("is_translation")
        self.source_language = dct.get("source_language")
        lm_eval_alias = self.name
        if "/" in lm_eval_alias:
            lm_eval_alias = lm_eval_alias.split("/")[-1]
        if self.subset:
            lm_eval_alias += f"_{self.subset}"
        self.lm_eval_alias = lm_eval_alias
        if self.instruction_config is None and "instruction_format" in dct:
            self.instruction_config = InstructionFormat(**dct["instruction_format"])

    def __str__(self) -> str:
        """String representation for logging."""
        parts = [self.name]
        if self.subset:
            parts.append(f"subset={self.subset}")
        if self.split:
            parts.append(f"split={self.split}")
        if self.num_samples:
            parts.append(f"samples={self.num_samples}")
        return ":".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "subset": self.subset,
            "split": self.split,
            "num_samples": self.num_samples,
        }

    @classmethod
    def _get_default_instruction_format(cls, name):
        path = Path(__file__).parent.joinpath("datasets.yaml")
        with open(path, "r", encoding="utf-8") as f:
            dct = yaml.load(f, Loader=yaml.Loader)
        instruction_config = InstructionFormat()
        if name not in dct:
            logger.warning("%s not in the predefined dict", name)
            return instruction_config
        dct = dct[name]
        if "instruction_format" in dct:
            instruction_config = InstructionFormat(**dct["instruction_format"])
        return instruction_config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        """Create from dictionary."""
        kwargs = dict(override_config=data.get("override_config"))
        if "instruction_format" in data:
            default_conf = cls._get_default_instruction_format(data["name"])
            default_conf.update(**data["instruction_format"])
            kwargs["instruction_config"] = default_conf

        return cls(
            name=data["name"],
            subset=data.get("subset"),
            split=data.get("split"),
            num_samples=data.get("num_samples"),
            **kwargs,
        )

    @classmethod
    def from_string(cls, dataset_str: str) -> "TaskConfig":
        """Parse dataset from string format like 'name:subset=x:split=y'."""
        parts = dataset_str.split(":")
        name = parts[0]

        # Initialize with defaults
        subset = None
        split = None
        num_samples = None

        ## TODO: doesnt have the instructionformat yet
        # Parse remaining parts for key=value pairs
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                if key == "subset":
                    subset = value
                elif key == "split":
                    split = value
                elif key == "samples":
                    num_samples = int(value)
        return cls(name=name, subset=subset, split=split, num_samples=num_samples)
