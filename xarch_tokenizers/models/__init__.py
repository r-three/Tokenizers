"""Model loading utilities.

This module provides functions to load tokenizers and language models from
transformers library or vLLM. It handles environment-specific paths and configurations,
particularly for Vector cluster.
"""

from typing import Literal, Optional

import torch
import transformer_lens
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
from xarch_tokenizers.utils.system import VECTOR_HF_MAPPING, Hosts, get_host


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a tokenizer for the specified model.
    If tokenizer available locally, uses the local path"""
    kwargs = dict()
    if "aya" in model_name.lower():
        kwargs["use_fast"] = True

    model_path = model_name
    if get_host() == Hosts.vector and model_name in VECTOR_HF_MAPPING:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def load_model(
    model_name,
    dtype: Optional[Literal["bfloat16", "float16", "float32"]] = "bfloat16",
    device_map: str = "auto",
):
    """Load a causal language model from Hugging Face."""
    kwargs = dict()
    if device_map:
        kwargs["device_map"] = device_map
    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        kwargs["local_files_only"] = True
    if dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model


def load_hooked_model(
    model_name,
    dtype: Optional[Literal["bfloat16", "float16", "float32"]] = "bfloat16",
    device_map: str = "auto",
):
    """ """
    kwargs = dict(device_map=device_map)
    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        kwargs["local_files_only"] = True
    return transformer_lens.HookedTransformer.from_pretrained(
        model_name, dtype=dtype, **kwargs
    )


def load_vllm_model(
    model_name,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,  # Options: "awq", "sq", None
    dtype: Optional[Literal["bfloat16", "float16", "float32"]] = "bfloat16",
):
    """Load a language model using vLLM for optimized inference."""
    kwargs = dict(trust_remote_code=True)
    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        kwargs["trust_remote_code"] = True
        kwargs["load_format"] = "hf"

    if dtype == "bfloat16":
        kwargs["dtype"] = "bfloat16"

    model = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,
        **kwargs,
    )
    return model
