import json
import os
import math
from dataclasses import dataclass, field

import datasets
from transformers import AutoTokenizer
from tqdm import tqdm

from xarch_tokenizers.config import Config
from xarch_tokenizers.experiment_config import load_config
from xarch_tokenizers.logging.logger import setup_logger
from xarch_tokenizers.logging.plot_utils import setup_styles
from xarch_tokenizers.models import load_tokenizer
from xarch_tokenizers.utils.system import VECTOR_HF_MAPPING

from compare_tokenizers import setup_tokenizer_environment


@dataclass
class TokenizerCompressionConfig(Config):
    tokenizer_names: list[str] = field(
        default_factory=list,
        metadata={"help": "Pass the name of the tokenizer to benchmark"}
    )
    dataset_path: str | None = "allenai/c4"
    dataset_name: str | None = "en"
    dataset_split: str = "validation"
    dataset_streaming: bool = False


def compute_compression(tok, ds, field="text"):
    b_in = 0
    t_out = 0
    for example in tqdm(ds, leave=False):
        text = example[field]
        b_in += len(text.encode("utf-8"))
        t_out += len(tok.encode(text))
    return b_in, t_out


if __name__ == "__main__":
    config = load_config(TokenizerCompressionConfig)
    models, tokenizers, logger, device = setup_tokenizer_environment(config)
    ds = datasets.load_dataset(
        config.dataset_path,
        name=config.dataset_name,
        split=config.dataset_split,
        streaming=config.dataset_streaming
    )

    results = []
    for name, tokenizer in tqdm(tokenizers.items()):
        b_in, out = compute_compression(tokenizer, ds)
        vocab_bits = math.ceil(math.log2(tokenizer.vocab_size + 1))
        b_out = out * vocab_bits / 8
        results.append(f"For {name}, vocab size={tokenizer.vocab_size}, vocab bits={vocab_bits}, bytes in={b_in}, bytes out={b_out}, tokens out={out}, compression_rate={b_in/b_out}")
    print("\n".join(results))
