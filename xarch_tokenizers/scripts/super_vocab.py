"""Create a shared vocabulary to keep models consistent.

Load tokenmonster with tokenmonster/...
"""

import argparse
import functools
import json
import operator as op
import os
import logging

import tokenizers

from xarch_tokenizers.models import load_tokenizer as hf_load_tokenizer
from xarch_tokenizers.utils import system


Vocab = dict[str, int]

parser = argparse.ArgumentParser(description="Create a Super Vocab of all vocabs.")
parser.add_argument("--tokenizers", required=True, nargs="+")
parser.add_argument("--output_dir", default="vocabs")

logging.basicConfig(level=logging.INFO)


def load_tokenizer(name: str) -> tokenizers.Tokenizer:
    if name.startswith("tokenmonster"):
        import tokenmonster
        return tokenmonster.load(name.split("/")[1])
    if system.get_host() == system.Hosts.vector:
        name = system.VECTOR_HF_MAPPING.get(name, name)
    hf_tok = hf_load_tokenizer(name)
    # Normalize to tokenizers if possible.
    if hasattr(hf_tok, "_tokenizer"):
        return hf_tok._tokenizer
    return hf_tok


def make_vocab(tok: tokenizers.Tokenizer) -> Vocab:
    try:
        vocab_size = tok.get_vocab_size()
    # Handle tokenmonster and transformers.
    except AttributeError:
        vocab_size = tok.vocab_size
    return {tok.id_to_token(i): i for i in range(vocab_size)}


def join_vocabs(vocabs: dict[str, Vocab]) -> Vocab:
    joint = functools.reduce(op.or_, [v.keys() for v in vocabs.values()])
    return {s: i for i, s in enumerate(sorted(joint))}


def main(args):
    logging.info("Loading Tokenizers.")
    tokenizers: dict[str, tokenizers.Tokenizer] = {
        name: load_tokenizer(name) for name in args.tokenizers
    }

    logging.info("Extracting Vocabularies.")
    tokenizer_vocabs: dict[str, Vocab] = {
        name: make_vocab(tokenizer) for name, tokenizer in tokenizers.items()
    }

    logging.info("Creating super set vocabulary")
    super_vocab = join_vocabs(tokenizer_vocabs)
    logging.info("Super set vocabulary has %d items", len(super_vocab))

    # Save the super vocab
    os.makedirs(args.output_dir, exist_ok=True)
    with open(d := os.path.join(args.output_dir, "super_vocab.json"), "w") as wf:
        logging.info("Saving super set vocab to '%s'", d)
        json.dump(super_vocab, wf)
    # Save each vocab mapping
    for name, vocab in tokenizer_vocabs.items():
        # Replace / with -- like the huggingface caching code does.
        with open(d := os.path.join(args.output_dir, f"{name.replace('/', '--')}_super_mapping.json"), "w") as wf:
            logging.info("Saving vocab mapping for %s to '%s'", name, d)
            json.dump({i: super_vocab[s] for s, i in vocab.items()}, wf)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
