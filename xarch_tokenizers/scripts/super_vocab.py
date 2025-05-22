"""Create a shared vocabulary to keep models consistent.

Load tokenmonster with tokenmonster/...
Load tiktoken with tiktoken/${model-name}
"""

import argparse
import collections
import functools
import json
import operator as op
import os
import re
import logging

import tokenizers
import transformers

from xarch_tokenizers.models import load_tokenizer as hf_load_tokenizer
from xarch_tokenizers.utils import system


Vocab = dict[str, list[int]]

parser = argparse.ArgumentParser(description="Create a Super Vocab of all vocabs.")
parser.add_argument("--tokenizers", required=True, nargs="+")
parser.add_argument("--output_dir", default="vocabs")

logging.basicConfig(level=logging.INFO)


class Tokenizer:
    """Tokenizer wrapper that unifies interface."""

    def __init__(self, name: str, tokenizer):
        self._name = name
        self.tokenizer = tokenizer

    @property
    def name(self):
        return self._name

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_token(self, i):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError

    @classmethod
    def load(cls, name):
        if name.startswith("tokenmonster"):
            return TokenMonsterTokenizer.load(name)
        if name.startswith("tiktoken"):
            return TikTokenTokenizer.load(name)
        if "tekken" in name:
            return MistralTokenizer.load(name)
        return HFTokenizer.load(name)


class HFTokenizer(Tokenizer):

    def info(self):
        return {
            "backend": "huggingface",
            "name": self.name
        }

    def get_vocab_size(self):
        if "byt5" in self.name:
            return self.tokenizer.vocab_size
        return self.tokenizer.get_vocab_size()

    def get_token(self, i):
        if "byt5" in self.name:
            token = self.tokenizer.convert_ids_to_tokens(i)
            # We are a special value.
            if len(token) > 1:
                return token
            as_int = ord(token)
            as_bytes = bytes([as_int])
            try:
                return as_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return as_int # as_bytes
        t = self.tokenizer.id_to_token(i)
        if isinstance(self.tokenizer.model, tokenizers.models.WordPiece):
            # If it is not a continuation character, then it is the start of a word. Other tokenizers start the word with a subword token that has a space to start.
            if not t.startswith("##"):
                return f" {t}"
            return re.sub(r"##([^#])", r"\1", t)
        if isinstance(self.tokenizer.model, tokenizers.models.Unigram) or any(n in self.name for n in ("gemma", "Phi-3", "Mistral-7B-Instruct-v0.3")):
            # Replace whitespace handling with actual whitespace.
            return t.replace("▁", " ")
        # BPE models.
        return real_unicode(t)

    @classmethod
    def load(cls, name):
        if system.get_host() == system.Hosts.vector:
            name = system.VECTOR_HF_MAPPING.get(name, name)
        try:
            tok = hf_load_tokenizer(name)
        except:
            tok = transformers.AutoTokenizer.from_pretrained(name)
        if hasattr(tok, "_tokenizer"):
            tok = tok._tokenizer
        return cls(name, tok)



class TikTokenTokenizer(Tokenizer):

    def info(self):
        return {
            "backend": "tiktoken",
            "name": self.name.split("/")[1]
        }

    def get_token(self, i):
        try:
            b = self.tokenizer.decode_single_token_bytes(i)
        except KeyError:
            return f"~~~~~undefined {i}~~~~~~"
        return b.decode("latin-1")

    def get_vocab_size(self):
        return self.tokenizer.n_vocab

    @classmethod
    def load(cls, name):
        import tiktoken
        tok = tiktoken.encoding_for_model(name.split("/")[1])
        return cls(name, tok)


class TokenMonsterTokenizer(Tokenizer):

    def info(self):
        return {
            "backend": "tokenmonster",
            "name": self.name.split("/")[1]
        }

    def get_token(self, i):
        return self.tokenizer.id_to_token(i)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    @classmethod
    def load(cls, name):
        import tokenmonster
        tok = tokenmonster.load(name.split("/")[1])
        return cls(name, tok)


class MistralTokenizer(Tokenizer):

    def info(self):
        return {
            "backend": "tekken",
            "name": "tekken"
        }

    def get_token(self, i):
        return self.tokenizer.id_to_piece(i)

    def get_vocab_size(self):
        return self.tokenizer.n_words

    @classmethod
    def load(cls, name):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        tok = MistralTokenizer.v3(is_tekken=True)
        tok = tok.instruct_tokenizer.tokenizer
        return cls(name, tok)


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


BYTES_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTES = {v: k for k, v in BYTES_TO_UNICODE.items()}

def real_unicode(word: str) -> str:
    bytes_word = []
    for c in word:
        if c != " ":
            if c in UNICODE_TO_BYTES:
                c = chr(UNICODE_TO_BYTES[c])
        bytes_word.append(c.encode("utf-8"))
    return b"".join(bytes_word).decode("utf-8")


def make_vocab(tok: Tokenizer) -> Vocab:
    # Track multiple values because tekken and tokenmonster are weird
    vocab = collections.defaultdict(list)
    for i in range(tok.get_vocab_size()):
        vocab[tok.get_token(i)].append(i)
    if len(vocab) != tok.get_vocab_size():
        logging.error("Built vocab size (%d) does not match declared vocab size (%d) for %s", len(vocab), tok.get_vocab_size(), tok.name)
    return vocab


def to_bytes(s: bytes | str | int) -> bytes:
    if isinstance(s, str):
        s = s.encode("utf-8")
    if isinstance(s, int):
        s = bytes([s])
    # Now s is def bytes
    return s


def join_vocabs(vocabs: dict[str, Vocab]) -> Vocab:
    joint = functools.reduce(op.or_, [v.keys() for v in vocabs.values()])
    return {s: i for i, s in enumerate(sorted(joint, key=to_bytes))}


def align_to_super(super_vocab, model_vocab):
    alignment = {}
    for piece, idxs in model_vocab.items():
        super_idx = super_vocab[piece]
        for i in idxs:
            alignment[i] = super_idx
    return alignment


def main(args):
    logging.info("Loading Tokenizers.")
    tokenizers: dict[str, tokenizers.Tokenizer] = {
        name: Tokenizer.load(name) for name in args.tokenizers
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
            json.dump(align_to_super(super_vocab, vocab), wf)

        with open(d := os.path.join(args.output_dir, f"{name.replace('/','--')}_vocab.json"), "w") as wf:
            logging.info("Saving vocab for %s to '%s'", name, d)
            json.dump(vocab, wf)

        with open(d := os.path.join(args.output_dir, f"{name.replace('/', '--')}_info.json"), "w") as wf:
            logging.info("Saving tokenizer info for %s to '%s'", name, d)
            json.dump(tokenizers[name].info(), wf)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
