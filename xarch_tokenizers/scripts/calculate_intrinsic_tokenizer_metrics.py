import sys
import os
import collections
import functools
import operator as op
import json
import re
import argparse

# Add the current directory (project_root) to sys.path
sys.path.append(os.path.abspath("."))

from datasets import load_dataset
from xarch_tokenizers.utils.word_tokenizers import TibetanTokenizer, load_tokenizer_assignments, load_word_tokenizer

import enum
import logging
import tokenizers
from typing import Dict, Generator, List, Optional, Tuple, Union
from typing_extensions import TypeAlias
import random
import transformers
from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from xarch_tokenizers.models import load_tokenizer as hf_load_tokenizer
from xarch_tokenizers.utils import system



Vocab = dict[str, list[int]]

LOG = logging.getLogger(__name__)

random.seed(42)  # Set the seed to a fixed value

ALIGNED_BOS = "~SPECIAL~ALIGNED~BOS~SYMBOL~"

TOKENIZER_NAMES =  {
    "Comma": "common-pile/comma-v0.1",
    "Llama-3.2" : "meta-llama/Llama-3.2-1B",
    "Phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "GPT-2": "gpt2",
    "GPT-4o": "tiktoken/gpt-4o",
    "BLOOM": "bigscience/bloom",
    "XGLM" : "facebook/xglm-564M",
    "Tekken": "mistralai/tekken",
    "ByT5": "google/byt5-small",
    "mBERT": "google-bert/bert-base-multilingual-cased",
    "Qwen-3": "Qwen/Qwen3-8B",
    "TokenMonster": "tokenmonster/english-32000-consistent-v1",
    "Gemma-2": "google/gemma-2-2b",
    "Aya": "CohereLabs/aya-expanse-8b"
}

TOKENIZER_TYPES =  {
    "Comma": "BPE",
    "mBERT": "WordPiece",
    "XGLM" : "SentencePiece_Unigram",
    "Gemma-2": "BPE", # BPE in HF but Originally SentencePiece_Unigram
    "Phi-3": "SentencePiece_BPE",
    "TokenMonster": "",
    "ByT5": "byte-level",
    "BLOOM": "BPE",
    "GPT-2": "BPE",
    "GPT-4o": "BPE",
    "Tekken": "BPE", #Use this link to call it: https://docs.mistral.ai/guides/tokenization/
    "Llama-3.2" : "BPE",
    "Qwen-3": "BPE",
    "Aya": "SentencePiece"
}

TOKENIZER_N_SPECIAL_TOKENS_PER_WORD =  {
    "Comma": 0, # Example: ['F', 'amil', 'ies']
    "mBERT": 2, # Example: ['[CLS]', 'Families', '[SEP]']
    "XGLM" : 1, # Example: ['▁Familie', 's', '</s>']
    "Gemma-2": 1, # Example: ['<bos>', 'Families']
    "Phi-3": 0, # Example: ['▁Famil', 'ies']
    "TokenMonster": 0, # Example: [np.uint16(586), np.uint16(17496)]
    "ByT5": 1, # Example: [73, 100, 112, 108, 111, 108, 104, 118, 1]
    "BLOOM": 0, # Example: ['Famil', 'ies']
    "GPT-2": 0, # Example: ['F', 'am', 'ilies']
    "GPT-4o": 0, # Example: [139342]
    "Tekken": 0, # Example: [109925, 1564]
    "Llama-3.2" : 1, # Example: ['<|begin_of_text|>', 'F', 'amilies']
    "Qwen-3": 0, # Example: ['F', 'amilies']
    "Aya": 1 # Example: ['<BOS_TOKEN>', 'Families']

}

LANGUAGE_KEYS = {'sentence_eng_Latn': "eng_Latn", #english
                 'sentence_zho_Hans': "zho_Hani", #chinese
                 'sentence_tur_Latn': "tur_Latn", #turkish
                 'sentence_pes_Arab': "fas_Arab", #persian
                 'sentence_ita_Latn': "ita_Latn", #Italian
                 }



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
    
    def get_vocab(self):
        raise NotImplementedError

    def get_token(self, i):
        raise NotImplementedError

    def get_bos_str(self):
        raise NotImplementedError
    
    def info(self):
        raise NotImplementedError
    
    def tokenize(self, input_text):
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
    def __init__(self, *args, bos_str: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bos_str = bos_str

    def info(self):
        return {"data": {"tokenizer": {
            "name": "huggingface",
            "path": self.name
        }}}

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
        if t == self.bos_str:
            return ALIGNED_BOS
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

    def get_vocab(self): #TODO
        # Track multiple values because tekken and tokenmonster are weird
        vocab = collections.defaultdict(list)
        for i in range(self.get_vocab_size()):
            vocab[to_bytes(self.get_token(i))].append(i)
        if len(vocab) != self.get_vocab_size():
            logging.error("Built vocab size (%d) does not match declared vocab size (%d) for %s", len(vocab), self.get_vocab_size(), self.info()["data"]["tokenizer"]["name"])
        return vocab

    def tokenize(self, input_text): #TODO
        encoded_output = self.tokenizer.encode(input_text)
        if hasattr(encoded_output, "tokens"):  # Case: tokenizers.Tokenizer object
            return encoded_output.tokens
        elif isinstance(encoded_output, list):  # Case: already a list of strings
            return encoded_output
        else:
            raise ValueError("Unexpected return type from tokenizer.encode()")

    @classmethod
    def load(cls, name):
        if system.get_host() == system.Hosts.vector:
            name = system.VECTOR_HF_MAPPING.get(name, name)
        try:
            tok = hf_load_tokenizer(name)
        except:
            tok = transformers.AutoTokenizer.from_pretrained(name)
        sts = getattr(tok, "special_tokens_map", {})
        if "bert" in name:
            bos_str = sts.get("cls_token")
        elif "t5" in name:
            bos_str = sts.get("pad_token")
        else:
            bos_str = sts.get("bos_token")
        if hasattr(tok, "_tokenizer"):
            tok = tok._tokenizer
        return cls(name, tok, bos_str=bos_str)

# Note, GPT4 and GPT4o don't have BOS
class TikTokenTokenizer(Tokenizer):

    def info(self):
        return {"data": {"tokenizer": {
            "name": "tiktoken",
            "path": self.name.split("/")[1]
        }}}

    def get_token(self, i):
        try:
            b = self.tokenizer.decode_single_token_bytes(i)
        except KeyError:
            return f"~~~~~undefined {i}~~~~~~"
        return b.decode("latin-1")

    def get_vocab_size(self):
        return self.tokenizer.n_vocab
    
    def get_vocab(self): #TODO
        # Track multiple values because tekken and tokenmonster are weird
        vocab = collections.defaultdict(list)
        for i in range(self.get_vocab_size()):
            vocab[to_bytes(self.get_token(i))].append(i)
        if len(vocab) != self.get_vocab_size():
            logging.error("Built vocab size (%d) does not match declared vocab size (%d) for %s", len(vocab), self.get_vocab_size(), self.info()["data"]["tokenizer"]["name"])
        return vocab

    def tokenize(self, input_text): #TODO
        return self.tokenizer.encode(input_text)
    
    @classmethod
    def load(cls, name):
        import tiktoken
        tok = tiktoken.encoding_for_model(name.split("/")[1])
        return cls(name, tok)

class TokenMonsterTokenizer(Tokenizer):

    def info(self):
        return {"data": {"tokenizer": {
            "name": "tokenmonster",
            "path": self.name.split("/")[1]
        }}}

    def get_token(self, i):
        return self.tokenizer.id_to_token(i)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def get_vocab(self): #TODO
        """Version that excludes ALL duplicate tokens, not just replacement chars"""
        vocab = collections.defaultdict(list)
        
        # Build full vocabulary first
        for i in range(self.get_vocab_size()):
            vocab[to_bytes(self.get_token(i))].append(i)
        
        # Find duplicates
        duplicates = {byte_seq: token_ids for byte_seq, token_ids in vocab.items() if len(token_ids) > 1}
        
        if duplicates:
            print(f"Found {len(duplicates)} duplicate byte sequences")
            for byte_seq, token_ids in duplicates.items():
                char_repr = byte_seq.decode('utf-8', errors='replace')
                print(f"  '{char_repr}': {len(token_ids)} duplicates")
        
        # Build clean vocab keeping only first occurrence of each duplicate
        clean_vocab = {}
        total_excluded = 0
        
        for byte_seq, token_ids in vocab.items():
            if len(token_ids) > 1:
                # Keep only the first token ID for duplicates
                clean_vocab[byte_seq] = [token_ids[0]]
                total_excluded += len(token_ids) - 1
            else:
                # Keep single tokens as-is
                clean_vocab[byte_seq] = token_ids
        
        print(f"Strict filtering: {len(clean_vocab)} unique tokens ({total_excluded} duplicates excluded) for {self.info()}")
        return clean_vocab

    def tokenize(self, input_text): #TODO
        return list(self.tokenizer.tokenize(input_text))   

    @classmethod
    def load(cls, name):
        import tokenmonster
        tok = tokenmonster.load(name.split("/")[1])
        return cls(name, tok)

class MistralTokenizer(Tokenizer):

    def info(self):
        return {"data": {"tokenizer": {
            "name": "tekken",
            "path": "tekken"
        }}}

    def get_token(self, i):
        if i == self.tokenizer.bos_id:
            return ALIGNED_BOS
        return self.tokenizer.id_to_piece(i)

    def get_vocab_size(self):
        return self.tokenizer.n_words
    
    def get_vocab(self): #TODO
        """Version that excludes ALL duplicate tokens, not just replacement chars"""
        vocab = collections.defaultdict(list)
        
        # Build full vocabulary first
        for i in range(self.get_vocab_size()):
            vocab[to_bytes(self.get_token(i))].append(i)
        
        # Find duplicates
        duplicates = {byte_seq: token_ids for byte_seq, token_ids in vocab.items() if len(token_ids) > 1}
        
        if duplicates:
            print(f"Found {len(duplicates)} duplicate byte sequences")
            for byte_seq, token_ids in duplicates.items():
                char_repr = byte_seq.decode('utf-8', errors='replace')
                print(f"  '{char_repr}': {len(token_ids)} duplicates")
        
        # Build clean vocab keeping only first occurrence of each duplicate
        clean_vocab = {}
        total_excluded = 0
        
        for byte_seq, token_ids in vocab.items():
            if len(token_ids) > 1:
                # Keep only the first token ID for duplicates
                clean_vocab[byte_seq] = [token_ids[0]]
                total_excluded += len(token_ids) - 1
            else:
                # Keep single tokens as-is
                clean_vocab[byte_seq] = token_ids
        
        print(f"Strict filtering: {len(clean_vocab)} unique tokens ({total_excluded} duplicates excluded) for {self.info()}")
        return clean_vocab

    def tokenize(self, input_text): #TODO
        return self.tokenizer.encode(input_text, False, False)

    @classmethod
    def load(cls, name):
        print(f"Loading MistralTokenizer for: {name}")
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            print("Import successful")
            
            tok = MistralTokenizer.v3(is_tekken=True)
            print("Tokenizer created")
            
            tok = tok.instruct_tokenizer.tokenizer
            print("Instruct tokenizer extracted")
            
            return cls(name, tok)
        except Exception as e:
            print(f"Error in MistralTokenizer.load: {e}")
            raise


def plot_tokenizer_vocab_overlap_symmetric(tokenizer_names):
    """
    Plots a heatmap showing the Jaccard Index (vocabulary overlap) between different tokenizers.

    Parameters:
    - tokenizer_names: Dict mapping display names to tokenizer paths.

    Returns:
    - A seaborn heatmap plot of the Jaccard vocabulary overlap.
    """
    # Load vocabularies for each tokenizer
    tokenizer_vocabularies = {}
    for name, tokenizer_dir in tokenizer_names.items():
        print("Retrieve the vocabulary for tokenizer: ", tokenizer_dir)
        tokenizer = Tokenizer.load(tokenizer_dir)
        vocab = tokenizer.get_vocab()
        tokenizer_vocabularies[name] = set(vocab)

    # Clean tokenizer display names (remove org prefix)
    clean_names = [name.split("/")[-1] for name in tokenizer_names]

    # Compute Jaccard ratio matrix
    ratio_matrix = []
    for name1 in tokenizer_names:
        row = []
        vocab1 = tokenizer_vocabularies[name1]
        for name2 in tokenizer_names:
            vocab2 = tokenizer_vocabularies[name2]
            intersection = vocab1.intersection(vocab2)
            union = vocab1.union(vocab2)
            ratio = len(intersection) / len(union)
            row.append(ratio)
        ratio_matrix.append(row)

    # Create DataFrame with clean names
    vocab_overlap_ratio_df = pd.DataFrame(ratio_matrix, index=clean_names, columns=clean_names)

    # Print high precision overlap matrix
    print("\n" + "="*80)
    print("TOKENIZER VOCABULARY OVERLAP MATRIX (4 decimal places)")
    print("="*80)

    
    # Find and print interesting statistics
    max_overlap = 0
    min_overlap = 1
    max_pair = None
    min_pair = None
    
    for i, name1 in enumerate(clean_names):
        for j, name2 in enumerate(clean_names):
            if i != j:  # Skip diagonal
                ratio = ratio_matrix[i][j]
                if ratio > max_overlap:
                    max_overlap = ratio
                    max_pair = (name1, name2)
                if ratio < min_overlap:
                    min_overlap = ratio
                    min_pair = (name1, name2)
    
    print(f"\nHIGHEST OVERLAP: {max_pair[0]} ↔ {max_pair[1]} = {max_overlap:.4f}")
    print(f"LOWEST OVERLAP:  {min_pair[0]} ↔ {min_pair[1]} = {min_overlap:.4f}")
    
    # Calculate average overlap (excluding diagonal)
    total_overlap = sum(ratio_matrix[i][j] for i in range(len(clean_names)) 
                       for j in range(len(clean_names)) if i != j)
    avg_overlap = total_overlap / (len(clean_names) * (len(clean_names) - 1))
    print(f"AVERAGE OVERLAP: {avg_overlap:.4f}")
    print("="*80)

    # Calculate dynamic figure size based on label length
    label_len = max(len(name) for name in clean_names)
    fig_size = max(1.5 * len(clean_names), 10)

    plt.figure(figsize=(fig_size, fig_size))

    # Set larger font scale
    sns.set(font_scale=1.2)

    # Plot heatmap with 4 decimal places
    sns.heatmap(
        vocab_overlap_ratio_df,
        annot=True,
        fmt=".4f",  # Changed to 4 decimal places
        cmap="YlGnBu",
        square=True,
        cbar_kws={"label": "Jaccard Ratio"},
        annot_kws={"size": 14},  # Slightly smaller font for 4 decimals
        linewidths=0.3,
        linecolor='gray'
    )

    # Rotate x-axis labels and make sure they are fully shown
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Add axis labels and title with padding
    plt.title("Symmetric Tokenizer Vocabulary Overlap (Jaccard Index)", fontsize=18, pad=30)
    plt.xlabel("Tokenizer B", fontsize=14, labelpad=10)
    plt.ylabel("Tokenizer A", fontsize=14, labelpad=10)

    # Adjust layout manually to ensure nothing is cut off
    plt.subplots_adjust(left=0.2, bottom=0.25, top=0.9, right=0.95)

    plt.savefig("symmetric_vocab_overlap.png")
    
    return vocab_overlap_ratio_df

def plot_tokenizer_vocab_overlap_asymmetric(tokenizer_names):
    """
    Plots a heatmap showing asymmetric vocabulary overlap between different tokenizers.
    For row i and column j: shows |V_i ∩ V_j| / |V_j| 
    (how much of tokenizer j's vocabulary is covered by tokenizer i)

    Parameters:
    - tokenizer_names: Dict mapping display names to tokenizer paths.

    Returns:
    - A seaborn heatmap plot of the asymmetric vocabulary overlap.
    """
    # Load vocabularies for each tokenizer
    tokenizer_vocabularies = {}
    for name, tokenizer_dir in tokenizer_names.items():
        print("Retrieve the vocabulary for tokenizer: ", tokenizer_dir)
        tokenizer = Tokenizer.load(tokenizer_dir)
        vocab = tokenizer.get_vocab()
        tokenizer_vocabularies[name] = set(vocab)

    # Clean tokenizer display names (remove org prefix)
    clean_names = [name.split("/")[-1] for name in tokenizer_names]

    # Compute asymmetric overlap ratio matrix
    # ratio_matrix[i][j] = |V_i ∩ V_j| / |V_j|
    ratio_matrix = []
    for name1 in tokenizer_names:
        row = []
        vocab1 = tokenizer_vocabularies[name1]
        for name2 in tokenizer_names:
            vocab2 = tokenizer_vocabularies[name2]
            intersection = vocab1.intersection(vocab2)
            # Asymmetric: intersection divided by vocab2 size (column tokenizer)
            ratio = len(intersection) / len(vocab2) if len(vocab2) > 0 else 0
            row.append(ratio)
        ratio_matrix.append(row)

    # Create DataFrame with clean names
    vocab_overlap_ratio_df = pd.DataFrame(ratio_matrix, index=clean_names, columns=clean_names)
    
    # Find and print interesting statistics
    max_coverage = 0
    min_coverage = 1
    max_pair = None
    min_pair = None
    
    for i, name1 in enumerate(clean_names):
        for j, name2 in enumerate(clean_names):
            if i != j:  # Skip diagonal
                ratio = ratio_matrix[i][j]
                if ratio > max_coverage:
                    max_coverage = ratio
                    max_pair = (name1, name2)
                if ratio < min_coverage:
                    min_coverage = ratio
                    min_pair = (name1, name2)
    
    print(f"\nHIGHEST COVERAGE: {max_pair[0]} covers {max_coverage:.4f} ({max_coverage*100:.2f}%) of {max_pair[1]}")
    print(f"LOWEST COVERAGE:  {min_pair[0]} covers {min_coverage:.4f} ({min_coverage*100:.2f}%) of {min_pair[1]}")
    
    # Calculate average coverage (excluding diagonal)
    total_coverage = sum(ratio_matrix[i][j] for i in range(len(clean_names)) 
                        for j in range(len(clean_names)) if i != j)
    avg_coverage = total_coverage / (len(clean_names) * (len(clean_names) - 1))
    print(f"AVERAGE COVERAGE: {avg_coverage:.4f} ({avg_coverage*100:.2f}%)")
    
    # Print interpretation guide
    print(f"\nINTERPRETATION:")
    print(f"- Diagonal = 1.0000 (100% self-coverage)")
    print(f"- Row → Column: How much of column tokenizer is covered by row tokenizer")
    print(f"- High values = Row tokenizer is a superset of column tokenizer")
    print(f"- Low values = Row tokenizer covers little of column tokenizer")
    print("="*90)

    # Calculate dynamic figure size based on label length
    label_len = max(len(name) for name in clean_names)
    fig_size = max(1.5 * len(clean_names), 10)

    plt.figure(figsize=(fig_size, fig_size))

    # Set larger font scale
    sns.set(font_scale=1.2)

    # Plot heatmap with 4 decimal places
    sns.heatmap(
        vocab_overlap_ratio_df,
        annot=True,
        fmt=".4f",  # 4 decimal places
        cmap="Reds",  # Different colormap to distinguish from symmetric version
        square=True,
        cbar_kws={"label": "Coverage Ratio |V_row ∩ V_col| / |V_col|"},
        annot_kws={"size": 14},  # Slightly smaller font for 4 decimals
        linewidths=0.3,
        linecolor='white',
        vmin=0,  # Ensure scale starts at 0
        vmax=1   # Ensure scale ends at 1
    )

    # Rotate x-axis labels and make sure they are fully shown
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Add axis labels and title with padding
    plt.title("Asymmetric Tokenizer Vocabulary Coverage\n(Row covers % of Column)", fontsize=16, pad=30)
    plt.xlabel("Column Tokenizer (Target)", fontsize=14, labelpad=10)
    plt.ylabel("Row Tokenizer (Source)", fontsize=14, labelpad=10)

    # Add text annotation explaining the asymmetry
    plt.figtext(0.5, 0.02, 
                "Entry (i,j) shows what fraction of tokenizer j's vocabulary is covered by tokenizer i",
                ha='center', fontsize=10, style='italic', color='gray')

    # Adjust layout manually to ensure nothing is cut off
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.85, right=0.95)

    plt.savefig("asymmetric_vocab_overlap.png")
    
    return vocab_overlap_ratio_df

def list_tokenizer_vocab_sizes(tokenizer_names):
    """
    Prints the vocabulary size of each tokenizer from the given list.

    Parameters:
    - tokenizer_names: List of Hugging Face tokenizer model names as strings.
    """

    vocab_sizes = {}
    for tokenizer_name, tokenizer_dir in tokenizer_names.items():
      try:
        tokenizer = Tokenizer.load(tokenizer_dir)
        vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
      except Exception as e:
        print(f"Error loading tokenizer {tokenizer_name}: {e}")
        print("-" * 40)
    return vocab_sizes

def compute_subword_fertility(
    tokenizer_names: dict,
    language_keys: dict,
    sample_size: int = 10000,
    dataset_name: str = "Muennighoff/flores200",
    dataset_config: str = "all",
    dataset_split: str = "dev"
):
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, trust_remote_code=True)
    
    # Filter to keep only examples with all selected languages
    examples = [ex for ex in dataset if all(lang in ex and ex[lang].strip() for lang in language_keys.keys())]
    
    sampled = random.sample(examples, min(sample_size, len(examples)))
    print(f"Sampled examples: {len(sampled)}")

    # Prepare texts per language
    language_texts = {lang: [ex[lang] for ex in sampled] for lang in language_keys.keys()}

    # Initialize result storage
    all_results = {}

    for tokenizer_name, tokenizer_dir in tokenizer_names.items():
        print(f"Processing tokenizer: {tokenizer_name}")
        tokenizer = Tokenizer.load(tokenizer_dir)

        # --- Fertility ---
        fertility_scores = {}
        for lang, texts in language_texts.items():
            print(f"Processing language: {language_keys[lang]}")
            cur_tokenizer = load_word_tokenizer(language_keys[lang])

            total_words = 0
            text_subwords = 0

            for text in texts:
                words = cur_tokenizer.word_tokenize(text)
                total_words += len(words)
                for word in words:
                    text_subwords += len(tokenizer.tokenize(word)) - TOKENIZER_N_SPECIAL_TOKENS_PER_WORD[tokenizer_name]
            

            fertility = text_subwords / total_words if total_words > 0 else 0
            fertility = round(fertility, 3)
            fertility_scores[lang.replace("sentence_", "")] = fertility

        # Store scores
        all_results[tokenizer_name] = {
            "fertility": fertility_scores
        }

    # Convert to DataFrame and save
    df = pd.DataFrame.from_dict(
        {model: metrics["fertility"] for model, metrics in all_results.items()},
        orient="index"
    )
    df.to_csv("fertility_results.csv", index_label="model_name")

    return all_results

def compute_parity(
    tokenizer_names: list,
    language_keys: list,
    sample_size: int = 10000,
    dataset_name: str = "Muennighoff/flores200",
    dataset_config: str = "all",
    dataset_split: str = "dev",
    ref_lang: str = "sentence_eng_Latn"
):
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)

    # Filter to keep only examples with all selected languages
    examples = [ex for ex in dataset if all(lang in ex and ex[lang].strip() for lang in language_keys.keys())]
    sampled = random.sample(examples, min(sample_size, len(examples)))

    # Prepare texts per language
    language_texts = {lang: [ex[lang] for ex in sampled] for lang in language_keys.keys()}

    # Initialize result storage
    all_results = {}

    for tokenizer_name, tokenizer_dir in tokenizer_names.items():
        print(f"Processing tokenizer: {tokenizer_name}")
        tokenizer = Tokenizer.load(tokenizer_dir)

        # --- Parity (relative to English) ---
        parity_scores = {}
        for lang in language_keys:
            if lang == ref_lang:
                continue
            ratios = []
            for t1, t2 in zip(language_texts[lang], language_texts[ref_lang]):

                len_t1 = len(tokenizer.tokenize(t1))
                len_t2 = len(tokenizer.tokenize(t2))
                if len_t2 > 0:
                    ratios.append(len_t1 / len_t2)
            parity = np.mean(ratios) if ratios else 0
            parity = round(parity, 3)
            parity_scores[lang.replace("sentence_", "")] = parity

        # Store scores
        all_results[tokenizer_name] = {
            "parity_to_eng": parity_scores
        }
    # Convert to DataFrame and save
    df = pd.DataFrame.from_dict(
        {model: metrics["parity_to_eng"] for model, metrics in all_results.items()},
        orient="index"
    )
    df.to_csv("parity_results.csv", index_label="model_name")


    return all_results

def compute_proportion_of_continued_words(
    tokenizer_names: dict,
    language_keys: dict,
    sample_size: int = 10000,
    dataset_name: str = "Muennighoff/flores200",
    dataset_config: str = "all",
    dataset_split: str = "dev",
    ref_lang: str = "sentence_eng_Latn"
):
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, trust_remote_code=True)

    # Filter to keep only examples with all selected languages
    examples = [ex for ex in dataset if all(lang in ex and ex[lang].strip() for lang in language_keys.keys())]
    sampled = random.sample(examples, min(sample_size, len(examples)))

    # Prepare texts per language
    language_texts = {lang: [ex[lang] for ex in sampled] for lang in language_keys.keys()}

    # Initialize result storage
    all_results = {}

    for tokenizer_name, tokenizer_dir in tokenizer_names.items():
        print(f"Processing tokenizer: {tokenizer_name}")
        tokenizer = Tokenizer.load(tokenizer_dir)

        # --- Proportion of Continued Words ---
        cont_word_scores = {}
        for lang, texts in language_texts.items():
            print(f"Evaluating {language_keys[lang]}")
            word_tokenizer = load_word_tokenizer(language_keys[lang])
            all_words = []
            split_word_count = 0

            for text in texts:
                words = word_tokenizer.word_tokenize(text)
                all_words.extend(words) #TODO: Make it more efficient
                for word in words:
                    # Encode word separately to avoid sentence-level effects
                    tokenized = tokenizer.tokenize(word)
                    if (len(tokenized) - TOKENIZER_N_SPECIAL_TOKENS_PER_WORD[tokenizer_name]) >= 2:
                        split_word_count += 1

            total_word_count = len(all_words)
            proportion = split_word_count / total_word_count if total_word_count > 0 else 0
            proportion = round(proportion, 3)
            cont_word_scores[lang.replace("sentence_", "")] = proportion

        # Store scores
        all_results[tokenizer_name] = {
            "proportion_of_continued_words": cont_word_scores
        }

    # Convert to DataFrame and save
    df = pd.DataFrame.from_dict(
        {model: metrics["proportion_of_continued_words"] for model, metrics in all_results.items()},
        orient="index"
    )
    df.to_csv("pcw_results.csv", index_label="model_name")

    return all_results

def plot_fertility_scores(results, dataset_name):
    tokenizers = list(results.keys())
    all_languages = [set(results[tokenizer]["fertility"].keys()) for tokenizer in tokenizers]
    common_languages = set.intersection(*all_languages)  # consistent across all tokenizers

    fertility_data = []

    for tokenizer in tokenizers:
        fertilities = [results[tokenizer]["fertility"][lang] for lang in common_languages]
        fertility_data.append(fertilities)

    fertility_data = np.array(fertility_data)

    num_tokenizers = len(tokenizers)
    num_languages = len(common_languages)
    group_width = 0.8
    bar_width = group_width / num_tokenizers
    x = np.arange(num_languages)

    # Plot Fertility
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, tokenizer in enumerate(tokenizers):
        ax.bar(x + i * bar_width - group_width/2 + bar_width/2, fertility_data[i], bar_width, label=tokenizer)

    ax.set_xlabel('Languages', fontsize=18)
    ax.set_ylabel('Fertility Scores', fontsize=18)
    ax.set_title(f'Fertility Scores by Tokenizer and Language\n({dataset_name})', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(common_languages, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=12)
    fig.subplots_adjust(right=0.8)  # make space on the right
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("fertility.png")
    
def plot_parity_scores(results, dataset_name):
    tokenizers = list(results.keys())
    for tokenizer in tokenizers:
      results[tokenizer]["parity_to_eng"]["eng_Latn"] = 1.0
    all_languages = [set(results[tokenizer]["parity_to_eng"].keys()) for tokenizer in tokenizers]
    common_languages = set.intersection(*all_languages)  # consistent across all tokenizers

    parity_data = []

    for tokenizer in tokenizers:
        parities = [results[tokenizer]["parity_to_eng"][lang] for lang in common_languages]
        parity_data.append(parities)

    parity_data = np.array(parity_data)

    num_tokenizers = len(tokenizers)
    num_languages = len(common_languages)
    group_width = 0.8
    bar_width = group_width / num_tokenizers
    x = np.arange(num_languages)

    # Plot Parity
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, tokenizer in enumerate(tokenizers):
        ax.bar(x + i * bar_width - group_width/2 + bar_width/2, parity_data[i], bar_width, label=tokenizer)

    ax.set_xlabel('Languages', fontsize=18)
    ax.set_ylabel('Parity Scores', fontsize=18)
    ax.set_title(f'Parity Scores by Tokenizer and Language\n({dataset_name})', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(common_languages, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=12)
    fig.subplots_adjust(right=0.8)  # make space on the right
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("parity.png")

def plot_pcw_scores(results, dataset_name):
    tokenizers = list(results.keys())
    all_languages = [set(results[tokenizer]["proportion_of_continued_words"].keys()) for tokenizer in tokenizers]
    common_languages = set.intersection(*all_languages)  # consistent across all tokenizers

    pcw_data = []

    for tokenizer in tokenizers:
        pcws = [results[tokenizer]["proportion_of_continued_words"][lang] for lang in common_languages]
        pcw_data.append(pcws)

    pcw_data = np.array(pcw_data)

    num_tokenizers = len(tokenizers)
    num_languages = len(common_languages)
    group_width = 0.8
    bar_width = group_width / num_tokenizers
    x = np.arange(num_languages)

    # Plot Parity
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, tokenizer in enumerate(tokenizers):
        ax.bar(x + i * bar_width - group_width/2 + bar_width/2, pcw_data[i], bar_width, label=tokenizer)

    ax.set_xlabel('Languages', fontsize=18)
    ax.set_ylabel('Proportion of Continued Words (PCW)', fontsize=18)
    ax.set_title(f'Proportion of Continued Words by Tokenizer and Language\n({dataset_name})', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(common_languages, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=12)
    fig.subplots_adjust(right=0.8)  # make space on the right
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("pcw.png")


def parse_tokenizer_argument(tokenizer_arg):
    """
    Parse tokenizer argument which can be either:
    1. A comma-separated list of tokenizer names from DEFAULT_TOKENIZER_NAMES
    2. 'all' to use all default tokenizers
    3. A JSON string with custom tokenizer mappings
    """
    if tokenizer_arg.lower() == 'all':
        return TOKENIZER_NAMES
    
    # Try to parse as JSON first (for custom tokenizer mappings)
    try:
        custom_tokenizers = json.loads(tokenizer_arg)
        if isinstance(custom_tokenizers, dict):
            return custom_tokenizers
    except json.JSONDecodeError:
        pass
    
    # Parse as comma-separated list of tokenizer names
    tokenizer_list = [name.strip() for name in tokenizer_arg.split(',')]
    selected_tokenizers = {}
    
    for name in tokenizer_list:
        if name in TOKENIZER_NAMES:
            selected_tokenizers[name] = TOKENIZER_NAMES[name]
        else:
            print(f"Warning: Tokenizer '{name}' not found in default list. Skipping.")
    
    return selected_tokenizers


def parse_language_argument(language_arg):
    """
    Parse language argument which can be either:
    1. A comma-separated list of language keys from DEFAULT_LANGUAGE_KEYS  
    2. 'all' to use all default languages
    3. A JSON string with custom language mappings
    """
    if language_arg.lower() == 'all':
        return LANGUAGE_KEYS
    
    # Try to parse as JSON first (for custom language mappings)
    try:
        custom_languages = json.loads(language_arg)
        if isinstance(custom_languages, dict):
            return custom_languages
    except json.JSONDecodeError:
        pass
    
    # Parse as comma-separated list of language keys
    language_list = [lang.strip() for lang in language_arg.split(',')]
    selected_languages = {}
    
    for lang in language_list:
        if lang in LANGUAGE_KEYS:
            selected_languages[lang] = LANGUAGE_KEYS[lang]
        else:
            print(f"Warning: Language '{lang}' not found in default list. Skipping.")
    
    return selected_languages

def tokenize_sentence_with_all_tokenizers(
    sentence: str,
    tokenizer_names: dict,
    output_file: str = "tokenization_results.txt",
    print_results: bool = True
):
    """
    Tokenize a given sentence with all specified tokenizers and save results to a file.
    
    Parameters:
    - sentence: The input sentence to tokenize
    - tokenizer_names: Dict mapping display names to tokenizer paths
    - output_file: Output file path to save results (default: "tokenization_results.txt")
    - print_results: Whether to print results to console (default: True)
    
    Returns:
    - Dictionary with tokenizer names as keys and tokenization results as values
    """
    
    results = {}
    
    # Header for output
    header = f"TOKENIZATION RESULTS FOR: '{sentence}'\n"
    header += "=" * (len(header) - 1) + "\n\n"
    
    if print_results:
        print(header)
    
    # Open file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        
        for tokenizer_name, tokenizer_path in tokenizer_names.items():
            try:
                print(f"Processing with {tokenizer_name}...")
                
                # Load tokenizer
                tokenizer = Tokenizer.load(tokenizer_path)
                
                # Tokenize the sentence
                tokens = tokenizer.tokenize(sentence)
                
                # Get token count
                token_count = len(tokens)
                
                # Store results
                results[tokenizer_name] = {
                    'tokens': tokens,
                    'count': token_count,
                    'tokenizer_path': tokenizer_path
                }
                
                # Format output
                output_text = f"TOKENIZER: {tokenizer_name}\n"
                output_text += f"PATH: {tokenizer_path}\n"
                output_text += f"TOKEN COUNT: {token_count}\n"
                output_text += f"TOKENS: {tokens}\n"
                
                # For readability, also show tokens with indices
                indexed_tokens = [f"{i}: '{token}'" for i, token in enumerate(tokens)]
                output_text += f"INDEXED TOKENS:\n"
                for indexed_token in indexed_tokens:
                    output_text += f"  {indexed_token}\n"
                
                output_text += "-" * 80 + "\n\n"
                
                # Print to console if requested
                if print_results:
                    print(output_text)
                
                # Write to file
                f.write(output_text)
                
            except Exception as e:
                error_msg = f"ERROR with {tokenizer_name}: {str(e)}\n"
                error_msg += "-" * 80 + "\n\n"
                
                if print_results:
                    print(error_msg)
                f.write(error_msg)
                
                results[tokenizer_name] = {
                    'error': str(e),
                    'tokenizer_path': tokenizer_path
                }
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary statistics
    successful_tokenizers = [name for name, result in results.items() if 'tokens' in result]
    if successful_tokenizers:
        token_counts = [results[name]['count'] for name in successful_tokenizers]
        
        summary = f"\nSUMMARY STATISTICS:\n"
        summary += f"Successfully processed tokenizers: {len(successful_tokenizers)}\n"
        summary += f"Token count range: {min(token_counts)} - {max(token_counts)}\n"
        summary += f"Average token count: {sum(token_counts)/len(token_counts):.2f}\n"
        
        # Find most and least efficient tokenizers
        min_tokens_tokenizer = successful_tokenizers[token_counts.index(min(token_counts))]
        max_tokens_tokenizer = successful_tokenizers[token_counts.index(max(token_counts))]
        
        summary += f"Most efficient (fewest tokens): {min_tokens_tokenizer} ({min(token_counts)} tokens)\n"
        summary += f"Least efficient (most tokens): {max_tokens_tokenizer} ({max(token_counts)} tokens)\n"
        
        if print_results:
            print(summary)
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(summary)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Analyze tokenizer intrinsic metrics across different languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run all analyses with all default tokenizers and languages
            python script.py --tokenizers all --languages all
            
            # Run specific analyses with selected tokenizers
            python script.py --tokenizers "GPT-2,BERT base model (uncased)" --languages all --analyses fertility,parity
            
            # Use custom tokenizers with JSON
            python script.py --tokenizers '{"Custom GPT":"gpt2","Custom BERT":"google-bert/bert-base-uncased"}' --languages all
            
            # Run only vocabulary overlap analysis
            python script.py --tokenizers all --analyses vocab_overlap
        """
    )
    
    parser.add_argument(
        '--tokenizers',
        type=str,
        default='all',
        help='Tokenizers to analyze. Options: "all", comma-separated list of names, or JSON mapping'
    )
    
    parser.add_argument(
        '--languages', 
        type=str,
        default='all',
        help='Languages to analyze. Options: "all", comma-separated list of keys, or JSON mapping'
    )
    
    parser.add_argument(
        '--analyses',
        type=str,
        default='all',
        help='Analyses to run. Options: "all", or comma-separated list from: vocab_sizes,vocab_overlap,fertility,parity,pcw,example_tokenizations'
    )
    
    parser.add_argument(
        '--sample_size',
        type=int,
        default=10000,
        help='Sample size for language analyses (default: 10000)'
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="Muennighoff/flores200",
        help='Dataset to use for language analyses (default: Muennighoff/flores200)'
    )
    
    parser.add_argument(
        '--dataset_config',
        type=str,
        default="all",
        help='Dataset configuration (default: all)'
    )
    
    parser.add_argument(
        '--dataset_split',
        type=str,
        default="dev",
        help='Dataset split to use (default: dev)'
    )
    parser.add_argument(
        '--sample_sentence',
        type=str,
        default="Hello World",
        help='Sample sentence for tokenization'
    )
    
    args = parser.parse_args()
    
    # Parse tokenizers and languages
    tokenizer_names = parse_tokenizer_argument(args.tokenizers)
    language_keys = parse_language_argument(args.languages)
    
    # Parse analyses
    if args.analyses.lower() == 'all':
        analyses = ['vocab_sizes', 'vocab_overlap', 'fertility', 'parity', 'pcw', 'example_tokenizations']
    else:
        analyses = [a.strip() for a in args.analyses.split(',')]
    
    print(f"Selected tokenizers: {list(tokenizer_names.keys())}")
    print(f"Selected languages: {list(language_keys.keys())}")
    print(f"Selected analyses: {analyses}")
    print(f"Sample size: {args.sample_size}")
    print("-" * 50)
    
    # Run analyses
    if 'vocab_sizes' in analyses:
        print("Computing vocabulary sizes...")
        vocab_sizes = list_tokenizer_vocab_sizes(tokenizer_names)
        print(vocab_sizes)
        print("-" * 50)
    
    if 'vocab_overlap' in analyses:
        print("Computing vocabulary overlap (symmetric)...")
        plot_tokenizer_vocab_overlap_symmetric(tokenizer_names)
        print("Computing vocabulary overlap (asymmetric)...")
        plot_tokenizer_vocab_overlap_asymmetric(tokenizer_names)
        print("-" * 50)
    
    if 'fertility' in analyses:
        print("Computing subword fertility...")
        fertility_results = compute_subword_fertility(
            tokenizer_names, 
            language_keys,
            sample_size=args.sample_size,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split
        )
        print("Plotting fertility scores...")
        plot_fertility_scores(fertility_results, args.dataset_name)
        print("-" * 50)
    
    if 'parity' in analyses:
        print("Computing parity scores...")
        parity_results = compute_parity(
            tokenizer_names,
            language_keys,
            sample_size=args.sample_size,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split
        )
        print("Plotting parity scores...")
        plot_parity_scores(parity_results, args.dataset_name)
        print("-" * 50)
    
    if 'pcw' in analyses:
        print("Computing proportion of continued words...")
        pcw_results = compute_proportion_of_continued_words(
            tokenizer_names,
            language_keys,
            sample_size=args.sample_size,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split
        )
        print("Plotting PCW scores...")
        plot_pcw_scores(pcw_results, args.dataset_name)
        print("-" * 50)

    if 'example_tokenizations' in analyses:
        print("Printing tokenization output of different tokenizers for the given input sentence...")
        tokenize_sentence_with_all_tokenizers(
            sentence=args.sample_sentence,
            tokenizer_names=TOKENIZER_NAMES,
            output_file="tokenization_results.txt",
            print_results=True
        )
        print("-" * 50)  

        
    print("Analysis complete!")


if __name__ == "__main__":
    main()









