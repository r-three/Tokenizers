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
from datasets import get_dataset_config_names

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

LANGUAGE_KEYS = {'r-three/tokenizer_robustness_completion_english': "eng_Latn", #english
                 'r-three/tokenizer_robustness_completion_math': "eng_Latn", #english
                 'r-three/tokenizer_robustness_completion_stem': "eng_Latn", #english
                 'r-three/tokenizer_robustness_completion_chinese': "zho_Hani", #chinese
                 'r-three/tokenizer_robustness_completion_turkish': "tur_Latn", #turkish
                 'r-three/tokenizer_robustness_completion_farsi': "fas_Arab", #persian
                 'r-three/tokenizer_robustness_completion_italian': "ita_Latn", #Italian
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

def compute_subword_fertility_per_config(
    tokenizer_names: dict,
    language_keys: dict,
    dataset_name: str = "Muennighoff/flores200",
    dataset_split: str = "test",
    text_column: str = "question",  # The column name that contains the text in your dataset
    output_dir: str = "/home/ehghaghi/scratch/ehghaghi/intrinsic_metrics/fertility/"
):
    """
    Compute fertility scores for all dataset configs.
    
    Args:
        tokenizer_names: dict mapping tokenizer names to paths
        language_keys: dict mapping config names to language codes for word tokenizer
        dataset_name: name of the dataset
        dataset_split: which split to use
        text_column: name of the column containing text (default: "question")
        output_dir: directory to save output CSV files
    """
    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)

    # Get all available configs for the dataset
    all_configs = get_dataset_config_names(dataset_name)
    print(f"Found {len(all_configs)} dataset configs")
    
    # Initialize result storage for all configs
    all_configs_results = {}
    
    for dataset_config in all_configs:
        print(f"\n{'='*60}")
        print(f"Processing config: {dataset_config}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
            
            # Check if this config should be processed
            if dataset_name not in language_keys:
                print(f"Config {dataset_name} not in language_keys, skipping...")
                continue
            
            # Get the language code for word tokenization
            language_code = language_keys[dataset_name]
            
            # Filter out empty examples
            examples = [ex for ex in dataset if text_column in ex and ex[text_column].strip()]
            
            if len(examples) == 0:
                print(f"No valid examples found for config {dataset_config}, skipping...")
                continue

            print(f"Sampled examples: {len(examples)}")

            # Extract texts
            texts = [ex[text_column] for ex in examples]

            # Initialize result storage for this config
            config_results = {}

            for tokenizer_name, tokenizer_dir in tokenizer_names.items():
                print(f"Processing tokenizer: {tokenizer_name}")
                tokenizer = Tokenizer.load(tokenizer_dir)

                # --- Fertility ---
                print(f"Processing language: {language_code}")
                cur_tokenizer = load_word_tokenizer(language_code)

                total_words = 0
                text_subwords = 0

                for text in texts:
                    words = cur_tokenizer.word_tokenize(text)
                    total_words += len(words)
                    for word in words:
                        text_subwords += len(tokenizer.tokenize(word)) - TOKENIZER_N_SPECIAL_TOKENS_PER_WORD[tokenizer_name]
                
                fertility = text_subwords / total_words if total_words > 0 else 0
                fertility = round(fertility, 3)

                # Store scores for this tokenizer
                config_results[tokenizer_name] = {
                    "fertility": {dataset_config: fertility}
                }

            # Store results for this config
            all_configs_results[dataset_config] = config_results
            
            # Convert to DataFrame and save for this config
            df = pd.DataFrame.from_dict(
                {model: metrics["fertility"][dataset_config] for model, metrics in config_results.items()},
                orient="index",
                columns=[dataset_config]
            )
            df.to_csv(f"{output_dir}/{dataset_name}/fertility_results_{dataset_config}.csv", index_label="model_name")
            
        except Exception as e:
            print(f"Error processing config {dataset_config}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save a combined summary
    print(f"\n{'='*60}")
    print(f"Processed {len(all_configs_results)} configs successfully")
    print(f"{'='*60}")
    
    # Create combined DataFrame with all configs
    if all_configs_results:
        combined_data = {}
        for config, tokenizers in all_configs_results.items():
            for tokenizer_name, metrics in tokenizers.items():
                if tokenizer_name not in combined_data:
                    combined_data[tokenizer_name] = {}
                combined_data[tokenizer_name][config] = metrics["fertility"][config]
        
        combined_df = pd.DataFrame.from_dict(combined_data, orient="index")
        combined_df.to_csv(f"{output_dir}/{dataset_name}/fertility_results_all_configs.csv", index_label="model_name")
        print("Saved combined results to fertility_results_all_configs.csv")
    
    return all_configs_results

def compute_pcw_per_config(
    tokenizer_names: dict,
    language_keys: dict,
    dataset_name: str = "Muennighoff/flores200",
    dataset_split: str = "test",
    text_column: str = "question",
    output_dir: str = "/home/ehghaghi/scratch/ehghaghi/intrinsic_metrics/pcw/"
):
    """
    Compute Proportion of Continued Words (PCW) scores for all dataset configs.
    
    Args:
        tokenizer_names: dict mapping tokenizer names to paths
        language_keys: dict mapping config names to language codes for word tokenizer
        dataset_name: name of the dataset
        dataset_split: which split to use
        text_column: name of the column containing text (default: "question")
        output_dir: directory to save output CSV files
    """
    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)

    # Get all available configs for the dataset
    all_configs = get_dataset_config_names(dataset_name)
    print(f"Found {len(all_configs)} dataset configs")
    
    # Initialize result storage for all configs
    all_configs_results = {}
    
    for dataset_config in all_configs:
        print(f"\n{'='*60}")
        print(f"Processing config: {dataset_config}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
            
            # Check if this config should be processed
            if dataset_name not in language_keys:
                print(f"Config {dataset_name} not in language_keys, skipping...")
                continue
            
            # Get the language code for word tokenization
            language_code = language_keys[dataset_name]
            
            # Filter out empty examples
            examples = [ex for ex in dataset if text_column in ex and ex[text_column].strip()]
            
            if len(examples) == 0:
                print(f"No valid examples found for config {dataset_config}, skipping...")
                continue

            print(f"Sampled examples: {len(examples)}")

            # Extract texts
            texts = [ex[text_column] for ex in examples]

            # Initialize result storage for this config
            config_results = {}

            for tokenizer_name, tokenizer_dir in tokenizer_names.items():
                print(f"Processing tokenizer: {tokenizer_name}")
                tokenizer = Tokenizer.load(tokenizer_dir)

                # --- Proportion of Continued Words ---
                print(f"Processing language: {language_code}")
                word_tokenizer = load_word_tokenizer(language_code)

                all_words = []
                split_word_count = 0

                for text in texts:
                    words = word_tokenizer.word_tokenize(text)
                    all_words.extend(words)
                    for word in words:
                        # Encode word separately to avoid sentence-level effects
                        tokenized = tokenizer.tokenize(word)
                        if (len(tokenized) - TOKENIZER_N_SPECIAL_TOKENS_PER_WORD[tokenizer_name]) >= 2:
                            split_word_count += 1

                total_word_count = len(all_words)
                proportion = split_word_count / total_word_count if total_word_count > 0 else 0
                proportion = round(proportion, 3)

                # Store scores for this tokenizer
                config_results[tokenizer_name] = {
                    "pcw": {dataset_config: proportion}
                }

            # Store results for this config
            all_configs_results[dataset_config] = config_results
            
            # Convert to DataFrame and save for this config
            df = pd.DataFrame.from_dict(
                {model: metrics["pcw"][dataset_config] for model, metrics in config_results.items()},
                orient="index",
                columns=[dataset_config]
            )
            df.to_csv(f"{output_dir}/{dataset_name}/pcw_results_{dataset_config}.csv", index_label="model_name")
            
        except Exception as e:
            print(f"Error processing config {dataset_config}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save a combined summary
    print(f"\n{'='*60}")
    print(f"Processed {len(all_configs_results)} configs successfully")
    print(f"{'='*60}")
    
    # Create combined DataFrame with all configs
    if all_configs_results:
        combined_data = {}
        for config, tokenizers in all_configs_results.items():
            for tokenizer_name, metrics in tokenizers.items():
                if tokenizer_name not in combined_data:
                    combined_data[tokenizer_name] = {}
                combined_data[tokenizer_name][config] = metrics["pcw"][config]
        
        combined_df = pd.DataFrame.from_dict(combined_data, orient="index")
        combined_df.to_csv(f"{output_dir}/{dataset_name}/pcw_results_all_configs.csv", index_label="model_name")
        print("Saved combined results to pcw_results_all_configs.csv")
    
    return all_configs_results

def plot_fertility_scores(results, dataset_name):
    """
    Plot fertility scores for all configs.
    
    Args:
        results: Nested dictionary {config: {tokenizer: {fertility: {...}}}}
        dataset_name: Name of the dataset
    """
    # Iterate through each config
    for config_name, config_results in results.items():
        print(f"Plotting fertility scores for config: {config_name}")
        
        tokenizers = list(config_results.keys())
        
        # Get common languages across all tokenizers for this config
        all_languages = [set(config_results[tokenizer]["fertility"].keys()) for tokenizer in tokenizers]
        common_languages = set.intersection(*all_languages)
        
        if not common_languages:
            print(f"No common languages found for config {config_name}, skipping...")
            continue

        fertility_data = []

        for tokenizer in tokenizers:
            fertilities = [config_results[tokenizer]["fertility"][lang] for lang in common_languages]
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
        ax.set_title(f'Fertility Scores by Tokenizer and Language\n({dataset_name} - {config_name})', fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(common_languages, rotation=45, ha='right', fontsize=16)
        ax.tick_params(axis='y', labelsize=12)
        fig.subplots_adjust(right=0.8)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        
        # Save with config name in filename
        plt.savefig(f"fertility_{config_name}.png")
        plt.close()
        
    print(f"Saved {len(results)} fertility plots")
    
def plot_pcw_scores(results, dataset_name):
    """
    Plot proportion of continued words scores for all configs.
    
    Args:
        results: Nested dictionary {config: {tokenizer: {proportion_of_continued_words: {...}}}}
        dataset_name: Name of the dataset
    """
    # Iterate through each config
    for config_name, config_results in results.items():
        print(f"Plotting PCW scores for config: {config_name}")
        
        tokenizers = list(config_results.keys())
        
        # Get common languages across all tokenizers for this config
        all_languages = [set(config_results[tokenizer]["proportion_of_continued_words"].keys()) for tokenizer in tokenizers]
        common_languages = set.intersection(*all_languages)
        
        if not common_languages:
            print(f"No common languages found for config {config_name}, skipping...")
            continue

        pcw_data = []

        for tokenizer in tokenizers:
            pcws = [config_results[tokenizer]["proportion_of_continued_words"][lang] for lang in common_languages]
            pcw_data.append(pcws)

        pcw_data = np.array(pcw_data)

        num_tokenizers = len(tokenizers)
        num_languages = len(common_languages)
        group_width = 0.8
        bar_width = group_width / num_tokenizers
        x = np.arange(num_languages)

        # Plot PCW
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, tokenizer in enumerate(tokenizers):
            ax.bar(x + i * bar_width - group_width/2 + bar_width/2, pcw_data[i], bar_width, label=tokenizer)

        ax.set_xlabel('Languages', fontsize=18)
        ax.set_ylabel('Proportion of Continued Words (PCW)', fontsize=18)
        ax.set_title(f'Proportion of Continued Words by Tokenizer and Language\n({dataset_name} - {config_name})', fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(common_languages, rotation=45, ha='right', fontsize=16)
        ax.tick_params(axis='y', labelsize=12)
        fig.subplots_adjust(right=0.8)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        
        # Save with config name in filename
        plt.savefig(f"pcw_{config_name}.png")
        plt.close()
        
    print(f"Saved {len(results)} PCW plots")

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

def main():
    parser = argparse.ArgumentParser(
        description="Analyze tokenizer intrinsic metrics across tokenizer robustness datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run all analyses with all default tokenizers
            python script.py --tokenizers all
            
            # Run specific analyses with selected tokenizers
            python script.py --tokenizers "GPT-2,BERT base model (uncased)" --analyses fertility,parity
            
            # Use custom tokenizers with JSON
            python script.py --tokenizers '{"Custom GPT":"gpt2","Custom BERT":"google-bert/bert-base-uncased"}'
            
            # Run only vocabulary overlap analysis
            python script.py --tokenizers all --analyses fertility
        """
    )
    
    parser.add_argument(
        '--tokenizers',
        type=str,
        default='all',
        help='Tokenizers to analyze. Options: "all", comma-separated list of names, or JSON mapping'
    )
    
    parser.add_argument(
        '--analyses',
        type=str,
        default='all',
        help='Analyses to run. Options: "all", or comma-separated list from: fertility,pcw'
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="r-three/tokenizer_robustness_completion_english",
        help='Dataset to use for language analyses (default: r-three/tokenizer_robustness_completion_english)'
    )
    
    parser.add_argument(
        '--dataset_split',
        type=str,
        default="test",
        help='Dataset split to use (default: test)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default="/home/ehghaghi/scratch/ehghaghi/intrinsic_metrics/",
        help='Dataset split to use (default: /home/ehghaghi/scratch/ehghaghi/intrinsic_metrics/)'
    )
    
    args = parser.parse_args()
    
    # Parse tokenizers and languages
    tokenizer_names = parse_tokenizer_argument(args.tokenizers)
    
    # Parse analyses
    if args.analyses.lower() == 'all':
        analyses = ['fertility', 'pcw']
    else:
        analyses = [a.strip() for a in args.analyses.split(',')]
    
    print(f"Selected tokenizers: {list(tokenizer_names.keys())}")
    print(f"Selected analyses: {analyses}")
    print("-" * 50)
    
    # Run analyses    
    if 'fertility' in analyses:
        print("Computing subword fertility...")
        fertility_results = compute_subword_fertility_per_config(
            tokenizer_names, 
            LANGUAGE_KEYS,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            output_dir=os.path.join(args.output_dir, "fertility")
        )
        print("Plotting fertility scores...")
        # plot_fertility_scores(fertility_results, args.dataset_name)
        print("-" * 50)
    
    if 'pcw' in analyses:
        print("Computing proportion of continued words...")
        pcw_results = compute_pcw_per_config(
            tokenizer_names,
            LANGUAGE_KEYS,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            output_dir=os.path.join(args.output_dir, "pcw")
        )
        print("Plotting PCW scores...")
        # plot_pcw_scores(pcw_results, args.dataset_name)
        print("-" * 50)

        
    print("Analysis complete!")


if __name__ == "__main__":
    main()









