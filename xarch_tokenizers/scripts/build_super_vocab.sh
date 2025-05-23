#!/usr/bin/env sh

python super_vocab.py \
    --tokenizers \
    google-bert/bert-base-uncased \
    google-bert/bert-base-multilingual-cased \
    google-t5/t5-base \
    google/mt5-base \
    facebook/xglm-564M \
    google/gemma-2-2b \
    microsoft/Phi-3-mini-4k-instruct \
    mistralai/Mistral-7B-Instruct-v0.3 \
    tokenmonster/englishcode-32000-consistent-v1 \
    google/byt5-small \
    bigscience/bloom \
    gpt2 \
    tiktoken/gpt-4 \
    tiktoken/gpt-4o \
    mistralai/tekken \
    meta-llama/Llama-3.2-1B \
    Qwen/Qwen3-8B \
    CohereLabs/aya-expanse-8b \
    common-pile/comma-v0.1
