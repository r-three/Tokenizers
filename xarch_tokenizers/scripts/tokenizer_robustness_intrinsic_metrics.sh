#!/bin/bash
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

source /home/ehghaghi/projects/aip-craffel/ehghaghi/Tokenizers/.venv/bin/activate


# export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache

# pip install datasets==3.6.0
# pip install tokenmonster==1.1.12
# pip install spacy
# pip install jieba


# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/calc_intrinsic_metrics_v2.py"
DATASET_NAME="r-three/tokenizer_robustness_completion_farsi"
OUTPUT_DIR="$SCRATCH/tokenizer_robustness_intrinsic_metrics_results"


# Default tokenizers (comma-separated)
TOKENIZERS="Comma,Llama-3.2,Phi-3,GPT-2,GPT-4o,BLOOM,XGLM,Tekken,ByT5,mBERT,Qwen-3,TokenMonster,Gemma-2,Aya"

# Run the Python script with the tokenizers and languages
python3 tokenizer_robustness_intrinsic_metrics.py \
    --tokenizers "$TOKENIZERS" \
    --dataset_name "$DATASET_NAME" \
    --analyses all \
    --output_dir "$OUTPUT_DIR"

