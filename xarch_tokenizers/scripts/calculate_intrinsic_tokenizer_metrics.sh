#!/bin/bash
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

source /home/ehghaghi/projects/aip-craffel/ehghaghi/Tokenizers/.venv/bin/activate


export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# pip install datasets==3.6.0
# pip install tokenmonster==1.1.12
# pip install spacy
# pip install jieba


# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/calculate_intrinsic_tokenizer_metrics.py"

# Default configuration
DEFAULT_SAMPLE_SIZE=10000

# Default tokenizers (comma-separated)
TOKENIZERS="Comma,Llama-3.2,Phi-3,GPT-2,GPT-4o,BLOOM,XGLM,Tekken,ByT5,mBERT,Qwen-3,TokenMonster,Gemma-2,Aya"

# Default languages (comma-separated) 
LANGUAGES="sentence_eng_Latn,sentence_zho_Hans,sentence_tur_Latn,sentence_pes_Arab,sentence_ita_Latn"

# Run the Python script with the tokenizers and languages
python3 calculate_intrinsic_tokenizer_metrics.py \
    --tokenizers "$TOKENIZERS" \
    --languages "$LANGUAGES" \
    --analyses all \
    --sample_size 10000
