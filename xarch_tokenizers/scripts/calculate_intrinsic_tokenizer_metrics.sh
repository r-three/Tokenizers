#!/bin/bash
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

source /home/ehghaghi/projects/aip-craffel/ehghaghi/Tokenizers/.venv/bin/activate


export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache

# # pip install datasets==3.6.0
# # pip install tokenmonster==1.1.12
# # pip install spacy
# # pip install jieba

# python calculate_intrinsic_tokenizer_metrics.py


# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/calculate_intrinsic_tokenizer_metrics.py"

# Default configuration
DEFAULT_SAMPLE_SIZE=10000

# Default tokenizers (comma-separated)
TOKENIZERS=TOKENIZERS="BERT multilingual base model (cased),BERT base model (uncased),T5,mT5,XGLM-564M,Gemma 2,Phi-3-Mini-4K-Instruct,Mistral v3,TokenMonster,ByT5 - Small,BLOOM,GPT-2,GPT-4 Tiktoken,GPT-4o Tiktoken,Mistral v3 (tekken),Llama-3.2 1B,Qwen3-8B,Aya Expanse 8B,Common Pile v1.0"

# Default languages (comma-separated) 
LANGUAGES="sentence_eng_Latn,sentence_zho_Hans,sentence_tur_Latn,sentence_pes_Arab,sentence_ita_Latn"

# Run the Python script with the tokenizers and languages
python3 calculate_intrinsic_tokenizer_metrics.py \
    --tokenizers "$TOKENIZERS" \
    --languages "$LANGUAGES" \
    --analyses all \
    --sample_size 10000

echo "Analysis complete!"