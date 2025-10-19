#!/bin/bash

# Simple lm-eval-harness script for OpenAI and Anthropic APIs

set -e

module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

# ============================================================================
# Configuration
# ============================================================================
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
# Choose provider: "openai" or "anthropic"

PROVIDER="openai"

# Model name
MODEL="babbage-002"  # For OpenAI:only openai-chat-completion models retrun logprobs not openai-completions. Two examples: davinci-002, and babbage-002
                # For Anthropic: claude-3-5-sonnet-20241022, claude-3-opus-20240229, etc.

# Tasks to run (comma-separated, no spaces)
TASKS="tokenizer_robustness_completion_math"

# Output directory

OUTPUT_DIR="/home/ehghaghi/scratch/ehghaghi/lm_eval_results/${PROVIDER}_${MODEL}_results/${TASKS}"
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache

# Number of few-shot examples
NUM_FEWSHOT=0

# ============================================================================
# Setup and Run
# ============================================================================

# Check API key
if [ "$PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: Set OPENAI_API_KEY first"
        echo "Run: export OPENAI_API_KEY='your-key'"
        exit 1
    fi
    MODEL_TYPE="openai-completions"
elif [ "$PROVIDER" = "anthropic" ]; then
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "Error: Set ANTHROPIC_API_KEY first"
        echo "Run: export ANTHROPIC_API_KEY='your-key'"
        exit 1
    fi
    MODEL_TYPE="anthropic"
else
    echo "Error: PROVIDER must be 'openai' or 'anthropic'"
    exit 1
fi

# Print info
echo "Running evaluation..."
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Tasks: $TASKS"
echo ""

# Run evaluation
lm_eval --model "$MODEL_TYPE" \
        --model_args model="$MODEL" \
        --tasks "$TASKS" \
        --num_fewshot "$NUM_FEWSHOT" \
        --output_path "$OUTPUT_DIR"
        

echo ""
echo "Done! Results in: $OUTPUT_DIR"