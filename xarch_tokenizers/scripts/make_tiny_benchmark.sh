#!/bin/bash

# ========== User Configuration ==========
DATASET_PATH="juletxara/mgsm"                             # HF dataset name, e.g., "glue", "super_glue", etc.
PERCENTAGE=10                                   # Percentage of data to keep (default: 10%)
OUTPUT_REPO="Malikeh1375/tiny-mgsm"           # HF Hub repo to push tiny version (e.g., malikeh/glue-tiny)
MIN_SAMPLES=100                                 # Minimum number of samples per split
DELAY_BETWEEN_PUSHES=10                         # Delay between pushed to HF repo
MAX_RETRIES=3                                 # Maximum retries to repush the data to HF after failure


# ========== Run Subsampling ==========
echo "🚀 Starting tiny benchmark subsampling for dataset: $DATASET_PATH"

python make_tiny_benchmark.py \
    --dataset_path "$DATASET_PATH" \
    --percentage "$PERCENTAGE" \
    --output_repo "$OUTPUT_REPO" \
    --min_samples "$MIN_SAMPLES" \
    --delay_between_pushes "$DELAY_BETWEEN_PUSHES" \
    --max_retries "$MAX_RETRIES"


echo "✅ All done! Tiny dataset pushed to: $OUTPUT_REPO"
