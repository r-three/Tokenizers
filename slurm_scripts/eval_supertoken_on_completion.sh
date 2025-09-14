#!/bin/bash
#SBATCH --job-name=eval_supertoken_models
#SBATCH --output=/project/aip-craffel/gsa/.slurm/%j.out
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=58G
#SBATCH --time=06:00:00
#SBATCH --account=aip-craffel

# sbatch /home/gsa/tokenizers/slurm_scripts/eval_supertoken_models.sh
models=("google-gemma-2-2b" "common-pile-comma-v0.1" "meta-llama-Llama-3.2-1B" "microsoft-Phi-3-mini-4k-instruct" "gpt2" "bigscience-bloom" "facebook-xglm-564M" "mistralai-tekken" "google-byt5-small" "google-bert-bert-base-multilingual-cased" "Qwen-Qwen3-8B" "tokenmonster-englishcode-32000-consistent-v1" "tiktoken-gpt-4o")
tokenizers=("google/gemma-2-2b" "common-pile/comma-v0.1-1t" "meta-llama/Llama-3.2-1B" "microsoft/Phi-3-mini-4k-instruct" "gpt2" "bigscience/bloom" "facebook/xglm-564M" "mistralai/tekken" "google/byt5-small" "google-bert/bert-base-multilingual-cased" "Qwen/Qwen3-8B" "tokenmonster/englishcode-32000-consistent-v1" "tiktoken/gpt-4o")

models=("tokenmonster-englishcode-32000-consistent-v1")
tokenizers=("tokenmonster/englishcode-32000-consistent-v1")

MEM="14G"
# Load necessary modules
module load slurm/killarney/24.05.7 StdEnv/2023 gcc/13.3 openmpi/5.0.3 cuda/12.6 python/3.10.13

cd /home/gsa/tokenizers
# Activate virtual environment
source /home/gsa/tokenizers/.venv/bin/activate

OUT_DIR="results/paper"

TOKENIZATION_ROBUSTNESS_TASKS="farsi_tokenizer_robustness_completion,tokenizer_robustness_completion_math,tokenizer_robustness_completion_stem,tokenizer_robustness_completion_english,tokenizer_robustness_completion_turkish"
TOKENIZATION_ROBUSTNESS_TASKS="farsi_tokenizer_robustness_completion,tokenizer_robustness_completion_math,tokenizer_robustness_completion_stem,tokenizer_robustness_completion_english,tokenizer_robustness_completion_italian,tokenizer_robustness_completion_chinese"
TOKENIZATION_ROBUSTNESS_TASKS="tokenizer_robustness_completion_math,tokenizer_robustness_completion_stem,tokenizer_robustness_completion_english,tokenizer_robustness_completion_italian,tokenizer_robustness_completion_chinese"
TOKENIZATION_ROBUSTNESS_TASKS="tokenizer_robustness_completion_math,tokenizer_robustness_completion_italian,tokenizer_robustness_completion_chinese"
# TOKENIZATION_ROBUSTNESS_TASKS="tokenizer_robustness_completion_english"
# TOKENIZATION_ROBUSTNESS_TASKS="tokenizer_robustness_completion_stem"
# TOKENIZATION_ROBUSTNESS_TASKS="tokenizer_robustness_completion_chinese"
# TOKENIZATION_ROBUSTNESS_TASKS="tokenizer_robustness_completion_italian"
TOKENIZATION_ROBUSTNESS_TASKS="tokenizer_robustness_completion_math"

EXTRA_BENCHMARKS=""

cnt=$((${#models[@]} - 1))
echo Will evaluate $cnt models
for i in $(seq 0 $cnt); do
	echo $i
	model="${models[i]}"
	tokenizer="${tokenizers[i]}"
	model_name="r-three/toksuite"
	model_name="supertoken_models-llama_$model"
	hf_out_path="r-three/supertoken_models-llama_$model"
	echo "Model: $model"
	echo "Tokenizer: $tokenizer"
	args=" "
	run_name="$(basename $model)-$(date +%s)"

	wandb_args=""
	# wandb_args=" --wandb_args project=supertoken-evaluation-paper,name=$run_name "
	# wandb_args=" --wandb_args project=toksuite-evals,entity=raffel-reports,group=$model_name,name=$run_name "
	common_args=" --model hf --model_args pretrained=${hf_out_path},tokenizer=${tokenizer},trust_remote_code=true --device cuda --verbosity DEBUG ${wandb_args} ${args} "

	# Run lm_eval with converted model
	echo "Running lm_eval..."

	# srun --ntasks=1 --nodes=1 --gres=gpu:1 \
	# 	--exclusive --mem=$MEM \
	# 	-o $OUT_DIR/logs/$run_name.log -e $OUT_DIR/logs/$run_name.err \
	lm_eval $common_args --log_samples \
		--tasks $TOKENIZATION_ROBUSTNESS_TASKS \
		--output_path $OUT_DIR &
done
wait

# lm_eval --model hf --model_args "pretrained=r-three/supertoken_models-llama_tiktoken-gpt-4o,tokenizer=tiktoken/gpt-4o " --device cuda --verbosity DEBUG --log_samples --tasks farsi_tokenizer_robustness_completion,tokenizer_robustness_completion_math,tokenizer_robustness_completion_stem,tokenizer_robustness_completion_english --output_path results/paper-debug
