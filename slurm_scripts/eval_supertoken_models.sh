#!/bin/bash
#SBATCH --job-name=eval_supertoken_models
#SBATCH --output=/project/aip-craffel/gsa/.slurm/%j.out
#SBATCH --gres=gpu:l40s:3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=aip-craffel

# sbatch /home/gsa/tokenizers/slurm_scripts/eval_supertoken_models.sh

# Load necessary modules
module load slurm/killarney/24.05.7 StdEnv/2023 gcc/13.3 openmpi/5.0.3 cuda/12.6 python/3.10.13

cd /home/gsa/tokenizers
# Activate virtual environment
source /home/gsa/tokenizers/.venv/bin/activate

export HF_ALLOW_CODE_EVAL=1

pairs=(
"gsaltintas/supertoken_models-llama_google-gemma-2-2b google/gemma-2-2b" 
# "gsaltintas/gsa-supertoken-gpt-4o gpt-4o"
)
for pair in "${pairs[@]}"; do
    read -r model tokenizer <<< "$pair"
    echo "Model: $model"
    echo "Tokenizer: $tokenizer"
	args=""
	if [[ $model == *gpt* ]]; then args=",tokenizer_backend=tiktoken"; fi
    run_name="tokenization-robustness-$(basename $model)-$(date +%s)"

    wandb_args=" --wandb_args project=supertoken-evaluation,name=$run_name "
    common_args="--model hf --model_args \"pretrained=$model,tokenizer=$tokenizer$args\" --device cuda --log_samples --verbosity DEBUG $wandb_args "

# Run lm_eval with converted model
echo "Running lm_eval..."
srun --exclusive --ntasks=1  \
lm_eval $common_args $wandb_args \
--tasks tokenizer_robustness_code_technical_content,tokenizer_robustness_context-dependent_ambiguities,tokenizer_robustness_mathematical_scientific_notation,tokenizer_robustness_morphological_challenges,tokenizer_robustness_multi-linguality,tokenizer_robustness_named_entities,tokenizer_robustness_orthographic_variations,tokenizer_robustness_social_media_informal_text,tokenizer_robustness_structural_text_elements,tokenizer_robustness_temporal_expressions \
--output_path "results/tokenization_robustness/v102-cleaned/supertoken/$model"

srun --exclusive --ntasks=1 lm_eval $common_args $wandb_args \
--tasks hellaswag,piqa,arc_easy,arc_challenge,include_base_44_turkish,include_base_44_italian,include_base_44_chinese,belebele_pes_Arab,belebele_eng_Latn,belebele_ita_Latn,belebele_tur_Latn,belebele_zho_Hans,xnli_en,xnli_tr,xnli_zh \
--output_path "results/supertoken-training/$model" --confirm_run_unsafe_code
# # TODO: humaneval

srun --exclusive --ntasks=1 lm_eval $common_args $wandb_args \
--tasks humaneval \
--output_path 'results/humaneval/$model' --confirm_run_unsafe_code
done
