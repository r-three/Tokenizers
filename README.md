# Tokenizer-level experiments
This code base contains code to experiment with tokenizers in the xarch project.
This repo eventually inherits from https://github.com/r-three/ca-merging.

## Contributing
- There is no restriction but in general try to add your scripts under `xarch_tokenizers/scripts` and create a corresponding config, see `EvaluationConfig` in [experiment_config](xarch_tokenizers/experiment_config.py).
- Please add all new dependencies with uv, e.g. `uv add XXX` 

## Set-up
We recommend using uv (install it with `pip install uv` if not already available).

### On Killarney
On the Killarney cluster, you need to first load the following modules:
```bash
module load slurm/killarney/24.05.7 StdEnv/2023  gcc/13.3  openmpi/5.0.3 cuda/12.6 python/3.10.13
```
and for the first time you run the code, you need to install the packages to the system:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# If you don't have a virtual environment already, you can either
# 1. Install the packages to the system
uv pip install -e . --system

# 2. Create a venv with uv
# make sure to load cuda (locally built with cuda-12.4)
uv venv --python 3.10
source .venv/bin/activate
## First run
uv sync --extra build
uv sync --all-extras
# on machines w/o cuda
uv sync --all-extras --all-groups  --no-install-package flash-attn
```
If you have another uv venv, you can add this package to the original projects `pyproject.toml` as below and run `uv sync --extra tokenizers` in the main directory:
```toml
[project.optional-dependencies]
tokenizers = ["xarch-tokenizers"]

[tool.uv.sources]
xarch-tokenizers = { path = "../tokenizers", editable = true }
```

## Sample Usage:
LM-Eval
```bash
uv run eval xarch_tokenizers/configs/mgsm/mgsm_eval_llama8B.yaml 
```

## Lm-Eval new datasets
Add new datasets under `xarch_tokenizers/lm_eval_datasets`, for local datasets follow [`tokenization_robustness`](xarch_tokenizers/lm_eval_datasets/tokenization_robustness/_tokenization_robustness.yaml), for HuggingFace datasets follow [`mgsm`](xarch_tokenizers/lm_eval_datasets/mgsm/_default_template.yaml) configs, create `task/task_subset.yaml` for datasets with subsets.

You can further override task specific settings, for an example see [this](xarch_tokenizers/configs/mgsm/mgsm_eval_qwen.yaml) config file.

You can also add custom metrics and filters for lm_eval_harness in [`lm_eval.py`](xarch_tokenizers/data/lm_eval.py).
## Tokenization Robustness Dataset
We have a custom dataset [here](data/custom_dataset.json), follow the format when adding new examples. To be able to run evaluation on this dataset, we need to convert its formatting see below.

Sample evaluation configs can be found here
```bash
python xarch_tokenizers/scripts/eval.py xarch_tokenizers/configs/tokenization_robustness/eval_llama8B.yaml
python xarch_tokenizers/scripts/eval.py xarch_tokenizers/configs/tokenization_robustness/eval_qwen_7B.yaml
python xarch_tokenizers/scripts/eval.py xarch_tokenizers/configs/tokenization_robustness/eval_qwen_05B.yaml
```

### Convert dataset to HF format and upload to HF
```bash
python xarch_tokenizers/scripts/convert_dataset_to_hf_format.py xarch_tokenizers/configs/tokenization_robustness/convert_v102_to_hf.yaml
```

## Other Functionality
- You can upload custom datasets to huggingface with [this](xarch_tokenizers/scripts/upload_dataset_to_hf.py) script.
- Token surgeon is our fork of the arcee's token surgeon script.



## Converting Supertoken Models
```bash

model="gpt4o"
tokenizer="tiktoken-gpt-4o"
model_name="craffel/supertoken_models"
model_path="$model_name/$model/"
tokenizer="blester125/supervocab-$tokenizer"
hf_model_path="$PROJECT/models/$model_name"
tokenizer_path="$PROJECT/tokenizers/$tokenizer"

# Create directories
mkdir -p "$hf_model_path"
mkdir -p "$hf_model_path"

huggingface-cli download $model_name --local-dir=$hf_model_path
huggingface-cli download $tokenizer --local-dir=$tokenizer_path
# Convert LLaMA weights to HuggingFace format
echo "Converting model weights to HuggingFace format..."
python -m xarch_tokenizers.scripts.convert_supertoken_models \
    --input_dir "$hf_model_path/$model" \
    --model_size 1B \
    --output_dir "$hf_model_path" \
    --llama_version 3 --tokenizer_version 3 \
    --tokenizer_path "$tokenizer_path" --only_model \
    --push_to_hub --output_dir gsaltintas/supertoken_models_$model \
    --only_model

# Run lm_eval with converted model
echo "Running lm_eval..."
lm_eval \
    --model_args "pretrained=$hf_model_path,tokenizer=$tokenizer" \
    --device cuda \
    --tasks tokenizer_robustness_code_technical_content,tokenizer_robustness_context-dependent_ambiguities,tokenizer_robustness_mathematical_scientific_notation,tokenizer_robustness_morphological_challenges,tokenizer_robustness_multi-linguality,tokenizer_robustness_named_entities,tokenizer_robustness_orthographic_variations,tokenizer_robustness_social_media_informal_text,tokenizer_robustness_structural_text_elements,tokenizer_robustness_temporal_expressions  \
    --log_samples \
    --verbosity DEBUG \
    --output_path "results/tokenizer_robustness/supertoken/$model"
```