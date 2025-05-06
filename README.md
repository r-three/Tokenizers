# Tokenizer-level experiments


## Set-up
We recommend using uv (install it with `pip install uv` if not already available).
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
## Sample Usage:
LM-Eval
```bash
uv run eval xarch_tokenizers/configs/mgsm/mgsm_eval_llama8B.yaml 
```

## Lm-Eval new datasets
Add new datasets under lm_eval_datasets

This repo inherits from https://github.com/r-three/ca-merging.

## Tokenizer Fail Cases
### Convert dataset
```bash
python xarch_tokenizers/scripts/convert_dataset_to_hf_format.py --dataset_path data/custom_dataset.json --output_path data/custom_dataset_hf.json
```


- `custom_dataset.json`
```bash
python xarch_tokenizers/scripts/eval.py xarch_tokenizers/configs/tokenization_robustness/eval_llama8B.yaml
python xarch_tokenizers/scripts/eval.py xarch_tokenizers/configs/tokenization_robustness/eval_qwen_7B.yaml
python xarch_tokenizers/scripts/eval.py xarch_tokenizers/configs/tokenization_robustness/eval_qwen_05B.yaml
```
