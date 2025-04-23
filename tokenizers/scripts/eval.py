"""Evaluation script that calls lm_eval.simple_evaluate"""

import json
from dataclasses import asdict

import pandas as pd
import torch
import wandb
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from tabulate import tabulate

from tokenizers.data.lm_eval import get_new_manager, load_tasks_with_overrides
from tokenizers.experiment_config import EvaluationConfig, load_config
from tokenizers.logging.logger import setup_logger
from tokenizers.logging.wandb_utils import setup_wandb
from tokenizers.models import load_tokenizer
from tokenizers.utils.lm_eval import dump_predictions
from tokenizers.utils.system import VECTOR_HF_MAPPING


def setup_lm_eval_environment(config):
    """Set up environment, models, and tokenizers."""
    config.save_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(config, "eval")
    logger.info("Config saved at %s", config.experiment_dir / "config.json")
    logger.info("Starting eval with config:")
    config.display()

    # Load tokenizer and model
    tokenizer_name = VECTOR_HF_MAPPING.get(config.tokenizer_name, config.tokenizer_name)
    tokenizer = load_tokenizer(tokenizer_name)
    model_name = VECTOR_HF_MAPPING.get(config.model_name, config.model_name)
    logger.info("Loading model: %s", model_name)
    model = HFLM(
        pretrained=model_name, tokenizer=tokenizer, device=device, dtype=config.dtype
    )

    return model, tokenizer, logger, device


def run_evaluation(config, model):
    """Run the evaluation on specified tasks."""
    manager = get_new_manager(
        load_existing=config.load_existing, override_config=config.get_override_config()
    )
    task_names = [dataset.lm_eval_alias for dataset in config.get_dataset_configs()]
    # Load tasks with overrides
    tasks = load_tasks_with_overrides(
        task_names,
        global_overrides=dict(),
        task_specific_overrides=config.get_override_config(),
        task_manager=manager,
    )

    kwargs = dict()
    if config.apply_chat_template:
        kwargs["apply_chat_template"] = True
    if config.chat_template:
        kwargs["apply_chat_template"] = config.chat_template

    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=config.num_fewshot,
        limit=config.num_samples,
        bootstrap_iters=0,
        random_seed=config.seed,
        batch_size="auto",
        device="cuda:0",
        task_manager=manager,
        cache_requests=False,
        **kwargs,
    )

    return results, tasks


def process_results(results, tasks, config):
    """Process and save evaluation results."""
    processed_results = []

    if config.dump_predictions:
        dump_predictions(results, config.experiment_dir)

    for dataset in tasks:
        res = results["results"][dataset]
        processed_results.append(
            {**res, "dataset": dataset, "model": config.model_name}
        )
        if config.use_wandb:
            wandb.log({f"{dataset}/{k}": v for k, v in res.items()})

    return processed_results


def save_results(processed_results, config):
    """Save results to CSV and JSON files."""
    df = pd.DataFrame(processed_results)
    print(tabulate(df, headers="keys", tablefmt="psql"))

    # Save as CSV
    df.to_csv(config.experiment_dir.joinpath("results.csv"))

    # Save as JSON with config
    with open(
        config.experiment_dir.joinpath("results.json"), "w", encoding="utf-8"
    ) as f:
        df_dict = df.to_dict(orient="records")
        result_dict = {
            "config": {
                k: v for k, v in asdict(config).items() if not k.startswith("_")
            },
            "results": df_dict,
        }
        json.dump(result_dict, f, indent=2, default=str)


def main():
    """Entry point for translate command."""
    config = load_config(EvaluationConfig)
    setup_wandb(config)
    model, tokenizer, logger, device = setup_lm_eval_environment(config)
    results, tasks = run_evaluation(config, model)
    task_names = results["group_subtasks"]
    processed_results = process_results(results, task_names, config)
    save_results(processed_results, config)


if __name__ == "__main__":
    main()

# 15655055
# sbatch --time 480 --mem-per-cpu 16G --cpus-per-task 4 --gres=gpu:a40:1 --job-name=mgsm --wrap="uv run eval ca/configs/mgsm_eval.yaml "

"""
15516921
sbatch --time 480 --mem-per-cpu 8G --cpus-per-task 4  --gres=gpu:a40:1 --job-name=8B-multi --wrap="uv run eval --datasets turkishmmlu_philosophy turkishmmlu_mathematics cmmlu_modern_chinese   \
    translated_turkishmmlu_philosophy translated_turkishmmlu_mathematics translated_cmmlu_modern_chinese \
    --num_samples 100000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct"
   
    
# 15516923
sbatch --time 480 --mem-per-cpu 8G --cpus-per-task 4  --gres=gpu:a40:1 --job-name=3B-multi --wrap="uv run eval --datasets turkishmmlu_philosophy turkishmmlu_mathematics cmmlu_modern_chinese   \
    translated_turkishmmlu_philosophy translated_turkishmmlu_mathematics translated_cmmlu_modern_chinese \
    --num_samples 100000 \
    --model_name meta-llama/Llama-3.2-3B-Instruct"
    
15516924
sbatch --time 480 --mem-per-cpu 8G --cpus-per-task 4  --gres=gpu:a40:1 --job-name=1B-multi --wrap="uv run eval --datasets turkishmmlu_philosophy turkishmmlu_mathematics cmmlu_modern_chinese   \
    translated_turkishmmlu_philosophy translated_turkishmmlu_mathematics translated_cmmlu_modern_chinese \
    --num_samples 100000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct"
    
    
#### Qwen
15517756
sbatch --time 480 --mem-per-cpu 8G --cpus-per-task 4  --gres=gpu:a40:1 --job-name=7B-multi --wrap="uv run eval --datasets turkishmmlu_philosophy turkishmmlu_mathematics cmmlu_modern_chinese   \
    translated_turkishmmlu_philosophy translated_turkishmmlu_mathematics translated_cmmlu_modern_chinese \
    --num_samples 100000 \
    --model_name Qwen/Qwen2.5-7B-Instruct"
    
15517755
sbatch --time 480 --mem-per-cpu 8G --cpus-per-task 4  --gres=gpu:a40:1 --job-name=3B-multi --wrap="uv run eval --datasets turkishmmlu_philosophy turkishmmlu_mathematics cmmlu_modern_chinese   \
    translated_turkishmmlu_philosophy translated_turkishmmlu_mathematics translated_cmmlu_modern_chinese \
    --num_samples 100000 \
    --model_name Qwen/Qwen2.5-3B-Instruct"
"""
