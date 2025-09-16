"""Defines reporting utilitis and results processing"""

import json
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .plot_utils import TASK_TO_PLOT_MAPPING

MODEL_PRETTY_NAMES = {
    "common-pile-comma-v0.1": "Comma",
    "meta-llama-Llama-3.2-1B": "Llama-3.2",
    "microsoft-Phi-3-mini-4k-instruct": "Phi-3",
    "gpt2": "GPT-2",
    "bigscience-bloom": "BLOON",
    "facebook-xglm-564M": "XGLM",
    "mistralai-tekken": "Tekken",
    "google-byt5-small": "ByT5",
    "google-bert-bert-base-multilingual-cased": "mBERT",
    "Qwen-Qwen3-8B": "Qwen-3",
    "tokenmonster-englishcode-32000-consistent-v1": "TokenMonster",
    "tiktoken-gpt-4o": "GPT-4o",
    "google-gemma-2-2b": "Gemma-2",
    "CohereLabs-aya-expanse-8b": "Aya",
}


def clean_model_name(model_name: str) -> str:
    model_name = model_name.split("/")[-1]
    model_name = model_name.replace("supertoken_models-llama_", "")
    model_name = MODEL_PRETTY_NAMES.get(model_name, model_name)
    return model_name


def load_results(base_dir: Union[Path, str], patterns: Optional[List[str]] = None):
    """Loads the results and configs from a base_dir into a unified dataframe

    patterns: Pass to filter the sub_folders, e.g. ["mgsm_eval_en*"]
    """
    base_dir = Path(base_dir)
    sub_dirs_to_look = []
    if patterns is None or len(patterns) == 0:
        patterns = ["*"]
    for pattern in patterns:
        sub_dirs_to_look.extend([p for p in base_dir.rglob(f"{pattern}") if p.is_dir()])

    res_dfs = []
    for sub_dir in sub_dirs_to_look:
        res_path = sub_dir / "results.json"
        if not res_path.exists():
            continue
        dct = json.loads(res_path.read_text())
        conf = dct["config"]
        res = dct["results"]
        res = pd.DataFrame.from_records(res)
        conf = pd.DataFrame.from_dict(conf, orient="index").transpose()
        df = res.join(conf, how="cross")
        res_dfs.append(df)

    print(len(sub_dirs_to_look))
    print("Processing results from ", sub_dirs_to_look)
    res_dfs = pd.concat(res_dfs, ignore_index=True)
    res_dfs["task"] = res_dfs["dataset"].apply(lambda x: TASK_TO_PLOT_MAPPING.get(x, x))
    return res_dfs


def load_predictions_lm_eval(
    base_dir: Union[Path, str],
    patterns: Optional[List[str]] = None,
    doc_ids: List[int] = None,
    filters: List[str] = ["flexible-extract"],
    metrics: List[str] = ["exact_match"],
    additional_docs: List[str] = [],
):
    """Loads the results and configs from a base_dir into a unified dataframe

    patterns: Pass to filter the sub_folders, e.g. ["mgsm_eval_en*"]
    """
    base_dir = Path(base_dir)
    sub_dirs_to_look = []
    if patterns is None or len(patterns) == 0:
        patterns = ["*"]
    for pattern in patterns:
        sub_dirs_to_look.extend([p for p in base_dir.rglob(f"{pattern}") if p.is_dir()])

    pred_dfs = []
    for sub_dir in sub_dirs_to_look:
        pred_path = sub_dir / "predictions.json"
        res_path = sub_dir / "results.json"
        if not pred_path.exists():
            continue

        dct = json.loads(res_path.read_text())
        conf = dct["config"]
        res = dct["results"]
        res = pd.DataFrame.from_records(res)
        conf = pd.DataFrame.from_dict(conf, orient="index").transpose()[
            ["model_name", "tokenizer_name", "experiment_name"]
        ]

        preds = json.loads(pred_path.read_text())
        for task in preds:
            task_preds = preds[task]
            if doc_ids is not None:
                task_preds = [
                    pred_ for pred_ in task_preds if pred_["doc_id"] in doc_ids
                ]
            task_preds = [
                {
                    "doc_id": pred_["doc_id"],
                    "target": pred_["target"],
                    "prompt": pred_["arguments"][0][0],
                    "resp": pred_["resps"][0][0],
                    "filtered_resps": pred_["filtered_resps"][0],
                }
                | {doc_arg: pred_["doc"][doc_arg] for doc_arg in additional_docs}
                | {metric: pred_[metric] for metric in metrics}
                for pred_ in task_preds
                if filters is None or pred_["filter"] in filters
            ]
            df = pd.DataFrame.from_records(task_preds)
            df["dataset"] = task
            df = df.join(conf, how="cross")
            # preds_.append({task: task_preds})
            pred_dfs.append(df)

    print(
        "Processing predictions from %d: %s" % (len(sub_dirs_to_look), sub_dirs_to_look)
    )
    pred_dfs = pd.concat(pred_dfs, ignore_index=True)
    pred_dfs["task"] = pred_dfs["dataset"].apply(
        lambda x: TASK_TO_PLOT_MAPPING.get(x, x)
    )
    return pred_dfs


def get_pretty_name(x):
    return x.split(",")[0]
    if isinstance(x, str):
        return x
    return x[0]


def load_all_samples(
    base_dir: Union[Path, str],
    patterns: Optional[List[str]] = None,
    flatten_doc: bool = False,
    match_date: bool = True,
    simplify_df: bool = True,
):
    """Loads all the samples from a base_dir into a unified dataframe
    simplify_df: If True, drops hash columns
    """
    base_dir = Path(base_dir)
    pred_files = []
    if patterns is None or len(patterns) == 0:
        patterns = ["*"]
    for pattern in patterns:
        pred_files.extend([p for p in base_dir.rglob(f"samples{pattern}.jsonl")])
    print(len(pred_files), pred_files[:5])
    samples = []
    for sample_path in pred_files:
        res_path = list(sample_path.parent.glob("results*.json"))[0]
        if not res_path.exists():
            continue
        date = res_path.stem.strip("results_")
        results = json.loads(res_path.read_text())
        model_name = results["model_name"]
        model_args = results["config"]["model_args"]
        if "tokenizer=" in model_args:
            tokenizer_name = model_args[model_args.index("tokenizer=") + 10 :]
            tokenizer_name = tokenizer_name.split(",")[0]
        else:
            tokenizer_name = model_name
        if match_date and date not in sample_path.stem:
            print(f"Skipping {sample_path} because date mismatch with {date}")
            continue
        task_name = (
            sample_path.stem.strip("samples_")
            .strip(f"_{date}")
            .split("2025")[0]
            .strip("_")
        )
        try:
            sample = pd.read_json(path_or_buf=sample_path, lines=True)
            sample["model_name"] = model_name
            sample["tokenizer_name"] = tokenizer_name
            sample["task"] = task_name
            samples.append(sample)
        except Exception as e:
            print(f"Error loading {sample_path}: {e}")

    samples = pd.concat(samples, ignore_index=True)
    if flatten_doc:
        df2 = pd.json_normalize(samples["doc"], max_level=1)
        samples = pd.concat([samples, df2], axis=1).drop(columns=["doc"])
        # samples["answer_choice"] = samples.apply(
        #     lambda x: x["choices"][x["target"]]
        #     if isinstance(x["target"], int) and (isinstance(x["choices"], list))
        #     else None,
        #     axis=1,
        # )
    # samples["task"] = samples["dataset"].apply(lambda x: TASK_TO_PLOT_MAPPING.get(x, x))
    if simplify_df:
        samples = samples.drop(
            columns=[col for col in samples.columns if "hash" in col]
        )
        samples = samples.drop(columns=["arguments", "filter", "filtered_resps"])
    print(samples["set_id"].unique())
    samples["task_pretty_name"] = samples["subcategories"].apply(get_pretty_name)
    samples["model_name"] = samples["model_name"].apply(clean_model_name)
    samples["is_canonical"] = (
        samples["variation_id"].apply(lambda x: str(x).split(".")[1]) == "0"
    )
    samples["var_id"] = samples["variation_id"].apply(
        lambda x: float(str(x).split(".")[1])
    )
    samples = samples.sort_values(["model_name", "set_id", "var_id"])
    return samples


def load_all_samples_old(
    base_dir: Union[Path, str],
    patterns: Optional[List[str]] = None,
    flatten_doc: bool = False,
    match_date: bool = True,
    simplify_df: bool = True,
):
    """Loads all the samples from a base_dir into a unified dataframe
    simplify_df: If True, drops hash columns
    """
    base_dir = Path(base_dir)
    sub_dirs_to_look = []
    if patterns is None or len(patterns) == 0:
        patterns = ["*"]
    for pattern in patterns:
        sub_dirs_to_look.extend(
            [
                p
                for p in base_dir.rglob(f"{pattern}")
                if p.is_dir() and len(list(p.glob("results*.json"))) > 0
            ]
        )

    samples = []
    for sub_dir in sub_dirs_to_look:
        res_path = list(sub_dir.rglob("results*.json"))[0]
        date = res_path.stem.strip("results_")
        results = json.loads(res_path.read_text())
        model_name = results["model_name"]
        for sample_path in sub_dir.rglob("samples*.jsonl"):
            if match_date and date not in sample_path.stem:
                print(f"Skipping {sample_path} because date mismatch with {date}")
                continue
            task_name = (
                sample_path.stem.strip("samples_")
                .strip(f"_{date}")
                .split("2025")[0]
                .strip("_")
            )
            try:
                sample = pd.read_json(path_or_buf=sample_path, lines=True)
                sample["model_name"] = model_name
                sample["task"] = task_name
                samples.append(sample)
            except Exception as e:
                print(f"Error loading {sample_path}: {e}")

    samples = pd.concat(samples, ignore_index=True)
    if flatten_doc:
        df2 = pd.json_normalize(samples["doc"], max_level=1)
        samples = pd.concat([samples, df2], axis=1).drop(columns=["doc"])
        samples["answer_choice"] = samples.apply(
            lambda x: x["choices"][x["target"]]
            if isinstance(x["target"], int) and (isinstance(x["choices"], list))
            else None,
            axis=1,
        )
    # samples["task"] = samples["dataset"].apply(lambda x: TASK_TO_PLOT_MAPPING.get(x, x))
    if simplify_df:
        samples = samples.drop(
            columns=[col for col in samples.columns if "hash" in col]
        )
        samples = samples.drop(columns=["arguments", "filter", "filtered_resps"])
    return samples


def load_predictions(
    base_dir: Union[Path, str],
    patterns: Optional[List[str]] = None,
    doc_ids: List[int] = None,
    filters: List[str] = ["flexible-extract"],
    metrics: List[str] = ["exact_match"],
    additional_docs: List[str] = [],
):
    """Loads the results and configs from a base_dir into a unified dataframe

    patterns: Pass to filter the sub_folders, e.g. ["mgsm_eval_en*"]
    """
    base_dir = Path(base_dir)
    sub_dirs_to_look = []
    if patterns is None or len(patterns) == 0:
        patterns = ["*"]
    for pattern in patterns:
        sub_dirs_to_look.extend([p for p in base_dir.rglob(f"{pattern}") if p.is_dir()])

    pred_dfs = []
    for sub_dir in sub_dirs_to_look:
        pred_path = sub_dir / "predictions.json"

        res_path = sub_dir / "results.json"

        if not pred_path.exists():
            continue

        dct = json.loads(res_path.read_text())
        conf = dct["config"]
        res = dct["results"]
        res = pd.DataFrame.from_records(res)
        conf = pd.DataFrame.from_dict(conf, orient="index").transpose()[
            ["model_name", "tokenizer_name", "experiment_name"]
        ]

        preds = json.loads(pred_path.read_text())
        for task in preds:
            task_preds = preds[task]
            if doc_ids is not None:
                task_preds = [
                    pred_ for pred_ in task_preds if pred_["doc_id"] in doc_ids
                ]
            task_preds = [
                {
                    "doc_id": pred_["doc_id"],
                    "target": pred_["target"],
                    "prompt": pred_["arguments"][0][0],
                    "resp": pred_["resps"][0][0],
                    "filtered_resps": pred_["filtered_resps"][0],
                }
                | {doc_arg: pred_["doc"][doc_arg] for doc_arg in additional_docs}
                | {metric: pred_[metric] for metric in metrics}
                for pred_ in task_preds
                if filters is None or pred_["filter"] in filters
            ]
            df = pd.DataFrame.from_records(task_preds)
            df["dataset"] = task
            df = df.join(conf, how="cross")
            # preds_.append({task: task_preds})
            pred_dfs.append(df)

    print(
        "Processing predictions from %d: %s" % (len(sub_dirs_to_look), sub_dirs_to_look)
    )
    pred_dfs = pd.concat(pred_dfs, ignore_index=True)
    pred_dfs["task"] = pred_dfs["dataset"].apply(
        lambda x: TASK_TO_PLOT_MAPPING.get(x, x)
    )
    return pred_dfs
