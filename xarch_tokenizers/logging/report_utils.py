"""Defines reporting utilitis and results processing"""

import json
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .plot_utils import TASK_TO_PLOT_MAPPING


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
