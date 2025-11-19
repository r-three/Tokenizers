import json
import math
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from category_mapping import SUBCATEGORY_TO_CATEGORY
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from xarch_tokenizers.logging.plot_utils import (
    MODEL_TO_COLOR,
    get_new_6_fig,
    get_new_fig,
    setup_styles,
)
from xarch_tokenizers.logging.report_utils import (
    clean_model_name,
    load_all_samples,
    load_predictions,
)

whitespace_robustness_dict = {}
METRIC = "acc_norm"

## same as exploratory plotting notebook `paper-benchmark-plots.ipynb`
def get_canonical_and_perturbed_df(
    samples,
    only_keep_canonical_true: bool = False,
    keep_remains_or_becomes_correct: bool = False,
    only_keep_perturbed_true: bool = False,
    verbose: bool = False,
):
    """returns filtered out samples too"""
    if (
        (only_keep_canonical_true and keep_remains_or_becomes_correct)
        or (only_keep_canonical_true and only_keep_perturbed_true)
        or (keep_remains_or_becomes_correct and only_keep_perturbed_true)
    ):
        raise ValueError(
            f"Please pass these exclusively, only_keep_canonical_true: true for canonical, model_name true, keep_remains_or_becomes_correct: True to include the ones where the perturbed version becomes correct"
        )
    if verbose:
        print(f"Processing {len(samples)} examples")
    canonical_samples = samples[samples["is_canonical"]]
    perturbed_samples = samples[~samples["is_canonical"]]
    canonical_values = canonical_samples[["set_id", "model_name", METRIC]].rename(
        columns={METRIC: f"canonical_{METRIC}"}
    )

    perturbed_samples = perturbed_samples.merge(
        canonical_values, on=["set_id", "model_name"], how="left"
    )
    perturbed_samples[f"canonical-perturbed_{METRIC}"] = (
        perturbed_samples[f"canonical_{METRIC}"] - perturbed_samples[METRIC]
    )

    if only_keep_canonical_true:
        correct_canonical_pairs = canonical_samples[canonical_samples[METRIC] == 1][
            ["set_id", "model_name"]
        ]
        if verbose:
            print(
                f"Cleaning samples, will be dropping entries (set_id-model pairs) where the canonical is predicted wrong."
            )
            print(
                f"Keeping {len(correct_canonical_pairs)} correct pairs, dropping {len(canonical_samples) - len(correct_canonical_pairs)} entries."
            )

        canonical_samples = canonical_samples[canonical_samples[METRIC] == 1]

        # Step 3: Filter perturbed_samples to keep only rows with correct canonical pairs
        # Method 1: Using merge (RECOMMENDED)
        perturbed_samples = perturbed_samples.merge(
            correct_canonical_pairs,
            on=["set_id", "model_name"],
            how="inner",  # Only keep rows that exist in both
        )
        samples = pd.concat((canonical_samples, perturbed_samples), ignore_index=True)

    if keep_remains_or_becomes_correct:
        # Get pairs where perturbed samples are correct (becomes or remains correct)
        correct_perturbed_pairs = perturbed_samples[perturbed_samples[METRIC] == 1][
            ["set_id", "model_name"]
        ].drop_duplicates()

        # Get pairs where canonical samples are correct (remains correct)
        correct_canonical_pairs = canonical_samples[canonical_samples[METRIC] == 1][
            ["set_id", "model_name"]
        ]

        # Combine: pairs that either remain correct OR become correct
        remains_or_becomes_correct = pd.concat(
            [correct_canonical_pairs, correct_perturbed_pairs]
        ).drop_duplicates()

        # Filter both datasets to only include these pairs
        canonical_samples = canonical_samples.merge(
            remains_or_becomes_correct, on=["set_id", "model_name"], how="inner"
        )

        perturbed_samples = perturbed_samples.merge(
            remains_or_becomes_correct, on=["set_id", "model_name"], how="inner"
        )
        samples = pd.concat((canonical_samples, perturbed_samples), ignore_index=True)
        if verbose:
            print(f"After filtering:{len(samples)}")
    if only_keep_perturbed_true:
        # Get pairs where perturbed samples are correct (becomes or remains correct)
        correct_perturbed_pairs = perturbed_samples[perturbed_samples[METRIC] == 1][
            ["set_id", "model_name"]
        ].drop_duplicates()

        # Get pairs where canonical samples are correct (remains correct)
        wrong_canonical_pairs = canonical_samples[canonical_samples[METRIC] == 0][
            ["set_id", "model_name"]
        ]
        # Find cases where canonical wrong AND perturbed correct (recovery cases)
        becomes_correct = pd.merge(
            wrong_canonical_pairs,
            correct_perturbed_pairs,
            how="inner",
            on=["set_id", "model_name"],
        )

        # Filter both datasets to only include recovery cases
        canonical_samples = canonical_samples.merge(
            becomes_correct, on=["set_id", "model_name"], how="inner"
        )

        perturbed_samples = perturbed_samples.merge(
            becomes_correct, on=["set_id", "model_name"], how="inner"
        )

        samples = pd.concat([canonical_samples, perturbed_samples], ignore_index=True)
        if verbose:
            print(f"After filtering: {len(samples)}")

    return samples, canonical_samples, perturbed_samples

def save_to_jsonl(latex_dict, filename="tokenization_results.jsonl"):
    """Save results to JSONL format"""
    records = []
    for model_name, row in latex_dict["raw_dataframe"].iterrows():
        record = {"model_name": model_name}
        for col, value in row.items():
            record[col] = value if not pd.isna(value) else None
        records.append(record)
    
    with open(filename, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    print(f"Results saved to {filename}")
    return filename

def get_canonical_name(s):
    """Map perturbation task names to canonical task names"""
    if "tokenizer_robustness_completion_english" in s:
        return "tokenizer_robustness_completion_english_canonical"
    elif "tokenizer_robustness_completion_chinese" in s:
        return "tokenizer_robustness_completion_chinese_canonical"
    elif "tokenizer_robustness_completion_turkish" in s:
        return "tokenizer_robustness_completion_turkish_canonical"
    elif "tokenizer_robustness_completion_italian" in s:
        return "tokenizer_robustness_completion_italian_canonical"
    elif "tokenizer_robustness_completion_farsi" in s:
        return "tokenizer_robustness_completion_farsi_canonical"
    elif "tokenizer_robustness_completion_math" in s:
        return "tokenizer_robustness_completion_math_canonical"
    elif "tokenizer_robustness_completion_stem" in s:
        return "tokenizer_robustness_completion_stem_canonical"
    else:
        return None


def calculate_robustness_drop(data, canonical_by_task):
    """Calculate robustness drop for a dataset"""
    if len(data) == 0:
        return np.nan
    
    robustness_drops = []
    weights_list = []
    
    for _, row in data.iterrows():
        canonical_task = get_canonical_name(row["task"])
        if canonical_task:
            canonical_key = (row["model_name"], canonical_task, row["langs"])
            if "math" in canonical_task:
                canonical_key = (row["model_name"], canonical_task, "eng_Latn")
            if canonical_key in canonical_by_task:
                canonical_acc = canonical_by_task[canonical_key]
                perturb_acc = row["acc_norm"]
                
                if canonical_acc > 0 and not pd.isna(canonical_acc) and not pd.isna(perturb_acc):
                    robustness_drop = (canonical_acc - perturb_acc) / canonical_acc
                    robustness_drops.append(robustness_drop)
                    weights_list.append(row["num_samples"])
    
    if robustness_drops:
        return np.average(robustness_drops, weights=weights_list)
    else:
        return np.nan

def extract_robustness_table_data(all_summaries, canonical_by_task, filter_mode="no_filter", latex_table_categories=dict(), group_langs: bool=True):
    """Extract robustness drop data for LaTeX table"""
    
    # Filter and prepare data
    filtered_summaries = all_summaries[all_summaries["filtering_mode"] == filter_mode].copy()
    if group_langs:
        filtered_summaries["language_split"] = filtered_summaries["langs"].apply(
        lambda x: "EN" if x == "eng_Latn" else "Non-EN"
    )
    else:
        filtered_summaries["language_split"] = filtered_summaries["langs"].map(
        {"eng_Latn": "EN", "pes_Arab": "FA", "tur_Latn": "TR", "ita_Latn": "IT", "zho_Hans": "ZH"}
    )
    
    results = {}
    
    for model in filtered_summaries["model_name"].unique():
        model_data = filtered_summaries[filtered_summaries["model_name"] == model]
        model_results = {"model_name": model}
        
        # Process each category for robustness drops
        for category, perturbations in latex_table_categories.items():
            if "(Math)" in category:
                data = model_data[
                    (model_data["task_pretty_name"].isin(perturbations)) &
                    (("math" in model_data["task"]) | ("stem" in model_data["task"]) )
                ]
            elif "(EN)" in category:
                # English only
                data = model_data[
                    (model_data["task_pretty_name"].isin(perturbations)) &
                    (model_data["language_split"] == "EN")
                ]
            elif "(FA)" in category:
                data = model_data[
                    (model_data["task_pretty_name"].isin(perturbations)) &
                    (model_data["language_split"] == "FA")
                ]
            elif "(ZH)" in category:
                data = model_data[
                    (model_data["task_pretty_name"].isin(perturbations)) &
                    (model_data["language_split"] == "ZH")
                ]
            elif "(TR)" in category:
                data = model_data[
                    (model_data["task_pretty_name"].isin(perturbations)) &
                    (model_data["language_split"] == "TR")
                ]
            elif "(IT)" in category:
                data = model_data[
                    (model_data["task_pretty_name"].isin(perturbations)) &
                    (model_data["language_split"] == "IT")
                ]
            elif "(Non-EN)" in category:
                # Non-English only
                data = model_data[
                    (model_data["task_pretty_name"].isin(perturbations)) &
                    (model_data["language_split"] == "Non-EN")
                ]
            else:
                # Combined (all languages)
                data = model_data[model_data["task_pretty_name"].isin(perturbations)]
            print(f"Processing model: {model}, category: {category}, data size: {len(data)}, perturbations: {perturbations}, \n\ttasks: {data['task'].unique()}")
            model_results[category] = calculate_robustness_drop(data, canonical_by_task)
        
        # Add whitespace robustness drop
        # model_results["Whitespace"] = whitespace_robustness_dict.get(model, np.nan)
        results[model] = model_results
    
    return results

def get_canonical_by_task(all_summaries, filter_mode="no_filter"):

    # Get canonical scores by task if calculating robustness drop
    canonical_by_task = {}
    if calculate_robustness_drop:
        canonical_data = all_summaries[
            (all_summaries["filtering_mode"] == filter_mode) &
            (all_summaries["task_pretty_name"] == "Canonical")
        ]
        for _, row in canonical_data.iterrows():
            key = (row["model_name"], row["task"], row["langs"])
            canonical_by_task[key] = row["acc_norm"]
    return canonical_by_task

def generate_robustness_latex_table_dict(all_summaries, canonical_by_task, filter_mode="no_filter", column_order=None, latex_table_categories=dict(), group_langs=True):
    """Generate complete data structure for robustness LaTeX table"""
    
    # Extract robustness data
    results = extract_robustness_table_data(all_summaries, canonical_by_task, filter_mode, latex_table_categories=latex_table_categories, group_langs=group_langs)
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.drop("model_name", axis=1)
    
    # Define column order
    if column_order is None:
        column_order = list(latex_table_categories.keys())
    
    # Select available columns
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    return {
        "data": df.to_dict("index"),
        "columns": available_columns,
        "models": df.index.tolist(),
        "raw_dataframe": df,
    }

def generate_robustness_latex_table(df_display, title: str="", header_mapping=dict()):
    """Generate formatted LaTeX table for robustness drops"""
    
    n_cols = len(df_display.columns)
    
    # Start table
    latex_str = f"""\\begin{{table}}[hp]
\\centering 
\\vspace*{{\\fill}}
\\caption{{{title}}}
\\label{{tab:multilingual_tokenization_robustness}}
\\footnotesize
\\begin{{tabularx}}{{\\textwidth}}{{l|*{{{n_cols}}}{{>{{\\centering\\arraybackslash}}X}}}}
\\toprule
\\textbf{{Model}} &"""
    
    # Column headers
    
    for col in df_display.columns:
        header_name = header_mapping.get(col, col)
        latex_str += f"\n\\rotatebox{{60}}{{\\textbf{{{header_name}}}}} &"
    latex_str = latex_str.rstrip(" &") + " \\\\\n\\midrule\n"
    
    # Find best and worst robustness (lowest and highest drops)
    best_values = {}  # Lowest drops (best robustness)
    worst_values = {}  # Highest drops (worst robustness)
    for col in df_display.columns:
        valid_vals = df_display[col].dropna()
        if len(valid_vals) > 1:
            best_values[col] = valid_vals.min()  # Lowest drop = best robustness
            worst_values[col] = valid_vals.max()  # Highest drop = worst robustness
    
    # Data rows
    for model in df_display.index:
        latex_str += model
        for col in df_display.columns:
            val = df_display.loc[model, col]
            if pd.isna(val):
                latex_str += " & ---"
            else:
                val_str = f"{val:.2f}"
                # Color formatting for robustness: green for low drops, red for high drops
                if col in best_values and abs(val - best_values[col]) < 1e-4:
                    val_str = f"\\textbf{{\\textcolor{{green!70!black}}{{{val_str}}}}}"  # Best robustness
                elif col in worst_values and abs(val - worst_values[col]) < 1e-4:
                    val_str = f"\\textcolor{{red}}{{{val_str}}}"  # Worst robustness
                latex_str += f" & {val_str}"
        latex_str += " \\\\\n"
    
    # Summary row
    latex_str += "\\midrule\nAvg"
    col_avgs = df_display.mean(axis=0, skipna=True)
    best_col_avg = col_avgs.min() if len(col_avgs.dropna()) > 0 else None  # Lowest avg drop
    worst_col_avg = col_avgs.max() if len(col_avgs.dropna()) > 0 else None  # Highest avg drop
    
    for col in df_display.columns:
        avg_val = col_avgs[col]
        if pd.isna(avg_val):
            latex_str += " & ---"
        else:
            val_str = f"{avg_val:.2f}"
            if best_col_avg is not None and abs(avg_val - best_col_avg) < 1e-6:
                val_str = f"\\textbf{{\\textcolor{{green!70!black}}{{{val_str}}}}}"
            elif worst_col_avg is not None and abs(avg_val - worst_col_avg) < 1e-6:
                val_str = f"\\textcolor{{red}}{{{val_str}}}"
            latex_str += f" & {val_str}"
    latex_str += " \\\\\n"
    
    # End table
    latex_str += f"""\\bottomrule
\\end{{tabularx}}
\\vspace*{{\\fill}}
\\end{{table}}"""
    
    return latex_str

def load_and_generate_robustness_latex_table(jsonl_file="robustness_results.jsonl", title: str = "", header_mapping=dict()):
    """Load JSONL data and generate formatted robustness LaTeX table"""
    
    # Load data
    records = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    
    df = pd.DataFrame(records).set_index('model_name')
    
    # Model name mapping
    model_mapping = {
        "google-byt5-small": "ByT5",
        "tokenmonster-englishcode-32000-consistent-v1": "TokenMonster", 
        "microsoft-Phi-3-mini-4k-instruct": "Phi-3",
        "gpt2": "GPT-2",
        "common-pile-comma-v0.1": "Comma",
        "google-bert-bert-base-multilingual-cased": "mBERT",
        "meta-llama-Llama-3.2-1B": "Llama-3.2",
        "mistralai-tekken": "Tekken",
        "Qwen-Qwen3-8B": "Qwen-3",
        "tiktoken-gpt-4o": "GPT-4o",
        "bigscience-bloom": "BLOOM",
        "aya-expanse": "Aya",
        "google-gemma-2-2b": "Gemma-2",
        "facebook-xglm-564m": "XGLM"
    }
    
    # Apply model name mapping
    df.index = [model_mapping.get(model, model) for model in df.index]
    
    # Add average column (mean robustness drop)
    df['Average'] = df.mean(axis=1, skipna=True)
    
    # Sort by average robustness (lowest drops first = best robustness)
    df = df.sort_values('Average', ascending=True)
    
    # Generate LaTeX
    return generate_robustness_latex_table(df, title=title, header_mapping=header_mapping)



## cano
def get_canonical_name(s):
    if "tokenizer_robustness_completion_english" in s:
        return "tokenizer_robustness_completion_english_canonical"
    elif "tokenizer_robustness_completion_chinese" in s:
        return "tokenizer_robustness_completion_chinese_canonical"
    elif "tokenizer_robustness_completion_turkish" in s:
        return "tokenizer_robustness_completion_turkish_canonical"
    elif "tokenizer_robustness_completion_italian" in s:
        return "tokenizer_robustness_completion_italian_canonical"
    elif "tokenizer_robustness_completion_farsi" in s:
        return "tokenizer_robustness_completion_farsi_canonical"
    elif "tokenizer_robustness_completion_math" in s:
        return "tokenizer_robustness_completion_math_canonical"
    elif "tokenizer_robustness_completion_stem" in s:
        return "tokenizer_robustness_completion_stem_canonical"
    

    elif "tokenizer_robustness_auto_english" in s:
        return "tokenizer_robustness_auto_english_canonical"
    elif "tokenizer_robustness_auto_chinese" in s:
        return "tokenizer_robustness_auto_chinese_canonical"
    elif "tokenizer_robustness_auto_turkish" in s:
        return "tokenizer_robustness_auto_turkish_canonical"
    elif "tokenizer_robustness_auto_italian" in s:
        return "tokenizer_robustness_auto_italian_canonical"
    elif "tokenizer_robustness_auto_farsi" in s:
        return "tokenizer_robustness_auto_farsi_canonical"
    
    elif "tokenizer_robustness_completion_english" in s:
        return "tokenizer_robustness_completion_english_canonical"
    elif "tokenizer_robustness_completion_chinese" in s:
        return "tokenizer_robustness_completion_chinese_canonical"
    elif "tokenizer_robustness_completion_turkish" in s:
        return "tokenizer_robustness_completion_turkish_canonical"
    elif "tokenizer_robustness_completion_italian" in s:
        return "tokenizer_robustness_completion_italian_canonical"
    elif "tokenizer_robustness_completion_farsi" in s:
        return "tokenizer_robustness_completion_farsi_canonical"
    else:
        return None  # Return None if no match found
    


def get_group_name(task_name):
    if "stem" in task_name:
        return "tokenizer_robustness_completion_stem"
    elif "english" in task_name:
        return "tokenizer_robustness_completion_english"
    elif "turkish" in task_name:
        return "tokenizer_robustness_completion_turkish"
    elif "farsi" in task_name:
        return "tokenizer_robustness_completion_farsi"
    elif "italian" in task_name:
        return "tokenizer_robustness_completion_italian"
    elif "chinese" in task_name:
        return "tokenizer_robustness_completion_chinese"
    elif "math" in task_name:
        return "tokenizer_robustness_completion_math"
    elif "general" in task_name:
        return "tokenizer_robustness_completion_general"


def format_language_name(s):
    if s == "eng_Latn":
        return "EN"
    elif s== "pes_Arab":
        return "FA"
    elif s=="tur_Latn":
        return "TR"
    elif s=="zho_Hans":
        return "ZH"
    elif s=="ita_Latn":
        return "IT"
    
def create_hierarchical_category_summary(all_summaries, category_mapping, filter_mode="no_filter", calculate_robustness_drop=True, group_langs: bool = True, groupby="model_name"):
    """
    Create a summary table with hierarchical columns: Category -> Language
    If calculate_robustness_drop=True, compute (canonical_acc - perturb_acc) / canonical_acc for each task
    
    Args:
        all_summaries: DataFrame with all summary data
        category_mapping: Dict mapping categories to perturbation types
        filter_mode: Filtering mode to use
        calculate_robustness_drop: Whether to calculate robustness drops
        group_langs: If True, group non-English languages together. If False, keep each language separate
    """
    # Filter data
    filtered_summaries = all_summaries[all_summaries["filtering_mode"] == filter_mode].copy()

    # Get canonical scores by task if calculating robustness drop
    canonical_by_task = {}
    if calculate_robustness_drop:
        canonical_data = all_summaries[
            (all_summaries["filtering_mode"] == filter_mode) &
            (all_summaries["task_pretty_name"] == "Canonical")
        ]
        for _, row in canonical_data.iterrows():
            key = (row[groupby], row["task"], row["langs"])
            canonical_by_task[key] = row["acc_norm"]

    # Create hierarchical structure
    results_data = []
    
    # Separate English and non-English languages
    non_eng_languages = [lang for lang in filtered_summaries["langs"].unique() if lang != "eng_Latn"]
    for model in filtered_summaries[groupby].unique():
        model_data = filtered_summaries[filtered_summaries[groupby] == model]
        row_data = []
        
        for category_name, perturbations in category_mapping.items():
            category_data = model_data[model_data["task_pretty_name"].isin(perturbations)]
            # English performance
            eng_data = category_data[category_data["langs"] == "eng_Latn"]
            if len(eng_data) > 0:
                if calculate_robustness_drop:
                    # Calculate robustness drop for each task, then average
                    robustness_drops = []
                    weights_list = []
                    
                    for _, row in eng_data.iterrows():
                        canonical_key = (row[groupby], get_canonical_name(row["task"]), row["langs"])
                        if canonical_key in canonical_by_task:
                            canonical_acc = canonical_by_task[canonical_key]
                            perturb_acc = row["acc_norm"]
                            
                            if canonical_acc > 0 and not pd.isna(canonical_acc) and not pd.isna(perturb_acc):
                                robustness_drop = (canonical_acc - perturb_acc) / canonical_acc
                                robustness_drops.append(robustness_drop)
                                weights_list.append(row["num_samples"])
                        else:
                            print(f"not fond {canonical_key}")

                    if robustness_drops:
                        eng_avg = np.average(robustness_drops, weights=weights_list)
                    else:
                        eng_avg = np.nan
                else:
                    # Regular accuracy calculation
                    acc_values = eng_data["acc_norm"]
                    weights = eng_data["num_samples"]
                    valid_mask = ~(pd.isna(acc_values) | pd.isna(weights)) & (weights > 0)
                    if valid_mask.sum() > 0:
                        eng_avg = np.average(acc_values[valid_mask], weights=weights[valid_mask])
                    else:
                        eng_avg = np.nan
            else:
                eng_avg = np.nan
            row_data.append(eng_avg)
            
            # Non-English performance
            if group_langs:
                # Group all non-English languages together
                non_eng_data = category_data[category_data["langs"] != "eng_Latn"]
                if len(non_eng_data) > 0:
                    if calculate_robustness_drop:
                        # Calculate robustness drop for each task, then average
                        robustness_drops = []
                        weights_list = []
                        
                        for _, row in non_eng_data.iterrows():
                            canonical_key = (row[groupby], get_canonical_name(row["task"]), row["langs"])
                            if canonical_key in canonical_by_task:
                                canonical_acc = canonical_by_task[canonical_key]
                                perturb_acc = row["acc_norm"]
                                
                                if canonical_acc > 0 and not pd.isna(canonical_acc) and not pd.isna(perturb_acc):
                                    robustness_drop = (canonical_acc - perturb_acc) / canonical_acc
                                    robustness_drops.append(robustness_drop)
                                    weights_list.append(row["num_samples"])
                        
                        if robustness_drops:
                            non_eng_avg = np.average(robustness_drops, weights=weights_list)
                        else:
                            non_eng_avg = np.nan
                    else:
                        # Regular accuracy calculation
                        acc_values = non_eng_data["acc_norm"]
                        weights = non_eng_data["num_samples"]
                        valid_mask = ~(pd.isna(acc_values) | pd.isna(weights)) & (weights > 0)
                        if valid_mask.sum() > 0:
                            non_eng_avg = np.average(acc_values[valid_mask], weights=weights[valid_mask])
                        else:
                            non_eng_avg = np.nan
                else:
                    non_eng_avg = np.nan
                
                row_data.append(non_eng_avg)
            else:
                # Keep each non-English language separate
                for lang in non_eng_languages:
                    lang_data = category_data[category_data["langs"] == lang]
                    if len(lang_data) > 0:
                        if calculate_robustness_drop:
                            # Calculate robustness drop for each task, then average
                            robustness_drops = []
                            weights_list = []
                            
                            for _, row in lang_data.iterrows():
                                lang_ = row["langs"] 
                                if "math" in row["task"]:
                                    lang_="eng_Latn"
                                canonical_key = (row[groupby], get_canonical_name(row["task"]), lang_)
                                if canonical_key in canonical_by_task:
                                    canonical_acc = canonical_by_task[canonical_key]
                                    perturb_acc = row["acc_norm"]
                                    
                                    if canonical_acc > 0 and not pd.isna(canonical_acc) and not pd.isna(perturb_acc):
                                        robustness_drop = (canonical_acc - perturb_acc) / canonical_acc
                                        robustness_drops.append(robustness_drop)
                                        weights_list.append(row["num_samples"])
                            
                            if robustness_drops:
                                lang_avg = np.average(robustness_drops, weights=weights_list)
                            else:
                                lang_avg = np.nan
                        else:
                            # Regular accuracy calculation
                            acc_values = lang_data["acc_norm"]
                            weights = lang_data["num_samples"]
                            valid_mask = ~(pd.isna(acc_values) | pd.isna(weights)) & (weights > 0)
                            if valid_mask.sum() > 0:
                                lang_avg = np.average(acc_values[valid_mask], weights=weights[valid_mask])
                            else:
                                lang_avg = np.nan
                    else:
                        lang_avg = np.nan
                    
                    row_data.append(lang_avg)
            
        results_data.append([model] + row_data)
    
    # Create column structure
    columns = ['model_name']
    column_tuples = []
    
    for category_name in category_mapping.keys():
        column_tuples.append((category_name, 'EN'))
        
        if group_langs:
            column_tuples.append((category_name, 'Non-English'))
        else:
            # Add each non-English language separately
            for lang in non_eng_languages:
                # Convert language code to more readable format
                lang_display = format_language_name(lang)
                column_tuples.append((category_name, lang_display))
    # Create DataFrame
    df_multi = pd.DataFrame(results_data, columns=['model_name'] + [f"{cat}_{lang}" for cat, lang in column_tuples])
    df_multi = df_multi.set_index('model_name')
    df_multi.columns = pd.MultiIndex.from_tuples(column_tuples, names=['Category', 'Language'])
    
    columns_to_keep = []
    for col in df_multi.columns:
        if not df_multi[col].isna().all():
            columns_to_keep.append(col)
    
    if len(columns_to_keep) < len(df_multi.columns):
        df_multi = df_multi[columns_to_keep]
        print(f"Dropped {len(df_multi.columns) - len(columns_to_keep)} columns")
    df_multi[('Avg', '')] = df_multi.mean(axis=1, skipna=True)
    return df_multi

def create_subcategory_robustness_table(all_summaries, subcategory_mapping, filter_mode="no_filter", calculate_robustness_drop=True):
    """
    Create robustness table showing subcategories broken down by language
    Similar to your Register & Style table structure
    """
    # Filter data
    filtered_summaries = all_summaries[all_summaries["filtering_mode"] == filter_mode].copy()
    
    # Get canonical scores by task
    canonical_by_task = {}
    if calculate_robustness_drop:
        canonical_data = all_summaries[
            (all_summaries["filtering_mode"] == filter_mode) &
            (all_summaries["task_pretty_name"] == "Canonical")
        ]
        for _, row in canonical_data.iterrows():
            key = (row["model_name"], row["task"], row["langs"])
            canonical_by_task[key] = row["acc_norm"]
    # Create results structure
    results_data = []
    
    for model in filtered_summaries["model_name"].unique():
        model_data = filtered_summaries[filtered_summaries["model_name"] == model]
        row_data = {"model_name": model}
        
        for subcategory_name, perturbations in subcategory_mapping.items():
            # Get data for this subcategory
            subcategory_data = model_data[model_data["task_pretty_name"].isin(perturbations)]
            
            # Group by language and calculate robustness drops
            for lang in subcategory_data["langs"].unique():
                lang_data = subcategory_data[subcategory_data["langs"] == lang]
                
                if len(lang_data) > 0 and calculate_robustness_drop:
                    robustness_drops = []
                    weights_list = []
                    
                    for _, row in lang_data.iterrows():
                        canonical_key = (row["model_name"], get_canonical_name(row["task"]), row["langs"])
                        if canonical_key in canonical_by_task:
                            canonical_acc = canonical_by_task[canonical_key]
                            perturb_acc = row["acc_norm"]
                            
                            if canonical_acc > 0 and not pd.isna(canonical_acc) and not pd.isna(perturb_acc):
                                robustness_drop = (canonical_acc - perturb_acc) / canonical_acc
                                robustness_drops.append(robustness_drop)
                                weights_list.append(row["num_samples"])
                    
                    if robustness_drops:
                        lang_avg = np.average(robustness_drops, weights=weights_list)
                        row_data[f"{subcategory_name}_{lang}"] = lang_avg
                    else:
                        row_data[f"{subcategory_name}_{lang}"] = np.nan
                else:
                    row_data[f"{subcategory_name}_{lang}"] = np.nan
        
        results_data.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(results_data).set_index("model_name")
    
    # Create MultiIndex columns (subcategory, language)
    column_tuples = []
    for col in df.columns:
        if "_" in col:
            parts = col.rsplit("_", 1)  # Split from right, only once
            subcategory = parts[0]
            language = parts[1]
            column_tuples.append((subcategory, language))
        else:
            column_tuples.append((col, ""))
    
    df.columns = pd.MultiIndex.from_tuples(column_tuples, names=['Subcategory', 'Language'])
    
    return df

def get_robustness_styled_subcategory_df(summary_df):
    """
    Style the subcategory robustness DataFrame similar to your Register & Style table
    """
    styled_data = summary_df.copy()
    
    # Sort by overall performance (average across all subcategories/languages)
    all_numeric_cols = styled_data.select_dtypes(include=[np.number])
    styled_data[('Overall', 'Mean')] = all_numeric_cols.mean(axis=1)
    styled_data = styled_data.sort_values(('Overall', 'Mean'), ascending=True)  # Lower drops = better
    
    # Add summary statistics
    styled_data.loc["─" * 20] = np.nan
    styled_data.loc["MEAN"] = styled_data.mean(axis=0)
    styled_data.loc["STD"] = styled_data.std(axis=0)
    
    # Robustness-specific highlighting
    def highlight_robustness_subcategory(s):
        if s.dtype != "object":
            aggregation_patterns = ["MEAN", "STD", "MIN", "MAX", "─", "═"]
            
            # Filter model-only data
            model_only_series = s.copy()
            for idx in s.index:
                if any(pattern in str(idx).upper() for pattern in aggregation_patterns):
                    model_only_series = model_only_series.drop(idx, errors="ignore")
            
            if len(model_only_series) > 0:
                max_drop = model_only_series.max()  # Worst robustness
                min_drop = model_only_series.min()  # Best robustness
            else:
                return [""] * len(s)
            
            styles = []
            for idx, val in s.items():
                if any(pattern in str(idx).upper() for pattern in aggregation_patterns):
                    styles.append("")
                elif pd.isna(val):
                    styles.append("")
                else:
                    if val > 0.4:  # Very poor robustness
                        styles.append("background-color: #8B0000; font-weight: bold; color: white")
                    elif abs(val - max_drop) < 1e-4:  # Worst in column
                        styles.append("background-color: #FFB6C1; font-weight: bold; color: #8B0000")
                    elif abs(val - min_drop) < 1e-4:  # Best in column
                        styles.append("background-color: #90EE90; font-weight: bold; color: #006400")
                    elif val > 0.25:  # Poor robustness
                        styles.append("background-color: #FFA07A; color: #8B0000")
                    elif val < 0.1:  # Good robustness
                        styles.append("background-color: #F0FFF0; color: #006400")
                    else:
                        styles.append("")
            return styles
        else:
            return [""] * len(s)
    
    # Apply styling
    styled_df = (
        styled_data.style.apply(highlight_robustness_subcategory, axis=0)
        .format(precision=6)  # More precision like your example
        .set_table_styles([
            {
                "selector": "th.level0",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                    ("background-color", "#f0f0f0"),
                    ("border-right", "2px solid black")
                ]
            },
            {
                "selector": "th.level1", 
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "normal"),
                    ("background-color", "#f8f8f8")
                ]
            },
            {
                "selector": "td", 
                "props": [
                    ("text-align", "center"), 
                    ("font-size", "10px")
                ]
            }
        ])
    )
    
    return styled_df



## From notebook later
FILTERS = [
    "no_filter",
    "only_canonical_correct",
    "canonical_wrong_at_least_one_perturb_correct",
    "remains_or_becomes_correct",
]
keys = {
    "no_filter": (False, False, False),
    "only_canonical_correct": (True, False, False),
    "canonical_wrong_at_least_one_perturb_correct": (False, False, True),
    "remains_or_becomes_correct": (False, True, False),
}
def get_summaries(all_samples, subcategories: List[str] = None):
    all_tasks = all_samples["task_pretty_name"].unique()
    all_tasks = all_samples["task"].unique()
    model_names = all_samples["model_name"].unique()
    ## cleanup secondary categories
    all_samples["final_category"] = all_samples["category"].apply(
        lambda x: str(x).split(",")[0]
    )
    all_summaries = []
    for filtering_mode in FILTERS:
        #################### filtering & clean-up ####################
        # don't rely on canonical_df, perturbed_df, always use samples
        samples, canonical_df, perturbed_df = get_canonical_and_perturbed_df(
            all_samples, *keys[filtering_mode], verbose=False
        )
        ## remove duplicate results just in case they were run twice
        dup_keys = [
            "model_name",
            "task",
            "subcategories",
            "lang",
            "set_id",
            "var_id",
            "question",
        ]
        duplicate_count = (
            samples.groupby(dup_keys, as_index=False).size()["size"] > 1
        ).sum()
        print(f"N duplicates: {duplicate_count}")
        samples = samples.drop_duplicates(subset=dup_keys)
        metadata = {"filtering_mode": filtering_mode}
        #################### process results for each model ####################
        for model in model_names:
            df_filter_ = samples["model_name"] == model
            metadata["model_name"] = model
            model_df = samples[df_filter_]
            #################### process results for each model & task pair ####################
            for task in all_tasks:
                tmp = model_df[model_df["task"] == task]
                if len(tmp) == 0 and filtering_mode == "no_filter":
                    warnings.warn(
                        f"Oh noo, check for model: {model}, task: {task} if it is run, the df is empty, skipping for now..."
                    )
                if len(tmp) == 0:
                    continue
                task_pretty_name = tmp["task_pretty_name"].unique()[0]
                if subcategories and task_pretty_name not in subcategories:
                    continue
                #################### extract canonical information ####################
                task_group_name = get_group_name(task)
                cor_canonical_task_name = f"{task_group_name}_canonical"
                ## apply filtering: canonical examples, match set ids and match tasks group name
                corr_canonical = model_df[
                    model_df["is_canonical"]
                    & model_df["set_id"].isin(tmp["set_id"].unique())
                    # & model_df[model_df["task"].str.startswith(task_group_name)]
                ]
                canonical_count = len(corr_canonical)

                if len(tmp) == 0:
                    pass
                    print(f"Empty df for task {task} under filter: {filtering_mode}")
                    continue
                category = tmp["final_category"].unique()
                category = SUBCATEGORY_TO_CATEGORY.get(task_pretty_name, "")
                metadata.update(
                    {
                        "task": task,
                        "task_pretty_name": task_pretty_name,
                        "canonical_count": canonical_count,
                        "num_samples": len(tmp),
                        "category": category,
                        "langs": ",".join([l for l in tmp["lang"].unique() if l != ""]),
                    }
                )
                # compute accuracy metrics
                acc_norm = tmp["acc_norm"].mean()
                acc_norm_std = tmp["acc_norm"].std()
                acc = tmp["acc"].mean()
                acc_std = tmp["acc"].std()
                summary = {
                    "acc_norm": tmp["acc_norm"].mean(),
                    "acc_norm_std": tmp["acc_norm"].std(),
                    "acc": tmp["acc"].mean(),
                    "acc_std": tmp["acc"].std(),
                    "canonical_task_name": cor_canonical_task_name,
                    "canonical_acc": corr_canonical["acc"].mean(),
                    "canonical_acc_norm": corr_canonical["acc_norm"].mean(),
                    "group_name": task_group_name,
                }
                all_summaries.append(metadata | summary)

    all_summaries = pd.DataFrame.from_records(all_summaries)
    return all_summaries
