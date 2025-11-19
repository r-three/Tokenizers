import json

import numpy as np
import pandas as pd


## for notebook views
# Modified function to exclude aggregation rows
def highlight_extremes(s):
    """
    Highlight the max value in green and min value in red for each column
    Excludes aggregation rows from min/max calculation
    """
    if s.dtype != "object":  # Only apply to numeric columns
        # Define patterns that identify aggregation rows
        aggregation_patterns = [
            "MEAN",
            "STD",
            "MIN",
            "MAX",
            "MEDIAN",
            "RANGE",
            "TOP_3_AVG",
            "BOTTOM_3_AVG",
            "TASK_WEIGHTED_AVG",
            "BEST_MODEL_PER_TASK",
            "WORST_MODEL_PER_TASK",
            "─",
            "═",
            "___",  # Visual separators
        ]

        # Filter out aggregation rows for min/max calculation
        model_only_series = s.copy()
        model_indices_to_exclude = []

        for idx in s.index:
            # Check if index matches any aggregation pattern
            if any(pattern in str(idx).upper() for pattern in aggregation_patterns):
                model_indices_to_exclude.append(idx)

        # Remove aggregation rows from calculation
        model_only_series = model_only_series.drop(
            model_indices_to_exclude, errors="ignore"
        )

        # Calculate min/max only from model rows
        if len(model_only_series) > 0:
            max_val = model_only_series.max()
            min_val = model_only_series.min()
        else:
            max_val = None
            min_val = None

        styles = []
        for idx, val in s.items():
            # Don't highlight aggregation rows at all
            if any(pattern in str(idx).upper() for pattern in aggregation_patterns):
                styles.append("")
            elif pd.isna(val):
                styles.append("")
            elif max_val is not None and val == max_val:
                styles.append(
                    "background-color: #90EE90; font-weight: bold; color: #006400"
                )  # Light green bg, dark green text
            elif min_val is not None and val == min_val:
                styles.append(
                    "background-color: #FFB6C1; font-weight: bold; color: #8B0000"
                )  # Light red bg, dark red text
            else:
                styles.append("")
        return styles
    else:
        return [""] * len(s)


def weighted_avg(group):
    """
    Calculate weighted average from a group that has both acc_norm and num_samples
    """
    if len(group) == 0:
        return "___"

    # group is a DataFrame with both columns
    acc_values = group["acc_norm"]
    weights = group["num_samples"]

    # Remove NaN values
    valid_mask = ~(pd.isna(acc_values) | pd.isna(weights)) & (weights > 0)

    if valid_mask.sum() > 0:
        return np.average(acc_values[valid_mask], weights=weights[valid_mask])
    else:
        return "___"


def get_styled_df(
    summaries,
    columns=["task_pretty_name", "langs"],
    index=["model_name"],
):
    pivot_df = (
        summaries.groupby([*index, *columns])
        .apply(weighted_avg)
        .reset_index()  # Convert back to DataFrame
        .rename(columns={0: "weighted_avg"})  # Name the result column
        .pivot_table(index=index, columns=columns, values="weighted_avg")
    )

    # Add aggregate columns
    pivot_df["Mean"] = pivot_df.mean(axis=1)
    # Sort by mean performance
    pivot_df = pivot_df.sort_values("Mean", ascending=False)

    # add column-wise stats
    pivot_df.loc["─" * 20] = np.nan  # Separator row
    pivot_df.loc["MEAN"] = pivot_df.mean(axis=0)
    pivot_df.loc["MAX"] = pivot_df.max(axis=0)
    pivot_df.loc["MIN"] = pivot_df.min(axis=0)
    pivot_df.loc["STD"] = pivot_df.std(axis=0)

    styled_df = (
        pivot_df.style.apply(highlight_extremes, axis=0)
        .format(precision=3)
        .set_table_styles(
            [
                {
                    "selector": "th.level0",
                    "props": [
                        ("border-right", "3px solid black"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [("text-align", "center"), ("font-size", "11px")],
                },
            ]
        )
    )
    styled_df = styled_df.format(na_rep="─────")
    return styled_df



def get_robustness_styled_df(jsonl_path="tokenization_results.jsonl"):
    """
    Create styled dataframe for robustness drops with highlighting for largest drops
    """
    records = []
    with open(jsonl_path, "r") as file:
        for line in file:
            records.append(json.loads(line))
    summary_df = pd.DataFrame.from_records(records)
    summary_df = summary_df.set_index("model_name")
    styled_data = summary_df.copy()
    
    # Add average column (mean robustness drop)
    styled_data['Average'] = styled_data.mean(axis=1, skipna=True, numeric_only=True)
    
    # Sort by average robustness (lowest drops first = best robustness)
    styled_data = styled_data.sort_values('Average', ascending=True)
    
    # Add summary statistics rows
    styled_data.loc["─" * 20] = np.nan
    styled_data.loc["MEAN"] = styled_data.mean(axis=0, numeric_only=True)
    # styled_data.loc["MAX"] = styled_data.max(axis=0, numeric_only=True)
    # styled_data.loc["MIN"] = styled_data.min(axis=0, numeric_only=True)
    styled_data.loc["STD"] = styled_data.std(axis=0, numeric_only=True)
    # styled_data.loc["RANGE"] = styled_data.max(axis=0, numeric_only=True) - styled_data.min(axis=0, numeric_only=True)
    
    # Custom highlighting function for robustness drops
    def highlight_robustness_drops(s):
        """
        Highlight robustness drops: 
        - Green for smallest drops (best robustness)
        - Red for largest drops (worst robustness)
        - Darker red for very large drops (> 0.5)
        """
        if s.dtype != "object":
            # Define patterns that identify aggregation rows
            aggregation_patterns = [
                "MEAN", "STD", "MIN", "MAX", "MEDIAN", "RANGE",
                "TOP_3_AVG", "BOTTOM_3_AVG", "TASK_WEIGHTED_AVG",
                "BEST_MODEL_PER_TASK", "WORST_MODEL_PER_TASK",
                "─", "═", "___",
            ]
            
            # Filter out aggregation rows for min/max calculation
            model_only_series = s.copy()
            model_indices_to_exclude = []
            for idx in s.index:
                if any(pattern in str(idx).upper() for pattern in aggregation_patterns):
                    model_indices_to_exclude.append(idx)
            
            model_only_series = model_only_series.drop(model_indices_to_exclude, errors="ignore")
            
            if len(model_only_series) > 0:
                max_drop = model_only_series.max()  # Worst robustness
                min_drop = model_only_series.min()  # Best robustness
            else:
                max_drop = None
                min_drop = None
            
            styles = []
            for idx, val in s.items():
                # Don't highlight aggregation rows
                if any(pattern in str(idx).upper() for pattern in aggregation_patterns):
                    styles.append("")
                elif pd.isna(val):
                    styles.append("")
                else:
                    # if val > 0.5:  # Very large drops (>50% performance loss)
                    #     styles.append("background-color: #8B0000; font-weight: bold; color: white")  # Dark red
                    # el
                    if max_drop is not None and abs(val - max_drop) < 1e-4:  # Largest drop in column
                        styles.append("background-color: #FFB6C1; font-weight: bold; color: #8B0000")  # Light red
                    elif min_drop is not None and abs(val - min_drop) < 1e-4:  # Smallest drop in column
                        styles.append("background-color: #90EE90; font-weight: bold; color: #006400")  # Light green
                    # elif val > 0.3:  # Moderate drops (>30% performance loss)
                    #     styles.append("background-color: #FFA07A; color: #8B0000")  # Light salmon
                    # elif val < 0.1:  # Small drops (<10% performance loss)
                    #     styles.append("background-color: #F0FFF0; color: #006400")  # Very light green
                    else:
                        styles.append("")
            return styles
        else:
            return [""] * len(s)
    
    # Apply styling
    styled_df = (
        styled_data.style.apply(highlight_robustness_drops, axis=0)
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                    ("background-color", "#f0f0f0")
                ]
            },
            {
                "selector": "td", 
                "props": [
                    ("text-align", "center"), 
                    ("font-size", "11px")
                ]
            }
        ])
        .format(precision=2)
    )
    
    styled_df = styled_df.format(na_rep="─────").format(precision=2)
    return styled_df
