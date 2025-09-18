import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
import math
import numpy as np
from datetime import datetime
import wget

# --- Plotting Utility Class for Aesthetics ---

class PlottingUtils:
    """A class to handle all plotting aesthetics and generation."""

    def __init__(self, color_map):
        self.color_map = color_map

    def apply_plot_style(self, ax=None):
        """Applies a consistent, clean style to the current plot."""
        if ax is None:
            ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)

    def create_styled_barplot(self, data, x, y, hue, title, xlabel, ylabel, filename, y_lim=(0, 1)):
        """Creates and saves a styled barplot."""
        plt.figure(figsize=(max(12, 0.5 * data[x].nunique()), 7))
        ax = sns.barplot(x=x, y=y, hue=hue, data=data, palette=self.color_map)
        for bar in ax.patches:
            bar.set_alpha(0.9)
            bar.set_edgecolor("k")
        self.apply_plot_style()
        plt.title(title, pad=40); plt.ylabel(ylabel); plt.xlabel(xlabel) # Add padding to title
        if y_lim:
            plt.ylim(y_lim)
        
        num_hue_categories = data[hue].nunique()
        plt.legend(title='Tokenizer', bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=min(num_hue_categories, 3), frameon=False)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout
        plt.savefig(filename); plt.close()
    
    def create_styled_horizontal_barplot(self, data, x, y, title, xlabel, ylabel, filename, palette='viridis'):
        """Creates and saves a styled horizontal barplot for top instances or words."""
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x=x, y=y, data=data, palette=palette)
        for bar in ax.patches:
            bar.set_alpha(0.9)
            bar.set_edgecolor("k")
        self.apply_plot_style()
        plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.tight_layout(); plt.savefig(filename); plt.close()

# --- Helper Functions ---

def clean_tokenizer_name(name):
    """Cleans the tokenizer name for legends."""
    cleaned_name = re.sub(r'.*supertoken_models-llama[_-]*', '', name)
    return cleaned_name if cleaned_name else name

def parse_canonical_file(file_path):
    """Parses the Markdown summary file."""
    if not os.path.exists(file_path): return None
    with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    canonical_correctness, in_results_table, header = {}, False, []
    for line in lines:
        if "## 📋 Detailed Question Results" in line: in_results_table = True; continue
        if not in_results_table or not line.strip().startswith('|'): continue
        parts = [p.strip() for p in line.strip().split('|')][1:-1]
        if not header and "Q#" in parts[0]: header = [clean_tokenizer_name(p) for p in parts]; continue
        if "---" in parts[0]: continue
        if header and len(parts) == len(header):
            q_num = int(parts[0])
            for i in range(3, len(header)):
                model_name, is_correct = header[i], '✅' in parts[i]
                canonical_correctness[(q_num, model_name)] = is_correct
    return canonical_correctness

def calculate_metric(group, metric):
    """Calculates the chosen metric (failure or flip rate)."""
    if metric == 'failure': return 1 - group['is_correct'].mean()
    elif metric == 'flip':
        was_canonically_correct = group['was_canonically_correct'].all()
        if not was_canonically_correct: return 0.0
        return 1 - group['is_correct'].mean()
    return 0.0

def extract_perturbation_instance(row):
    """Extracts the instance from the 'perturbation' column."""
    try: return row['perturbation']
    except (IndexError, AttributeError): return None

def compute_pis(df, group_by_cols):
    """Computes the Perturbation Impact Score (PIS) and its components."""
    word_failure_rates = df.groupby(group_by_cols + ['word_index'])['is_correct'].apply(lambda x: 1 - x.mean()).reset_index(name='failure_rate')
    grouped = word_failure_rates.groupby(group_by_cols); n_total = grouped['word_index'].nunique()
    words_with_failure = word_failure_rates[word_failure_rates['failure_rate'] > 0].groupby(group_by_cols)
    s_avg = words_with_failure['failure_rate'].mean(); f_scope = words_with_failure['word_index'].nunique() / n_total
    result = pd.DataFrame({'total_words_tested': n_total, 'words_failed': words_with_failure['word_index'].nunique(), 'avg_severity': s_avg, 'scope_factor': f_scope}).fillna(0)
    result['pis'] = result['avg_severity'] * result['scope_factor']; result['adjusted_pis'] = result['pis'] * np.log10(1 + result['total_words_tested'])
    return result.sort_values(by='adjusted_pis', ascending=False).reset_index()

# --- Analysis and Reporting Functions ---

def calculate_impact_scores(df, output_dir):
    """Calculates PIS at four different levels and saves as CSV files."""
    impact_dir = os.path.join(output_dir, 'impact_scores'); os.makedirs(impact_dir, exist_ok=True)
    print("\n--- Starting Impact Score (PIS) Analysis ---")
    
    print("Calculating PIS: Overall by Perturbation Type...")
    pis_type_overall = compute_pis(df, group_by_cols=['tokenizer', 'perturbation_type'])
    pis_type_overall.to_csv(os.path.join(impact_dir, 'pis_overall_by_type.csv'), index=False)
    
    print("Calculating PIS: Overall by Perturbation Instance...")
    pis_instance_overall = compute_pis(df, group_by_cols=['tokenizer', 'perturbation_type', 'perturbation_instance'])
    pis_instance_overall.to_csv(os.path.join(impact_dir, 'pis_overall_by_instance.csv'), index=False)
    
    print("✅ PIS analysis complete. Reports saved.")

def plot_impact_scores(output_dir, plotter):
    """Reads the PIS reports and generates summary plots."""
    impact_dir = os.path.join(output_dir, 'impact_scores')
    plots_dir = os.path.join(impact_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print("\n--- Plotting Impact Scores ---")

    try:
        df_type = pd.read_csv(os.path.join(impact_dir, 'pis_overall_by_type.csv'))
        plotter.create_styled_barplot(data=df_type, x='perturbation_type', y='adjusted_pis', hue='tokenizer',
                                      title='Adjusted Perturbation Impact Score (PIS) by Type', xlabel='Perturbation Type',
                                      ylabel='Adjusted PIS', filename=os.path.join(plots_dir, 'pis_by_type.png'), y_lim=None)
        print("Generated PIS summary plots by perturbation type.")
    except FileNotFoundError: print("Could not find pis_overall_by_type.csv, skipping type plots.")

    instance_plots_dir = os.path.join(plots_dir, 'pis_by_instance_per_type')
    os.makedirs(instance_plots_dir, exist_ok=True)

    try:
        df_instance = pd.read_csv(os.path.join(impact_dir, 'pis_overall_by_instance.csv'))
        print("Generating PIS plots by instance for each perturbation type...")
        for p_type in df_instance['perturbation_type'].unique():
            type_df = df_instance[df_instance['perturbation_type'] == p_type]
            if type_df['perturbation_instance'].nunique() > 50:
                print(f"  -> Skipping instance plots for '{p_type}': Too many unique instances")
                continue
            
            plotter.create_styled_barplot(data=type_df, x='perturbation_instance', y='adjusted_pis', hue='tokenizer',
                                          title=f'PIS for Instances of "{p_type}"', xlabel='Specific Instance',
                                          ylabel='Adjusted PIS', filename=os.path.join(instance_plots_dir, f'pis_instances_for_{p_type}.png'), y_lim=None)
            
    except FileNotFoundError: print("Could not find pis_overall_by_instance.csv, skipping instance plots.")

    try:
        df_instance = pd.read_csv(os.path.join(impact_dir, 'pis_overall_by_instance.csv'))
        for tokenizer in df_instance['tokenizer'].unique():
            top_instances = df_instance[df_instance['tokenizer'] == tokenizer].nlargest(15, 'adjusted_pis')
            if top_instances.empty: continue
            plotter.create_styled_horizontal_barplot(data=top_instances, x='adjusted_pis', y='perturbation_instance',
                                                     title=f'Top 15 Most Impactful Instances for\n{tokenizer}',
                                                     xlabel='Adjusted PIS', ylabel='Perturbation Instance',
                                                     filename=os.path.join(plots_dir, f'top_instances_{tokenizer}.png'))
        print("Generated Top 15 instance plots for each tokenizer.")
    except FileNotFoundError: print("Could not find pis_overall_by_instance.csv, skipping top instance plots.")

def calculate_and_plot_word_robustness(df, output_dir, plotter):
    """Calculates and plots the most robust words."""
    robustness_dir = os.path.join(output_dir, 'robustness_reports')
    plots_dir = os.path.join(robustness_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print("\n--- Starting Word Robustness Analysis ---")

    # Overall Robustness
    print("Calculating overall word robustness...")
    overall_robustness = df.groupby(['tokenizer', 'affected_word']).agg(
        success_rate=('is_correct', 'mean'),
        n_tests=('is_correct', 'size')
    ).reset_index()
    overall_robustness.to_csv(os.path.join(robustness_dir, 'overall_word_robustness.csv'), index=False)
    
    print("✅ Robustness analysis complete. Reports saved.")
    print("\n--- Plotting Word Robustness ---")

    # Plot top 15 overall robust words
    for tokenizer in df['tokenizer'].unique():
        top_words = overall_robustness[
            (overall_robustness['tokenizer'] == tokenizer) &
            (overall_robustness['n_tests'] >= 5) # Ensure some statistical significance
        ].nlargest(15, 'success_rate')
        
        if top_words.empty: continue
        plotter.create_styled_horizontal_barplot(data=top_words, x='success_rate', y='affected_word',
                                                 title=f'Top 15 Most Robust Words (Overall) for\n{tokenizer}',
                                                 xlabel='Success Rate', ylabel='Word',
                                                 filename=os.path.join(plots_dir, f'top_robust_words_{tokenizer}.png'),
                                                 palette='mako')
    print("Generated plots for top overall robust words.")

# --- Main Orchestration Function ---

def analyze_llm_robustness(folder_path, canonical_file, metric, filter_by_canonical_correctness, output_subfolder):
    main_output_dir = os.path.join('plots', output_subfolder)
    print(f"✅ All plots will be saved in: {main_output_dir}")
    
    canonical_data = parse_canonical_file(canonical_file) if canonical_file else None
    if (metric == 'flip' or filter_by_canonical_correctness) and not canonical_data:
        print("Error: A valid canonical file is required for this configuration."); return

    # Section 1: Data Loading and Merging
    all_data_list = []
    if not os.path.isdir(folder_path): print(f"Error: Folder does not exist: {folder_path}"); return
    files = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]
    if not files: print(f"No TSV files found in {folder_path}"); return
    for file in files:
        file_path = os.path.join(folder_path, file)
        base_name = os.path.splitext(file)[0]; tokenizer_name = clean_tokenizer_name(base_name)
        try:
            df = pd.read_csv(file_path, sep='\t'); df['tokenizer'] = tokenizer_name; all_data_list.append(df)
        except Exception as e: print(f"Error reading {file}: {e}"); continue
    if not all_data_list: print("No data processed."); return
    combined_df = pd.concat(all_data_list, ignore_index=True)
    combined_df['perturbation_instance'] = combined_df.apply(extract_perturbation_instance, axis=1)
    combined_df.dropna(subset=['perturbation_instance'], inplace=True)

    # Section 2: Setup Colors and Plotting Utility
    tokenizers = sorted(combined_df['tokenizer'].unique())
    viz_palette = [
        '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
        '#00BCD4', '#795548', '#607D8B', '#E91E63', '#8BC34A'
    ]
    colors = [viz_palette[i % len(viz_palette)] for i in range(len(tokenizers))]
    color_map = dict(zip(tokenizers, colors))
    plotter = PlottingUtils(color_map)

    # Section 3: Integrate Canonical Data
    if canonical_data:
        can_df = pd.DataFrame(canonical_data.items(), columns=['key', 'was_canonically_correct'])
        can_df[['question_num', 'tokenizer']] = pd.DataFrame(can_df['key'].tolist(), index=can_df.index)
        can_df.drop('key', axis=1, inplace=True)
        combined_df = pd.merge(combined_df, can_df, on=['question_num', 'tokenizer'], how='left')
        combined_df['was_canonically_correct'].fillna(False, inplace=True)

    # Section 4: Prediction-Level Filtering Step
    if filter_by_canonical_correctness:
        initial_rows = len(combined_df)
        combined_df = combined_df[combined_df['was_canonically_correct'] == True]
        print(f"\nFiltered out predictions where the canonical answer was incorrect. Rows reduced from {initial_rows} to {len(combined_df)}.")

    metric_name = "Flip Rate" if metric == 'flip' else "Failure Rate"
    print(f"\n--- Starting Analysis using Metric: {metric_name} ---")

    # Section 5: General, Per-Question, and Per-Word Plots
    general_plot_dir = os.path.join(main_output_dir, 'general_plots')
    instance_per_q_dir = os.path.join(main_output_dir, 'instance_plots_per_question')
    word_plots_base_dir = os.path.join(main_output_dir, 'word_plots_per_question')
    word_plots_summaries_dir = os.path.join(word_plots_base_dir, 'summaries')
    os.makedirs(general_plot_dir, exist_ok=True)
    os.makedirs(instance_per_q_dir, exist_ok=True)
    os.makedirs(word_plots_summaries_dir, exist_ok=True)
    
    # General Plot
    overall_rates = combined_df.groupby(['tokenizer', 'perturbation_type']).apply(calculate_metric, metric).reset_index(name='rate')
    plotter.create_styled_barplot(data=overall_rates, x='perturbation_type', y='rate', hue='tokenizer',
                                  title=f'Overall {metric_name} Across All Tokenizers', xlabel='Perturbation Type',
                                  ylabel=metric_name, filename=os.path.join(general_plot_dir, f'overall_comparison.png'))
    print("Generated overall comparison plot.")
    
    # Per-Question Instance Summaries
    print("\nGenerating per-question instance summary grids...")
    for q_num in sorted(combined_df['question_num'].unique()):
        q_df = combined_df[combined_df['question_num'] == q_num]
        
        instance_data_cache = {}
        for p_type in q_df['perturbation_type'].unique():
            type_df = q_df[q_df['perturbation_type'] == p_type]
            rate_instance = type_df.groupby(['tokenizer', 'perturbation_instance']).apply(calculate_metric, metric).reset_index(name='rate')
            if rate_instance.empty: continue
            instance_data_cache[p_type] = rate_instance
        
        if not instance_data_cache: continue
        
        # --- LAYOUT FIX ---
        n_plots = len(instance_data_cache)
        n_cols = 1 if n_plots == 1 else 2
        n_rows = math.ceil(n_plots / n_cols)
        # --- END FIX ---

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6), squeeze=False); axes = axes.flatten()
        
        handles, labels = None, None
        for i, (p_type, data) in enumerate(instance_data_cache.items()):
            sns.barplot(ax=axes[i], x='perturbation_instance', y='rate', hue='tokenizer', data=data, palette=color_map)
            for bar in axes[i].patches: bar.set_alpha(0.9); bar.set_edgecolor("k")
            axes[i].set_title(f'Instances of "{p_type}"'); axes[i].set_ylabel(metric_name); axes[i].set_xlabel(''); axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            plotter.apply_plot_style(ax=axes[i])
            if i == 0: handles, labels = axes[i].get_legend_handles_labels()
            if axes[i].get_legend() is not None: axes[i].get_legend().remove()
            
        for i in range(n_plots, len(axes)): fig.delaxes(axes[i])
        
        if handles:
            fig.legend(handles, labels, title='Tokenizer', bbox_to_anchor=(0.5, 1.0), loc='upper center', ncol=min(len(color_map), 3), frameon=False)
        
        # --- SPACING FIX ---
        fig.suptitle(f'Summary of {metric_name} for Instances in Question {q_num}', fontsize=20, y=1.05)
        # --- END FIX ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(instance_per_q_dir, f'q{q_num}_summary_instances.png')); plt.close()

    # Per-Word Plots and Summaries (Grouped by Question)
    print("\nGenerating per-word plots and summary grids...")
    for q_num in sorted(combined_df['question_num'].unique()):
        q_df = combined_df[combined_df['question_num'] == q_num]
        word_data_cache = {}

        # Create question-specific subfolder
        q_word_plots_dir = os.path.join(word_plots_base_dir, f'q{q_num}')
        os.makedirs(q_word_plots_dir, exist_ok=True)

        for word in sorted(q_df['affected_word'].unique()):
            word_df = q_df[q_df['affected_word'] == word]
            if len(word_df) < 5: continue
            rate_instance = word_df.groupby(['tokenizer', 'perturbation_instance']).apply(calculate_metric, metric).reset_index(name='rate')
            if rate_instance.empty or rate_instance['perturbation_instance'].nunique() < 2: continue
            word_data_cache[word] = rate_instance
            
            sanitized_word = "".join([c for c in word if c.isalnum()]).lower()
            plotter.create_styled_barplot(data=rate_instance, x='perturbation_instance', y='rate', hue='tokenizer',
                                          title=f'{metric_name} for Instances Perturbing "{word}" in Q{q_num}', 
                                          xlabel='Specific Instance', ylabel=metric_name,
                                          filename=os.path.join(q_word_plots_dir, f'word_{sanitized_word}.png'))
        
        if not word_data_cache: continue
        n_plots = len(word_data_cache); n_cols = 3; n_rows = math.ceil(n_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5), squeeze=False); axes = axes.flatten()
        
        handles, labels = None, None
        for i, (word, data) in enumerate(word_data_cache.items()):
            sns.barplot(ax=axes[i], x='perturbation_instance', y='rate', hue='tokenizer', data=data, palette=color_map)
            for bar in axes[i].patches: bar.set_alpha(0.9); bar.set_edgecolor("k")
            axes[i].set_title(f'Word: "{word}"'); axes[i].set_ylabel(metric_name); axes[i].set_xlabel('')
            axes[i].set_ylim(0, 1); axes[i].tick_params(axis='x', rotation=45, labelsize=9)
            plotter.apply_plot_style(ax=axes[i])
            if i == 0: handles, labels = axes[i].get_legend_handles_labels()
            if axes[i].get_legend() is not None: axes[i].get_legend().remove()

        for i in range(n_plots, len(axes)): fig.delaxes(axes[i])
        
        if handles:
            fig.legend(handles, labels, title='Tokenizer', bbox_to_anchor=(0.5, 1.0), loc='upper center', ncol=min(len(color_map), 3), frameon=False)
        
        fig.suptitle(f'Summary of {metric_name} for Perturbed Words in Question {q_num}', fontsize=20, y=1.03)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(word_plots_summaries_dir, f'q{q_num}_summary_words.png')); plt.close()

    # Section 6: Run the Impact Score Calculation
    calculate_impact_scores(combined_df, main_output_dir)
    
    # Section 7: Plot the Impact Scores
    plot_impact_scores(main_output_dir, plotter)

    # Section 8: Run and Plot Word Robustness
    calculate_and_plot_word_robustness(combined_df, main_output_dir, plotter)
    
    print("\n✅ Full analysis complete.")


if __name__ == "__main__":
    # --- Setup Font and Plotting Style ---
    try:
        font_path = 'AtkinsonHyperlegible-Regular.ttf'
        if not os.path.exists(font_path):
            print("Downloading Atkinson Hyperlegible font...")
            wget.download("https://github.com/googlefonts/atkinson-hyperlegible/raw/main/fonts/ttf/AtkinsonHyperlegible-Regular.ttf")
        matplotlib.font_manager.fontManager.addfont(font_path)
        plt.rcParams.update({"font.family": "Atkinson Hyperlegible", "font.size": 12})
        print("Font setup complete.")
    except Exception as e:
        print(f"Could not set up custom font, using default. Error: {e}")

    parser = argparse.ArgumentParser(description="Run a detailed analysis of LLM robustness, including Impact Scores and plots.")
    parser.add_argument("folder_path", type=str, help="Path to the folder with TSV result files.")
    parser.add_argument("--output_subfolder", type=str, default=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                        help="Name of the subfolder in 'plots/' to save results. Defaults to current timestamp.")
    parser.add_argument("--canonical_file", type=str, default=None,
                        help="Path to the Markdown file with canonical answers.")
    parser.add_argument("--metric", type=str, choices=['failure', 'flip'], default='failure',
                        help="Metric to compute for plots: 'failure' or 'flip' rate.")
    parser.add_argument("--filter_by_canonical_correctness", action='store_true',
                        help="Exclude predictions from analysis if their corresponding canonical answer was incorrect.")
    
    args = parser.parse_args()
    analyze_llm_robustness(args.folder_path, args.canonical_file, args.metric, args.filter_by_canonical_correctness, args.output_subfolder)


