import random

from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


tokenizer_names = tokenizer_names = [
    "openai-community/gpt2-medium",
    "EleutherAI/pythia-160m"
]

popular_names = {
  "English": [
    "Noah", "Liam", "Oliver", "Elijah", "James",
    "William", "Benjamin", "Lucas", "Henry", "Alexander",
    "Olivia", "Amelia", "Emma", "Charlotte", "Ava",
    "Sophia", "Isabella", "Mia", "Luna", "Ellie"
  ],
  "Chinese (Mandarin)": [
    "伟", "芳", "娜", "敏", "静",
    "丽", "强", "磊", "军", "洋",
    "勇", "艳", "杰", "娟", "涛",
    "明", "超", "秀英", "霞", "平"
  ],
  "Hindi": [
    "आर्यन", "विवान", "आदित्य", "कृष्णा", "अर्जुन",
    "राहुल", "अमन", "सौरभ", "अभिषेक", "रोहित",
    "सिया", "अनन्या", "आरोही", "काव्या", "आयुषी",
    "प्रिया", "नेहा", "श्रुति", "पूजा", "साक्षी"
  ],
  "Spanish": [
    "Mateo", "Santiago", "Sebastián", "Leonardo", "Matías",
    "Emiliano", "Diego", "Daniel", "Alejandro", "Gabriel",
    "Sofía", "Valentina", "Isabella", "Camila", "Valeria",
    "Lucía", "Emma", "Martina", "Paula", "María"
  ],
  "Arabic": [
    "محمد", "أحمد", "علي", "يوسف", "عمر",
    "عبدالله", "خالد", "سعيد", "حسن", "إبراهيم",
    "فاطمة", "زينب", "مريم", "أسماء", "نور",
    "سارة", "هند", "ليلى", "جميلة", "رنا"
  ],
  "French": [
    "Gabriel", "Raphaël", "Léo", "Louis", "Maël",
    "Noah", "Jules", "Adam", "Arthur", "Isaac",
    "Louise", "Ambre", "Alba", "Jade", "Emma",
    "Rose", "Alma", "Alice", "Romy", "Anna"
  ],
  "Bengali": [
    "আর্য", "সৌরভ", "রাহুল", "সুমন", "আকাশ",
    "রাকিব", "সজল", "নাসির", "তানভীর", "রফিক",
    "সুমাইয়া", "নুসরাত", "সাবরিনা", "তানিয়া", "রিমা",
    "মেহজাবিন", "সাবিনা", "ফারহানা", "শারমিন", "সানজিদা"
  ],
  "Portuguese": [
    "Francisco", "Lourenço", "Tomás", "Vicente", "João",
    "Duarte", "Afonso", "Gabriel", "Miguel", "Santiago",
    "Maria", "Alice", "Benedita", "Matilde", "Leonor",
    "Carolina", "Aurora", "Camila", "Margarida", "Beatriz"
  ],
  "Russian": [
    "Александр", "Максим", "Дмитрий", "Артём", "Иван",
    "Михаил", "Даниил", "Егор", "Андрей", "Алексей",
    "София", "Анна", "Мария", "Ева", "Виктория",
    "Василиса", "Варвара", "Александра", "Полина", "Алиса"
  ],
  "Urdu": [
    "محمد", "احمد", "علی", "عمر", "یوسف",
    "عبداللہ", "حسن", "فہد", "کاشف", "سلمان",
    "فاطمہ", "عائشہ", "مریم", "نور", "سارہ",
    "حنا", "علیزہ", "زینب", "عائشہ", "نمرہ"
  ]
}

language_keys = ['sentence_eng_Latn', 'sentence_zho_Hans', 'sentence_deu_Latn', 'sentence_spa_Latn', 'sentence_fra_Latn', 'sentence_arb_Arab', 'sentence_por_Latn', 'sentence_rus_Cyrl', 'sentence_jpn_Jpan', 'sentence_kor_Hang', 'sentence_hin_Deva', 'sentence_ben_Beng']


def plot_tokenizer_vocab_overlap(tokenizer_names):
    """
    Plots a heatmap showing the Jaccard Index (vocabulary overlap) between different tokenizers.

    Parameters:
    - tokenizer_names: List of Hugging Face tokenizer model names as strings.

    Returns:
    - A seaborn heatmap plot of the Jaccard vocabulary overlap.
    """
    # Load vocabularies for each tokenizer
    tokenizer_vocabularies = {}
    for name in tokenizer_names:
        tokenizer = AutoTokenizer.from_pretrained(name)
        vocab = set(tokenizer.get_vocab().keys())
        tokenizer_vocabularies[name] = vocab

    # Clean tokenizer display names (remove org prefix)
    clean_names = [name.split("/")[-1] for name in tokenizer_names]

    # Compute Jaccard ratio matrix
    ratio_matrix = []
    for name1 in tokenizer_names:
        row = []
        vocab1 = tokenizer_vocabularies[name1]
        for name2 in tokenizer_names:
            vocab2 = tokenizer_vocabularies[name2]
            intersection = vocab1.intersection(vocab2)
            union = vocab1.union(vocab2)
            ratio = len(intersection) / len(union)
            row.append(ratio)
        ratio_matrix.append(row)

    # Create DataFrame with clean names
    vocab_overlap_ratio_df = pd.DataFrame(ratio_matrix, index=clean_names, columns=clean_names)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(vocab_overlap_ratio_df, annot=True, fmt=".2f", cmap="YlGnBu", square=True, 
                cbar_kws={"label": "Jaccard Ratio"})
    plt.title("Tokenizer Vocabulary Overlap (Jaccard Index)")
    plt.ylabel("Tokenizer A")
    plt.xlabel("Tokenizer B")
    plt.tight_layout()
    plt.show()

def list_tokenizer_vocab_sizes(tokenizer_names):
    """
    Prints the vocabulary size of each tokenizer from the given list.

    Parameters:
    - tokenizer_names: List of Hugging Face tokenizer model names as strings.
    """
    for model_name in tokenizer_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            vocab_size = len(tokenizer)
            print(f"Tokenizer: {model_name}")
            print(f"Vocabulary Size: {vocab_size}")
        except Exception as e:
            print(f"Error loading tokenizer {model_name}: {e}")
        print("-" * 40)

def plot_avg_tokens_per_name(tokenizer_names, popular_names, figsize=(16, 7)):
    """
    Computes and plots the average number of tokens per name for different tokenizers and languages.

    Args:
        tokenizer_names (list of str): List of HuggingFace tokenizer model names.
        popular_names (dict): Dictionary where keys are language names and values are lists of names.
        figsize (tuple): Size of the matplotlib figure.
    """
    n_token_per_lang_per_tokenizer = {}

    # Step 1: Compute average token counts
    for tokenizer_name in tokenizer_names:
        print(f"Loading {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        n_token_per_lang = {}

        for language, names in popular_names.items():
            total_tokens = sum(len(tokenizer(name).input_ids) for name in names)
            avg_tokens = total_tokens / len(names)
            n_token_per_lang[language] = avg_tokens

        n_token_per_lang_per_tokenizer[tokenizer_name] = n_token_per_lang

    # Step 2: Prepare for plotting
    languages = sorted(set(lang for d in n_token_per_lang_per_tokenizer.values() for lang in d))
    tokenizers = list(n_token_per_lang_per_tokenizer.keys())

    avg_token_counts = []
    for tokenizer in tokenizers:
        tokenizer_data = n_token_per_lang_per_tokenizer[tokenizer]
        counts = [tokenizer_data.get(lang, 0) for lang in languages]
        avg_token_counts.append(counts)

    avg_token_counts = np.array(avg_token_counts)

    # Step 3: Plot bar chart
    x = np.arange(len(languages))
    width = 0.8 / len(tokenizers)

    fig, ax = plt.subplots(figsize=figsize)
    for i, tokenizer in enumerate(tokenizers):
        ax.bar(x + i * width, avg_token_counts[i], width, label=tokenizer.split("/")[-1])

    ax.set_xlabel('Languages', fontsize=18)
    ax.set_ylabel('Avg. Tokens per Name', fontsize=18)
    ax.set_title('Average Tokens per Popular Name by Tokenizer and Language', fontsize=20)
    ax.set_xticks(x + width * (len(tokenizers) - 1) / 2)
    ax.set_xticklabels(languages, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

def compute_fertility_parity(
    tokenizer_names: list,
    language_keys: list,
    sample_size: int = 10000,
    dataset_name: str = "Muennighoff/flores200",
    dataset_config: str = "all",
    dataset_split: str = "dev",
    ref_lang: str = "sentence_eng_Latn"
):
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)

    # Filter to keep only examples with all selected languages
    examples = [ex for ex in dataset if all(lang in ex and ex[lang].strip() for lang in language_keys)]
    sampled = random.sample(examples, min(sample_size, len(examples)))

    # Prepare texts per language
    language_texts = {lang: [ex[lang] for ex in sampled] for lang in language_keys}

    # Initialize result storage
    all_results = {}

    for tokenizer_name in tokenizer_names:
        print(f"Processing tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        # --- Fertility ---
        fertility_scores = {}
        for lang, texts in language_texts.items():
            total_tokens = sum(len(tokenizer.tokenize(t)) for t in texts)
            total_words = sum(len(t.split()) for t in texts)
            fertility = total_tokens / total_words if total_words > 0 else 0
            fertility_scores[lang.replace("sentence_", "")] = fertility

        # --- Parity (relative to English) ---
        parity_scores = {}
        for lang in language_keys:
            if lang == ref_lang:
                continue
            ratios = []
            for t1, t2 in zip(language_texts[lang], language_texts[ref_lang]):
                len_t1 = len(tokenizer.tokenize(t1))
                len_t2 = len(tokenizer.tokenize(t2))
                if len_t2 > 0:
                    ratios.append(len_t1 / len_t2)
            parity = np.mean(ratios) if ratios else 0
            parity_scores[lang.replace("sentence_", "")] = parity

        # Store scores
        all_results[tokenizer_name] = {
            "fertility": fertility_scores,
            "parity_to_eng": parity_scores
        }

    return all_results

def plot_fertility_parity_scores(results):
    """
    Plots fertility and parity scores across tokenizers and languages.

    Parameters:
    - results: Dictionary in the format:
        {
            "tokenizer_name": {
                "fertility": {"lang1": value, "lang2": value, ...},
                "parity": {"lang1": value, "lang2": value, ...}
            },
            ...
        }
    """
    tokenizers = list(results.keys())
    languages = list(next(iter(results.values()))["fertility"].keys())

    fertility_data = []
    parity_data = []

    for tokenizer in tokenizers:
        fertility = results[tokenizer]["fertility"]
        parity = results[tokenizer]["parity_to_eng"]
        common_languages = [lang for lang in languages if lang in fertility and lang in parity]
        fertilities = [fertility[lang] for lang in common_languages]
        parities = [parity[lang] for lang in common_languages]
        fertility_data.append(fertilities)
        parity_data.append(parities)

    fertility_data = np.array(fertility_data)
    parity_data = np.array(parity_data)

    x = np.arange(len(common_languages))
    width = 0.1

    # Plot Fertility
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, tokenizer in enumerate(tokenizers):
        ax.bar(x + i * width, fertility_data[i], width, label=tokenizer)
    ax.set_xlabel('Languages', fontsize=18)
    ax.set_ylabel('Fertility Scores', fontsize=18)
    ax.set_title('Fertility Scores by Tokenizer and Language', fontsize=20)
    ax.set_xticks(x + width * (len(tokenizers) - 1) / 2)
    ax.set_xticklabels(common_languages, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot Parity
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, tokenizer in enumerate(tokenizers):
        ax.bar(x + i * width, parity_data[i], width, label=tokenizer)
    ax.set_xlabel('Languages', fontsize=18)
    ax.set_ylabel('Parity Scores', fontsize=18)
    ax.set_title('Parity Scores by Tokenizer and Language', fontsize=20)
    ax.set_xticks(x + width * (len(tokenizers) - 1) / 2)
    ax.set_xticklabels(common_languages, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


