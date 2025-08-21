#!/usr/bin/env python3
"""
Script to upload Code-Switching Multilingual Dataset to Hugging Face Hub
"""

import os
import json
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value
import pandas as pd
from huggingface_hub import login, create_repo, upload_folder
from huggingface_hub import HfApi

# Configuration
DATASET_NAME = "code-switching-tokenizer-robustness"
HF_USERNAME = "Malikeh1375"  # Replace with the target HF username
DATASET_REPO = f"{HF_USERNAME}/{DATASET_NAME}"

JSON_FILE_PATH = "code_switching_dataset.json"  # Update this path

def load_json_data(json_file_path):
    """Load the JSON dataset"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_subset_data(data, language_pair, split_name):
    """Create data for a specific language pair and split"""
    split_data = data[split_name]
    
    subset_records = []
    for record in split_data:
        subset_records.append({
            'id': record['id'],
            'concept': record['concept'],
            'explanation': record['explanation'],
            'text': record[language_pair],
            'language_pair': language_pair,
            'is_code_switched': language_pair != 'en' and language_pair != 'fa' and language_pair != 'tur' and language_pair != 'ita' and language_pair != 'zho'
        })
    
    return subset_records

def create_combined_data(data, split_name):
    """Create combined data with all language pairs for a split"""
    split_data = data[split_name]
    
    combined_records = []
    language_pairs = ['en', 'en-rus', 'en-zho', 'en-jpn', 'en-deu', 'en-spa', 
                     'en-fra', 'en-ita', 'en-tha', 'en-arb', 'en-tur', 
                     'en-kor', 'en-hin', 'en-ben', 'en-tel', 'en-swa', 'fa', 'tur', 'ita', 'zho']
    
    for record in split_data:
        # Add all language variants
        for language_pair in language_pairs:
            combined_records.append({
                'id': record['id'],
                'concept': record['concept'],
                'explanation': record['explanation'],
                'text': record[language_pair],
                'language_pair': language_pair,
                'is_code_switched': language_pair != 'en' and language_pair != 'fa' and language_pair != 'tur' and language_pair != 'ita' and language_pair != 'zho'
            })
    
    return combined_records

def create_model_card():
    """Create a comprehensive model card"""
    return """---
license: cc-by-4.0
language:
- en
- ru
- zh
- ja
- de
- es
- fr
- it
- th
- ar
- tr
- ko
- hi
- bn
- te
- sw
- fa
multilinguality:
- multilingual
size_categories:
- n<1K
task_categories:
- text-generation
- text-classification
pretty_name: Code-Switching Dataset for Tokenizer Robustness Analysis
tags:
- code-switching
- tokenizer-robustness
- multilingual
- cross-lingual
- evaluation
configs:
- config_name: en
  data_files:
    - split: train
      path: en/train-*
    - split: test
      path: en/test-*
    - split: validation
      path: en/validation-*
- config_name: en-rus
  data_files:
    - split: train
      path: en-rus/train-*
    - split: test
      path: en-rus/test-*
    - split: validation
      path: en-rus/validation-*
- config_name: en-zho
  data_files:
    - split: train
      path: en-zho/train-*
    - split: test
      path: en-zho/test-*
    - split: validation
      path: en-zho/validation-*
- config_name: en-jpn
  data_files:
    - split: train
      path: en-jpn/train-*
    - split: test
      path: en-jpn/test-*
    - split: validation
      path: en-jpn/validation-*
- config_name: en-deu
  data_files:
    - split: train
      path: en-deu/train-*
    - split: test
      path: en-deu/test-*
    - split: validation
      path: en-deu/validation-*
- config_name: en-spa
  data_files:
    - split: train
      path: en-spa/train-*
    - split: test
      path: en-spa/test-*
    - split: validation
      path: en-spa/validation-*
- config_name: en-fra
  data_files:
    - split: train
      path: en-fra/train-*
    - split: test
      path: en-fra/test-*
    - split: validation
      path: en-fra/validation-*
- config_name: en-ita
  data_files:
    - split: train
      path: en-ita/train-*
    - split: test
      path: en-ita/test-*
    - split: validation
      path: en-ita/validation-*
- config_name: en-tha
  data_files:
    - split: train
      path: en-tha/train-*
    - split: test
      path: en-tha/test-*
    - split: validation
      path: en-tha/validation-*
- config_name: en-arb
  data_files:
    - split: train
      path: en-arb/train-*
    - split: test
      path: en-arb/test-*
    - split: validation
      path: en-arb/validation-*
- config_name: en-tur
  data_files:
    - split: train
      path: en-tur/train-*
    - split: test
      path: en-tur/test-*
    - split: validation
      path: en-tur/validation-*
- config_name: en-kor
  data_files:
    - split: train
      path: en-kor/train-*
    - split: test
      path: en-kor/test-*
    - split: validation
      path: en-kor/validation-*
- config_name: en-hin
  data_files:
    - split: train
      path: en-hin/train-*
    - split: test
      path: en-hin/test-*
    - split: validation
      path: en-hin/validation-*
- config_name: en-ben
  data_files:
    - split: train
      path: en-ben/train-*
    - split: test
      path: en-ben/test-*
    - split: validation
      path: en-ben/validation-*
- config_name: en-tel
  data_files:
    - split: train
      path: en-tel/train-*
    - split: test
      path: en-tel/test-*
    - split: validation
      path: en-tel/validation-*
- config_name: en-swa
  data_files:
    - split: train
      path: en-swa/train-*
    - split: test
      path: en-swa/test-*
    - split: validation
      path: en-swa/validation-*
- config_name: fa
  data_files:
    - split: train
      path: fa/train-*
    - split: test
      path: fa/test-*
    - split: validation
      path: fa/validation-*
- config_name: tur
  data_files:
    - split: train
      path: tur/train-*
    - split: test
      path: tur/test-*
    - split: validation
      path: tur/validation-*
- config_name: ita
  data_files:
    - split: train
      path: ita/train-*
    - split: test
      path: ita/test-*
    - split: validation
      path: ita/validation-*
- config_name: zho
  data_files:
    - split: train
      path: zho/train-*
    - split: test
      path: zho/test-*
    - split: validation
      path: zho/validation-*
- config_name: combined
  data_files:
    - split: train
      path: combined/train-*
    - split: test
      path: combined/test-*
    - split: validation
      path: combined/validation-*
---

# Code-Switching Dataset for Tokenizer Robustness Analysis

## Dataset Description

This dataset is designed for **tokenizer robustness testing** in multilingual and code-switching contexts. It contains identical content expressed across 16 different language variants, including pure English and 15 English-X code-switching pairs, allowing researchers to isolate tokenization effects from semantic differences when evaluating language models.

### Purpose

- **Tokenizer Comparison**: Compare how different tokenizers (BPE, SentencePiece, byte-level) handle code-switching text
- **Perplexity Testing**: Measure perplexity differences on semantically identical content with varying language mixing
- **Robustness Evaluation**: Assess model performance across different code-switching patterns
- **Cross-lingual Research**: Support studies on multilingual text processing and code-switching understanding

## Dataset Structure

### Language Pairs

The dataset includes 16 language variants:

1. **English** (`en`): Pure English baseline
2. **English-Russian** (`en-rus`): Code-switching with Cyrillic script
3. **English-Chinese** (`en-zho`): Code-switching with Simplified Chinese (Hans)
4. **English-Japanese** (`en-jpn`): Code-switching with Japanese scripts (Hiragana/Katakana/Kanji)
5. **English-German** (`en-deu`): Code-switching with German
6. **English-Spanish** (`en-spa`): Code-switching with Spanish
7. **English-French** (`en-fra`): Code-switching with French
8. **English-Italian** (`en-ita`): Code-switching with Italian
9. **English-Thai** (`en-tha`): Code-switching with Thai script
10. **English-Arabic** (`en-arb`): Code-switching with Arabic script (RTL)
11. **English-Turkish** (`en-tur`): Code-switching with Turkish
12. **English-Korean** (`en-kor`): Code-switching with Hangul script
13. **English-Hindi** (`en-hin`): Code-switching with Devanagari script
14. **English-Bengali** (`en-ben`): Code-switching with Bengali script
15. **English-Telugu** (`en-tel`): Code-switching with Telugu script
16. **English-Swahili** (`en-swa`): Code-switching with Swahili
17. **Persian** (`fa`): Pure Persian baseline`
18. **Turkish** (`tur`): Pure Turkish baseline
19. **Italian** (`ita`): Pure Italian baseline
20. **Chinese** (`zho`): Pure Chinese baseline

### Content Coverage

The dataset covers diverse topics and domains:

- **Technology**: Digital innovation, AI, software development
- **Education**: Learning systems, academic development
- **Health**: Medical care, wellness, mental health
- **Culture**: Food traditions, art, music, heritage
- **Environment**: Conservation, climate change, sustainability
- **Society**: Community, relationships, social responsibility
- **Science**: Innovation, research, discovery
- **Economics**: Finance, markets, personal economics

### Code-Switching Patterns

Each code-switched text demonstrates natural mixing patterns:

```
"Technology has become неотъемлемой частью нашей повседневной жизни. We use smartphones чтобы общаться, computers для работы, and various apps to manage наши расписания."
```

Features:
- **Intra-sentential switching**: Language changes within sentences
- **Natural flow**: Realistic code-switching patterns
- **Functional distribution**: Content words vs function words
- **Cultural adaptation**: Context-appropriate language mixing

## Dataset Splits

- **Train**: 20 samples per language variant (320 total entries)
- **Validation**: 5 samples per language variant (80 total entries)
- **Test**: 5 samples per language variant (80 total entries)
- **Total**: 30 unique concepts, 480 total text entries across all variants

## Usage Examples

### Loading Specific Language Pairs

```python
from datasets import load_dataset

# Load specific language pair
english_data = load_dataset("Malikeh1375/code-switching-tokenizer-robustness", "en")
russian_cs_data = load_dataset("Malikeh1375/code-switching-tokenizer-robustness", "en-rus")
chinese_cs_data = load_dataset("Malikeh1375/code-switching-tokenizer-robustness", "en-zho")

# Load all data combined
all_data = load_dataset("Malikeh1375/code-switching-tokenizer-robustness", "combined")
```

### Tokenizer Robustness Testing

```python
from transformers import AutoTokenizer

# Compare tokenizers on code-switching text
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Same content, different languages
english_text = english_data['train'][0]['text']
russian_cs_text = russian_cs_data['train'][0]['text']

# Tokenize and compare
en_tokens_gpt2 = tokenizer_gpt2.encode(english_text)
ru_cs_tokens_gpt2 = tokenizer_gpt2.encode(russian_cs_text)

print(f"English tokens (GPT-2): {len(en_tokens_gpt2)}")
print(f"Russian CS tokens (GPT-2): {len(ru_cs_tokens_gpt2)}")
```

### Perplexity Analysis Across Languages

```python
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
    return torch.exp(outputs.loss).item()

# Compare perplexity across language variants
results = {}
for lang_pair in ['en', 'en-rus', 'en-zho', 'en-jpn']:
    dataset = load_dataset("Malikeh1375/code-switching-tokenizer-robustness", lang_pair)
    perplexities = [calculate_perplexity(sample['text']) for sample in dataset['test']]
    results[lang_pair] = sum(perplexities) / len(perplexities)

for lang, ppl in results.items():
    print(f"{lang}: {ppl:.2f}")
```

### Filtering and Analysis

```python
# Filter by code-switching status
english_only = all_data['train'].filter(lambda x: not x['is_code_switched'])
code_switched = all_data['train'].filter(lambda x: x['is_code_switched'])

# Filter by specific topics
tech_samples = all_data['train'].filter(lambda x: 'technology' in x['concept'].lower())

# Group by language family or script
cyrillic_data = all_data['train'].filter(lambda x: x['language_pair'] == 'en-rus')
cjk_data = all_data['train'].filter(lambda x: x['language_pair'] in ['en-zho', 'en-jpn', 'en-kor'])
```

## Dataset Subsets

Each language pair is available as a separate subset:

### Monolingual
- `en`: Pure English baseline

### Code-Switching Pairs
- `en-rus`: English-Russian (Cyrillic script)
- `en-zho`: English-Chinese Simplified (Han script)
- `en-jpn`: English-Japanese (Mixed scripts)
- `en-deu`: English-German (Latin script)
- `en-spa`: English-Spanish (Latin script)
- `en-fra`: English-French (Latin script)
- `en-ita`: English-Italian (Latin script)
- `en-tha`: English-Thai (Thai script)
- `en-arb`: English-Arabic (Arabic script, RTL)
- `en-tur`: English-Turkish (Latin script with diacritics)
- `en-kor`: English-Korean (Hangul script)
- `en-hin`: English-Hindi (Devanagari script)
- `en-ben`: English-Bengali (Bengali script)
- `en-tel`: English-Telugu (Telugu script)
- `en-swa`: English-Swahili (Latin script)
- `fa`: Pure Persian baseline
- `tur`: Pure Turkish baseline
- `ita`: Pure Italian baseline
- `zho`: Pure Chinese baseline
### Combined
- `combined`: All language variants together with metadata

## Key Features for Research

### Controlled Experimental Design

- **Semantic Equivalence**: All passages express identical concepts across languages
- **Systematic Variations**: Only language mixing differs between variants
- **Balanced Complexity**: Equal conceptual difficulty across language pairs
- **Natural Patterns**: Realistic code-switching behaviors

### Tokenization Stress Points

- **Script Diversity**: Latin, Cyrillic, Arabic, CJK, Indic, Thai scripts
- **Directionality**: LTR and RTL text mixing
- **Character Encoding**: Various Unicode ranges and normalization issues
- **Subword Boundaries**: Different segmentation patterns across languages
- **Code-switching Points**: Intra-sentential language transitions

### Multilingual Robustness

- **Script Systems**: Tests handling of different writing systems
- **Language Families**: Indo-European, Sino-Tibetan, Afroasiatic, Niger-Congo
- **Morphological Complexity**: Agglutinative, fusional, analytic languages
- **Cultural Context**: Appropriate code-switching scenarios

## Applications

1. **Tokenizer Development**: Identify weaknesses in multilingual text processing
2. **Model Evaluation**: Assess cross-lingual capabilities and biases
3. **Code-switching Research**: Study natural language mixing patterns
4. **Multilingual NLP**: Improve handling of mixed-language content
5. **Educational Technology**: Better support for multilingual learners
6. **Social Media Analysis**: Process multilingual user-generated content

## Limitations

- **Template-Based**: Follows consistent structural patterns
- **Domain Scope**: Limited to general knowledge topics
- **Length Constraints**: Medium-length passages (150-200 tokens)
- **Artificial Mixing**: Some code-switching may not reflect natural usage
- **Script Coverage**: Not exhaustive of all world writing systems

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{code_switching_tokenizer_robustness_2025,
  title={Code-Switching Dataset for Tokenizer Robustness Analysis},
  author={Malikeh1375},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/Malikeh1375/code-switching-tokenizer-robustness}
}
```

## License

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
"""

def main():
    """Main function to create and upload the dataset"""
    
    if not os.path.exists(JSON_FILE_PATH):
        print(f"Error: {JSON_FILE_PATH} not found. Please save your JSON data first.")
        return
    
    data = load_json_data(JSON_FILE_PATH)
    
    # Create datasets for each language pair
    datasets = {}
    
    language_pairs = ['en', 'en-rus', 'en-zho', 'en-jpn', 'en-deu', 'en-spa', 
                     'en-fra', 'en-ita', 'en-tha', 'en-arb', 'en-tur', 
                     'en-kor', 'en-hin', 'en-ben', 'en-tel', 'en-swa', 'fa', 'tur', 'ita', 'zho']
    
    # Create individual language pair datasets
    for language_pair in language_pairs:
        train_data = create_subset_data(data, language_pair, 'train')
        val_data = create_subset_data(data, language_pair, 'validation')
        test_data = create_subset_data(data, language_pair, 'test')
        
        datasets[language_pair] = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
    
    # Create combined dataset with all language pairs
    combined_train = create_combined_data(data, 'train')
    combined_val = create_combined_data(data, 'validation') 
    combined_test = create_combined_data(data, 'test')
    
    datasets['combined'] = DatasetDict({
        'train': Dataset.from_list(combined_train),
        'validation': Dataset.from_list(combined_val),
        'test': Dataset.from_list(combined_test)
    })
    
    # Print dataset info
    print("Dataset created successfully!")
    print("\nDataset structure:")
    for subset_name, dataset_dict in datasets.items():
        print(f"\n{subset_name.upper()} subset:")
        for split_name, dataset in dataset_dict.items():
            print(f"  {split_name}: {len(dataset)} samples")
    
    # Create model card
    model_card = create_model_card()
    
    # Upload to Hugging Face Hub
    print(f"\nUplogging to Hugging Face Hub: {DATASET_REPO}")
    
    try:
        # Upload each subset separately
        for subset_name, dataset_dict in datasets.items():
            print(f"Uploading {subset_name} subset...")
            dataset_dict.push_to_hub(
                DATASET_REPO,
                config_name=subset_name,
                commit_message=f"Add {subset_name} language subset"
            )
        
        # Upload model card
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message="Add comprehensive dataset documentation"
        )
        
        print(f"\n✅ Dataset successfully uploaded to: https://huggingface.co/datasets/{DATASET_REPO}")
        print("\n📋 Available language subsets:")
        for lang_pair in language_pairs:
            lang_name = {
                'en': 'English (baseline)',
                'en-rus': 'English-Russian', 'en-zho': 'English-Chinese',
                'en-jpn': 'English-Japanese', 'en-deu': 'English-German',
                'en-spa': 'English-Spanish', 'en-fra': 'English-French',
                'en-ita': 'English-Italian', 'en-tha': 'English-Thai',
                'en-arb': 'English-Arabic', 'en-tur': 'English-Turkish',
                'en-kor': 'English-Korean', 'en-hin': 'English-Hindi',
                'en-ben': 'English-Bengali', 'en-tel': 'English-Telugu',
                'en-swa': 'English-Swahili', 'fa': 'Persian (baseline)',
                'tur': 'Turkish (baseline)','ita': 'Italian (baseline)', 
                'zho': 'Chinese (baseline)'
            }.get(lang_pair, lang_pair)
            print(f"  - {lang_pair}: {lang_name}")
        print("  - combined: All languages together")
        
        print(f"\n🚀 Usage example:")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('{DATASET_REPO}', 'en-rus')")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\n💡 Make sure you have:")
        print("1. Installed huggingface_hub: pip install huggingface_hub datasets")
        print("2. Logged in: huggingface-cli login")
        print("3. Updated HF_USERNAME in the script")
        print("4. Your JSON file exists at the specified path")

if __name__ == "__main__":
    main()