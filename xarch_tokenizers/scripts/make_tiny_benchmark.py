import argparse
import time
import random
from datasets import load_dataset, get_dataset_config_names, DatasetDict
from huggingface_hub import HfApi
from huggingface_hub.utils import HFValidationError

random.seed(42)

def subsample_and_push(dataset_name, split_ratio, min_samples, output_repo, delay_between_pushes=10, max_retries=5):
    api = HfApi()
    config_names = get_dataset_config_names(dataset_name)

    # If dataset has configs
    if True:
        try:
            repo_info = api.dataset_info(output_repo)

            # This is a LIST of dicts
            dataset_info = repo_info.card_data.get("dataset_info", [])

            existing_configs = []
            if isinstance(dataset_info, list):
                # Multi-config dataset
                existing_configs = [entry.get("config_name") for entry in dataset_info]
                existing_splits = []
                for entry in dataset_info:
                    for split in entry.get("splits", []):
                        existing_splits.append(split.get("name"))
            elif isinstance(dataset_info, dict):
                # Single-config dataset
                existing_configs = ["default"]
                existing_splits = [split.get("name") for split in dataset_info.get("splits", [])]


            print("Configs:", existing_configs)

        except Exception as e:
            print(f"❌ An error occurred: {e}")
            existing_configs = set()


        for config in config_names:
            config_label = config.replace("/", "__")

            if config_label in existing_configs:
                print(f"✅ Skipping existing config: {config_label}")
                continue

            print(f"\n🔹 Processing config: {config_label}")

            try:
                dataset = load_dataset(dataset_name, config)
            except Exception as e:
                print(f"❌ Failed to load config '{config}': {e}")
                continue

            new_dataset = DatasetDict()
            for split in dataset.keys():

                data = dataset[split]
                total = len(data)
                sample_size = max(int(total * split_ratio), min_samples)

                subsampled_data = data.select(random.sample(range(total), sample_size)) if sample_size < total else data
                new_dataset[split] = subsampled_data
                print(f"   → Subsampled split '{split}': {len(subsampled_data)}/{total}")

            if not new_dataset:
                print(f"⚠️ No new splits to push for config '{config_label}'.")
                continue

            for retry in range(max_retries):
                try:
                    new_dataset.push_to_hub(repo_id=output_repo, config_name=config_label)
                    print(f"✅ Successfully pushed: {output_repo} (config: {config_label})")
                    break
                except HFValidationError as e:
                    print(f"❌ Validation error for '{config_label}': {e}")
                    break
                except Exception as e:
                    wait = 2 ** retry
                    print(f"⚠️ Push error for '{config_label}': {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            else:
                print(f"❌ Gave up on config '{config_label}' after {max_retries} retries.")

            print(f"⏳ Waiting {delay_between_pushes}s before next push...")
            time.sleep(delay_between_pushes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample HF dataset and push all splits/configs as tiny benchmark")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset name (e.g., belebele)")
    parser.add_argument("--percentage", type=float, default=10.0, help="Subsampling percentage (default: 10)")
    parser.add_argument("--output_repo", type=str, required=True, help="Target HF repo to push subsampled data")
    parser.add_argument("--min_samples", type=int, default=100, help="Minimum samples per split")
    parser.add_argument("--delay_between_pushes", type=int, default=100, help="Delay between HF pushes")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries if push fails")

    args = parser.parse_args()

    subsample_and_push(
        dataset_name=args.dataset_path,
        split_ratio=args.percentage / 100.0,
        min_samples=args.min_samples,
        output_repo=args.output_repo,
        delay_between_pushes=args.delay_between_pushes,
        max_retries=args.max_retries
    )

