"""
For tiktoken tokenizers, Lingua expects a .tiktoken file that can be loaded with the tiktoken library.

This script extracts the actual tokenizer file that tiktoken uses
for a specified model and saves it as a .tiktoken file you can point to directly.
"""

import argparse
import base64
from pathlib import Path

import tiktoken


def extract_tiktoken(model_name: str, output_path: str):
    """
    Extract the tokenizer from tiktoken library for a specified model.

    Args:
        model_name: Name of the model/encoding to extract (e.g., 'gpt-4o', 'o200k_base', 'cl100k_base')
        output_path: Where to save the .tiktoken file
    """
    print(f"Extracting {model_name} tokenizer from tiktoken library...")

    # Map common model names to encoding names
    model_to_encoding = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "cl100k_base",
        "text-davinci-002": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "davinci-002": "cl100k_base",
        "babbage-002": "cl100k_base",
        "gpt-3": "r50k_base",
        "davinci": "r50k_base",
        "curie": "r50k_base",
        "babbage": "r50k_base",
        "ada": "r50k_base",
        "gpt-2": "r50k_base",
    }

    # Determine encoding name
    encoding_name = model_to_encoding.get(model_name, model_name)

    try:
        print(f"Trying encoding: {encoding_name}")
        enc = tiktoken.get_encoding(encoding_name)
        tokenizer_name = encoding_name
    except Exception as e:
        print(f"Error: Could not load tokenizer '{encoding_name}': {e}")

        # Show available encodings
        print("\nAvailable encodings:")
        try:
            available = tiktoken.list_encoding_names()
            for name in sorted(available):
                print(f"  - {name}")
        except:
            print("  - o200k_base (GPT-4o)")
            print("  - cl100k_base (GPT-4, GPT-3.5)")
            print("  - r50k_base (GPT-3, GPT-2)")
            print("  - p50k_base (older models)")

        return False

    # Get the mergeable ranks (token -> rank mapping)
    mergeable_ranks = enc._mergeable_ranks

    print(f"Loaded {tokenizer_name} tokenizer")
    print(f"   Vocabulary size: {len(mergeable_ranks):,}")

    # Show some sample tokens
    sample_items = list(mergeable_ranks.items())[:5]
    print("   Sample tokens:")
    for token_bytes, rank in sample_items:
        try:
            token_str = token_bytes.decode("utf-8", errors="replace")
            print(f"     {repr(token_str)} -> {rank}")
        except:
            print(f"     {repr(token_bytes)} -> {rank}")

    # Write to tiktoken format
    output_path = Path(output_path)
    print(f"Writing to: {output_path}")

    with open(output_path, "wb") as f:
        for token_bytes, rank in sorted(mergeable_ranks.items(), key=lambda x: x[1]):
            # TikToken format: base64(token) + ' ' + str(rank) + '\n'
            token_b64 = base64.b64encode(token_bytes).decode("ascii")
            line = f"{token_b64} {rank}\n"
            f.write(line.encode("ascii"))

    print(f"Successfully created {output_path}")
    print(f"   File size: {output_path.stat().st_size:,} bytes")

    # Test the file
    print("\nTesting the created file...")
    try:
        from tiktoken.load import load_tiktoken_bpe

        test_ranks = load_tiktoken_bpe(str(output_path))
        print(f"File loads successfully with {len(test_ranks):,} tokens")

        # Test encoding
        test_text = "Hello, world!"
        tokens = enc.encode(test_text)
        decoded = enc.decode(tokens)
        print(f"Encoding test: '{test_text}' -> {tokens} -> '{decoded}'")

    except Exception as e:
        print(f"Error testing file: {e}")
        return False

    print(f"\nSuccess! Use this in your Lingua config:")
    print(f"tokenizer:")
    print(f"  name: tiktoken")
    print(f"  path: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract tiktoken file for any model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py gpt-4o
  python script.py gpt-4 --output /path/to/gpt4.tiktoken
  python script.py o200k_base --output ./tokenizer.tiktoken
  python script.py cl100k_base

Supported models/encodings:
  gpt-4o, gpt-4o-mini       -> o200k_base
  gpt-4, gpt-3.5-turbo      -> cl100k_base  
  gpt-3, gpt-2              -> r50k_base
  Or use encoding names directly: o200k_base, cl100k_base, r50k_base, p50k_base
        """,
    )

    parser.add_argument(
        "model",
        help="Model name or encoding name (e.g., 'gpt-4o', 'o200k_base', 'cl100k_base')",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output path for the .tiktoken file (default: auto-generated based on model)",
        default=None,
    )

    args = parser.parse_args()

    # Generate default output path if not provided
    if args.output is None:
        # Clean model name for filename
        clean_name = args.model.replace("-", "_").replace(".", "_")
        args.output = f"/project/aip-craffel/gsa/tokenizers/tiktoken-{clean_name}/{clean_name}_official.tiktoken"

    # Create directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = extract_tiktoken(args.model, args.output)

    if not success:
        print(f"\n💡 Alternative: Use built-in tokenizer in your config:")
        print("tokenizer:")
        print("  name: tiktoken")
        print(f"  path: {args.model}  # if it's a valid encoding name")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
