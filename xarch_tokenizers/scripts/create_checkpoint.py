"""Create a checkpoint for a specific tokenizer based on the super set vocabulary."""

import argparse
import copy
import logging
import json
import os

import torch


parser = argparse.ArgumentParser(description="Create a checkpoint for a specific tokenizer.")
parser.add_argument("--tokenizers", required=True, nargs="+")
parser.add_argument("--vocab_dir", default="vocabs")
parser.add_argument("--model_dir", default="models")
parser.add_argument("--superset", default="models/superset.pt")
parser.add_argument("--embedding_layer", default="tok_embeddings.weight")
parser.add_argument("--output_layer", default="output.weight")

logging.basicConfig(level=logging.INFO)


def select_specific(t: torch.Tensor, mapping: dict[int, int]) -> torch.Tensor:
    # Assumes that we contiguous, otherwise we need to do a gather and scatter.
    idx = [mapping[str(i)] for i in range(len(mapping))]
    # Use fancy indexing.
    return t[idx]


def main(args):

    logging.info("Loading superset model from '%s'", args.superset)
    super_set = torch.load(args.superset)
    logging.info("Extracting embedding table at '%s'", args.embedding_layer)
    super_set_embedding = super_set[args.embedding_layer]
    logging.info("Extracting output layer at '%s'", args.output_layer)
    super_set_output = super_set[args.output_layer]

    for tokenizer in args.tokenizers:
        with open(d := os.path.join(args.vocab_dir, f"{tokenizer.replace('/', '--')}_super_mapping.json")) as f:
            logging.info("Loading superset vocab mapping for %s from '%s'", tokenizer, d)
            mapping = json.load(f)

        logging.info("Extracting values from embedding table.")
        new_embedding = select_specific(super_set_embedding, mapping)
        logging.info("Extracting values from output layer.")
        new_output = select_specific(super_set_output, mapping)

        logging.info("Adding embedding/output to the checkpoint.")
        # Don't use deep copy so we have only have one copy of tensors.
        new_model = copy.copy(super_set)
        new_model[args.embedding_layer] = new_embedding
        new_model[args.output_layer] = new_output

        os.makedirs(args.model_dir, exist_ok=True)
        checkpoint = os.path.join(args.model_dir, f"{tokenizer.replace('/', '--')}.pt")
        logging.info("Saving model for %s at '%s'", tokenizer, checkpoint)
        torch.save(new_model, checkpoint)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
