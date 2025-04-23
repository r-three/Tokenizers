from typing import Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

from .data_config import InstructionFormat, TaskConfig


def format_chat_template(instruction_format: InstructionFormat, example: Dict) -> str:
    """Formats example according to an instruction format,
    taken from
    https://github.com/r-three/ft-intrinsic-dim/blob/808c281884e115c51496b4b3ae0123636bcf9cee/scripts/archive/llama_full.py#L219"""

    example_json = None

    instruction = instruction_format.instruction
    usr_query_format = instruction_format.user_format
    usr_input = instruction_format.user_input
    assistant_output = instruction_format.assistant_output

    # when there are multiple user inputs
    if isinstance(usr_input, dict):
        s = ""
        for k in usr_input:
            if isinstance(usr_input[k], dict):
                s += "\n" + k + " - "
                data_key = usr_input[k]["key_name"]
                data_field = example[data_key]  # e.g. choices
                fields = usr_input[k][
                    "fields"
                ]  # e.g. subfield of choices; labels, description
                concat_symbol = usr_input[k][
                    "concat_symbol"
                ]  # how to concat the fields

                if concat_symbol:
                    s += "\n"
                    for pair in zip(data_field[fields[0]], data_field[fields[1]]):
                        s += pair[0] + " " + concat_symbol + " " + pair[1] + "\n"
                else:
                    s += data_field[fields[0]]
            else:
                s += k + " - " + str(example[usr_input[k]]) + " [SEP] \n"

        usr_query = s.strip().strip("\n")[: -len("[SEP]")].strip()
    elif usr_query_format:
        usr_query = usr_query_format.format(**example)
    else:
        usr_query = example[usr_input]

    example_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": usr_query},
        {"role": "assistant", "content": str(example[assistant_output])},
    ]

    return example_json


def get_dataset(
    task_config: TaskConfig,
    tokenizer: AutoTokenizer,
    doc_ids: Optional[List[int]] = None,
    apply_chat_template: bool = True,
    drop_columns: bool = True,
):
    kwargs = dict()
    if task_config.split:
        kwargs["split"] = task_config.split
    if task_config.subset:
        kwargs["name"] = task_config.subset
    dataset = load_dataset(
        path=task_config.name,
        trust_remote_code=task_config.trust_remote_code,
        **kwargs,
    )
    if doc_ids:
        dataset = dataset.select(doc_ids)
    if apply_chat_template:
        if drop_columns:
            kwargs = {"remove_columns": dataset.column_names}
        instruction_format = task_config.instruction_config
        dataset = dataset.map(
            lambda x: {
                "text": tokenizer.apply_chat_template(
                    format_chat_template(instruction_format, x),
                    tokenize=False,
                    add_special_token=False,
                    add_generation_prompt=False,
                )
            },
            **kwargs,
        )
    else:
        # TODO
        pass
    return dataset
