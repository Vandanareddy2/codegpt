import os

os.environ["HF_HOME"] = "D:/huggingface_cache"      # redirect cache to D: (C: has limited space)
os.environ["HF_HUB_DISABLE_XET"] = "1"               # avoid unstable xet download protocol

from datasets import load_dataset


# Downloads (or loads from cache) the raw dataset
def load_raw_dataset():
    return load_dataset("jtatman/python-code-dataset-500k", split="train")


# Formats a batch of instruction/output pairs into Alpaca-style training strings
def format_example_batched(batch):
    texts = []
    for instruction, output in zip(batch["instruction"], batch["output"]):   # pair rows positionally
        formatted = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
        texts.append(formatted)
    return {"text": texts}    # new column: {"text": [...]}