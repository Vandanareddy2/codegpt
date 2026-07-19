import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
from datasets import load_dataset

dataset = load_dataset("jtatman/python-code-dataset-500k", split="train")
print(len(dataset))
example = dataset[0]
system_values = set(dataset["system"][:1000])
print(len(system_values))

def format_example_batched(batch):
    texts = []
    for instruction, output in zip(batch["instruction"], batch["output"]):
        formatted = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
        texts.append(formatted)
    return {"text": texts}

dataset = dataset.map(format_example_batched, batched=True, batch_size=1000)
print(dataset[0]["text"])

split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = split_dataset["train"]
temp_dataset = split_dataset["test"]

temp_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = temp_split["train"]
test_dataset = temp_split["test"]

print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")