from datasets import load_dataset

dataset = load_dataset("jtatman/python-code-dataset-500k", split="train")
print(len(dataset))
print(dataset[0])