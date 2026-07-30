from datasets import load_from_disk

train = load_from_disk("data/train")
val = load_from_disk("data/val")
test = load_from_disk("data/test")

print(f"Train: {len(train)} rows")
print(f"Val: {len(val)} rows")
print(f"Test: {len(test)} rows")

print("\nSample from train:")
print(train[0]["text"][:300])