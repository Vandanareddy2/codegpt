from src.data.dataset_loader import load_raw_dataset, format_example_batched


def main():
    dataset = load_raw_dataset()
    print(f"Loaded {len(dataset)} rows")

    dataset = dataset.map(format_example_batched, batched=True, batch_size=1000)

    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)    # 70% train, 30% temp
    train_dataset = split_dataset["train"]
    temp_dataset = split_dataset["test"]

    temp_split = temp_dataset.train_test_split(test_size=0.5, seed=42)   # split temp 50/50 -> val/test
    val_dataset = temp_split["train"]
    test_dataset = temp_split["test"]

    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")

    train_dataset.save_to_disk("D:/AI-Projects/codegpt/data/train")
    val_dataset.save_to_disk("D:/AI-Projects/codegpt/data/val")
    test_dataset.save_to_disk("D:/AI-Projects/codegpt/data/test")

    print("Saved train/val/test splits to D:/AI-Projects/codegpt/data/")


if __name__ == "__main__":
    main()