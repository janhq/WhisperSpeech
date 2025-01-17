from datasets import load_dataset, DatasetDict


def split_and_push_dataset():
    # Load the full dataset
    print("Loading dataset...")
    full_dataset = load_dataset("capleaf/viVoice")
    full_data = full_dataset["train"]

    # Select only required columns and rename
    print("Processing columns...")
    full_data = full_data.select_columns(["audio", "text"])
    full_data = full_data.rename_column("text", "transcription")

    # Shuffle the dataset with a fixed seed
    print("Shuffling dataset...")
    shuffled_data = full_data.shuffle(seed=42)

    # Calculate split sizes
    total_size = len(shuffled_data)
    test_size = 10000
    val_size = 10000
    train_size = total_size - val_size - test_size

    print(f"Total samples: {total_size}")
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    # Create splits
    print("Creating splits...")
    train_dataset = shuffled_data.select(range(train_size))
    val_dataset = shuffled_data.select(range(train_size, train_size + val_size))
    test_dataset = shuffled_data.select(range(train_size + val_size, total_size))

    # Create a DatasetDict
    final_dataset = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    # Push to the Hub
    print("Pushing to Hugging Face Hub...")
    final_dataset.push_to_hub("jan-hq/viVoice")

    print("Done! Dataset has been split and uploaded successfully.")


if __name__ == "__main__":
    split_and_push_dataset()
