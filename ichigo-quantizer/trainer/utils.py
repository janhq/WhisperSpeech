from torch.utils.data import DataLoader


def setup_dataloaders(train_dataset, val_dataset, config):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    return train_loader, val_loader


def clean_whisper_text(text: str) -> str:
    """Clean up Whisper special tokens and normalize text"""
    special_tokens = [
        "<|vi|>",
        "<|transcribe|>",
        "<|translate|>",
        "<|notimestamps|>",
        "<|nospeech|>",
        "<|endoftext|>",
        "<|pl|>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")

    # Clean up extra spaces and punctuation
    text = text.strip()
    text = " ".join(text.split())
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    return text
