import regex as re
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
    # Remove all language tokens <|xx|>
    text = re.sub(r"<\|[a-z]{2}\|>", "", text)

    # Remove other special tokens
    special_tokens = [
        "<|transcribe|>",
        "<|translate|>",
        "<|notimestamps|>",
        "<|nospeech|>",
        "<|endoftext|>",
        "<|startoftranscript|>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")

    # Remove repeated quotes and dashes
    text = re.sub(r'["\-]\s*["\-]\s*["\-]\s*', "", text)

    # Remove standalone quotes and other punctuation artifacts
    text = re.sub(r'(?<!\w)[\'"]\s*|\s*[\'"](?!\w)', " ", text)
    text = re.sub(r"[,\.]\s*[,\.]\s*", ".", text)

    # Remove non-printable characters and Unicode artifacts
    text = re.sub(
        r"[^\x20-\x7E\u0080-\u00FF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]", "", text
    )

    # Clean up extra spaces and punctuation
    text = text.strip()
    text = " ".join(text.split())
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")

    # Remove trailing punctuation
    text = text.rstrip(".,\"' ")

    return text
