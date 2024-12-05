from typing import List
import torch
import re
from evaluate import load


def preprocess_vietnamese_text(text: str) -> str:
    """
    Preprocess Vietnamese text by:
    - Removing extra spaces
    - Normalizing punctuation
    - Handling special cases

    Args:
        text (str): Input Vietnamese text

    Returns:
        str: Preprocessed text
    """
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    # Lower
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[.,!?;""\'\[\]\(\){}<>:،、]', '', text)
    # Trim
    return text.strip()


def tokens_to_text(tokens: torch.Tensor, tokenizer) -> str:
    """
    Convert token IDs to text using the provided tokenizer

    Args:
        tokens (torch.Tensor): Token IDs
        tokenizer: Tokenizer object with decode method

    Returns:
        str: Decoded text
    """
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(tokens.tolist())
    else:
        raise ValueError("Tokenizer must have a decode method")


def compute_wer_cer(
    predictions: List[torch.Tensor], targets: List[torch.Tensor], tokenizer
) -> tuple[float, float]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER) for Vietnamese text

    Args:
        predictions (List[torch.Tensor]): List of predicted token IDs
        targets (List[torch.Tensor]): List of target token IDs
        tokenizer: Tokenizer object for converting tokens to text

    Returns:
        tuple[float, float]: (WER, CER) scores
    """
    # Load metrics from Hugging Face evaluate
    wer_metric = load("wer")
    cer_metric = load("cer")

    # Convert tokens to text
    pred_texts = [tokens_to_text(pred, tokenizer) for pred in predictions]
    target_texts = [tokens_to_text(target, tokenizer) for target in targets]

    # Preprocess texts
    pred_texts = [preprocess_vietnamese_text(text) for text in pred_texts]
    target_texts = [preprocess_vietnamese_text(text) for text in target_texts]

    # Ensure non-empty texts
    pred_texts = [text if text.strip() else "empty" for text in pred_texts]
    target_texts = [text if text.strip() else "empty" for text in target_texts]

    try:
        # Calculate WER using Hugging Face evaluate
        wer = wer_metric.compute(references=target_texts, predictions=pred_texts)
    except:
        wer = 1.0

    try:
        # Calculate CER using Hugging Face evaluate
        cer = cer_metric.compute(references=target_texts, predictions=pred_texts)
    except:
        cer = 1.0

    return wer, cer
