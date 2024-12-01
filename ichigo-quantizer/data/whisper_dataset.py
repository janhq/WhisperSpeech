from typing import Tuple, Optional, Iterator

import torch
import torch.nn.functional as F
import torchaudio
import whisper
import webdataset as wds
from torch.utils.data import IterableDataset
from datasets import load_dataset, Dataset


def load_whisper_dataset(
    dataset_dir: str,
    txt_label: str = "transcription",
    model: str = "medium",
    language: str = "vi",
    weight: float = 1,
    validation: bool = False,
    num_samples: Optional[int] = None,
) -> wds.DataPipeline:
    """
    Load and prepare a dataset for Whisper model training.

    Args:
        dataset_dir: Path to the dataset directory
        txt_label: Key for transcription text in dataset
        model: Whisper model size ("tiny", "base", "small", "medium", "large")
        language: Language code for tokenization
        weight: Dataset weight for sampling
        validation: Whether to load validation split
        num_samples: Number of samples to load (None for all)

    Returns:
        DataPipeline: Processed dataset pipeline ready for training
    """
    split = "validation" if validation else "train"
    ds = load_dataset(dataset_dir, split=split)

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    adapter_ds = WhisperDatasetAdapter(ds, language, model, txt_label)
    pipeline = wds.DataPipeline(adapter_ds, wds.shuffle(1000), wds.batched(1))

    pipeline.total_samples = len(ds)
    pipeline.weight = weight

    return pipeline


def load_test_dataset(
    dataset_dir: str,
    txt_label: str = "transcription",
    model: str = "medium",
    language: str = "vi",
    num_samples: Optional[int] = None,
) -> wds.DataPipeline:
    """
    Load test dataset for evaluation.

    Args:
        dataset_dir: Path to the dataset directory
        txt_label: Key for transcription text in dataset
        model: Whisper model size
        language: Language code for tokenization
        num_samples: Number of samples to load (None for all)

    Returns:
        DataPipeline: Processed test dataset
    """
    ds = load_dataset(dataset_dir, split="test")

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    adapter_ds = WhisperDatasetAdapter(ds, language, model, txt_label)
    pipeline = wds.DataPipeline(adapter_ds, wds.batched(1))
    pipeline.total_samples = len(ds)

    return pipeline


class WhisperDatasetAdapter(IterableDataset):
    """
    Adapter class to process audio datasets for Whisper model training.

    Handles audio preprocessing, tokenization, and formatting of inputs/outputs.
    """

    def __init__(
        self,
        dataset: Dataset,
        language: str,
        model: str,
        txt_label: str = "transcription",
    ) -> None:
        """
        Initialize the dataset adapter.

        Args:
            dataset: Input dataset
            language: Language code for tokenization
            model: Whisper model size
            txt_label: Key for transcription text in dataset
        """
        self.dataset = dataset
        self.language = language
        self.model = model
        self.txt_label = txt_label
        self.total_samples = len(dataset)
        self.weight = 1.0
        self.max_audio_length = 30 * 16000  # 30 seconds at 16kHz

    def pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Pad or trim audio to maximum length.

        Args:
            audio: Input audio tensor

        Returns:
            Processed audio tensor of fixed length
        """
        if len(audio) > self.max_audio_length:
            return audio[: self.max_audio_length]
        return F.pad(audio, (0, self.max_audio_length - len(audio)), value=0)

    def _process_example(
        self, example: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single dataset example.

        Args:
            example: Dictionary containing audio and transcription

        Returns:
            Tuple containing:
                - Processed audio samples
                - Attention mask
                - Input tokens
                - Output tokens
        """
        # Process audio
        audio_data = example["audio"]
        samples = torch.tensor(audio_data["array"], dtype=torch.float32)

        # Ensure mono audio
        if samples.dim() == 2:
            samples = samples.mean(0)

        # Resample if needed
        if audio_data["sampling_rate"] != 16000:
            resampler = torchaudio.transforms.Resample(
                audio_data["sampling_rate"], 16000
            )
            samples = resampler(samples)

        # Normalize audio
        if samples.abs().max() > 0:
            samples = samples / samples.abs().max()

        # Pad or trim
        samples = self.pad_audio(samples)

        # Create mask for attention
        mask = torch.zeros(30 * 16000 // 320, dtype=torch.bool)
        audio_frames = min(len(samples), self.max_audio_length) // 320
        mask[:audio_frames] = 1

        # Process text tokens
        text = example[self.txt_label]
        tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=self.language, task="transcribe"
        )
        tokens = list(tokenizer.sot_sequence_including_notimestamps) + tokenizer.encode(
            text
        )

        # Pad tokens
        max_tokens = 200
        rpad = max_tokens - len(tokens)
        in_ttoks = F.pad(
            torch.tensor(tokens, dtype=torch.long),
            (0, rpad),
            value=tokenizer.eot,
        )
        out_ttoks = F.pad(
            torch.tensor(tokens[1:] + [tokenizer.eot], dtype=torch.long),
            (0, rpad),
            value=-100,
        )

        return samples, mask, in_ttoks, out_ttoks

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate over the dataset.

        Yields:
            Tuple containing processed audio and text tensors
        """
        for example in self.dataset:
            try:
                yield self._process_example(example)
            except Exception as e:
                print(f"Skipping sample due to error: {e}")
                continue
