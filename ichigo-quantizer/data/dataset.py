from typing import Tuple, Optional
from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import torch.nn.functional as F
import torchaudio
import whisper


class WhisperDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        txt_label: str = "transcription",
        model: str = "medium",
        language: str = "vi",
        num_samples: Optional[int] = None,
        task: str = "train",
    ):
        """Initialize dataset with same parameters as before"""
        self.dataset = load_dataset(dataset_dir, split=split)
        if num_samples:
            self.dataset = self.dataset.select(
                range(min(num_samples, len(self.dataset)))
            )

        self.txt_label = txt_label
        self.language = language
        self.model = model
        self.task = task
        self.max_audio_length = 30 * 16000  # 30 seconds at 16kHz
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=language, task="transcribe"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return self._process_example(example)

    def pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if len(audio) > self.max_audio_length:
            return audio[: self.max_audio_length]
        return F.pad(audio, (0, self.max_audio_length - len(audio)), value=0)

    def _process_example(
        self, example: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        tokens = tokenizer.encode(text)

        # Pad tokens
        max_tokens = (
            200 if self.task == "inference" else 20
        )  # TODO: don't hardcode this
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


def load_whisper_dataset(
    dataset_dir: str,
    txt_label: str = "transcription",
    model: str = "medium",
    language: str = "vi",
    validation: bool = False,
    num_samples: Optional[int] = None,
) -> WhisperDataset:
    split = "validation" if validation else "train"
    return WhisperDataset(
        dataset_dir=dataset_dir,
        split=split,
        txt_label=txt_label,
        model=model,
        language=language,
        num_samples=num_samples,
    )


def load_test_dataset(
    dataset_dir: str,
    txt_label: str = "transcription",
    model: str = "medium",
    language: str = "vi",
    num_samples: Optional[int] = None,
) -> WhisperDataset:
    return WhisperDataset(
        dataset_dir=dataset_dir,
        split="test",
        txt_label=txt_label,
        model=model,
        language=language,
        num_samples=num_samples,
        task="inference",
    )
