from typing import Tuple, Optional, List
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
import torch
import torch.nn.functional as F
import torchaudio
import whisper
from lightning.fabric.utilities.rank_zero import rank_zero_only


class WeightedDataset(Dataset):
    """Wrapper for dataset with weight"""

    def __init__(self, dataset: Dataset, weight: float = 1.0):
        self.dataset = dataset
        self.weight = weight

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.weight


class WhisperDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        txt_label: str = "transcription",
        language: str = "vi",
        num_samples: Optional[int] = None,
        task: str = "train",
        concat_samples: bool = True,
        max_tokens: int = 200,
    ):
        if "libritts_r_filtered" in dataset_dir:
            if split == "validation":
                self.dataset = load_dataset(dataset_dir, "clean", split="dev.clean")
            else:
                self.dataset = load_dataset(
                    dataset_dir, "clean", split="train.clean.360"
                )

            self.dataset = self.dataset.select_columns(["audio", "text_normalized"])
            self.dataset = self.dataset.rename_column(
                "text_normalized", "transcription"
            )
            if rank_zero_only.rank == 0:
                print(f"🚀 Loaded {len(self.dataset)} samples from {dataset_dir}")
        else:
            self.dataset = load_dataset(dataset_dir, split=split)
            if rank_zero_only.rank == 0:
                print(f"🚀 Loaded {len(self.dataset)} samples from {dataset_dir}")

        if num_samples:
            self.dataset = self.dataset.select(
                range(min(num_samples, len(self.dataset)))
            )

        self.txt_label = txt_label
        self.language = language
        self.task = task
        self.max_audio_length = 30 * 16000  # 30 seconds at 16kHz
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=language, task="transcribe"
        )

        self.max_tokens = max_tokens

        # Concatenate samples
        self.concat_samples = concat_samples

        if self.concat_samples:
            print("🔗 Concatenating samples to maximize usage of 30-second window")
            self.grouped_indices = self._group_samples()

            # # Process all samples once to gather statistics
            # self.stats = []

            # for idx in range(len(self)):
            #     self.__getitem__(idx)

            # # Print summary after processing all samples
            # self._print_stats_summary()

    def _get_audio_duration(self, example: dict) -> float:
        return len(example["audio"]["array"]) / example["audio"]["sampling_rate"]

    def _group_samples(self) -> List[List[int]]:
        """Group samples to maximize usage of 30-second window"""
        groups = []
        current_group = []
        current_duration = 0.0

        for idx in range(len(self.dataset)):
            duration = self._get_audio_duration(self.dataset[idx])

            if current_duration + duration <= 30.0:
                current_group.append(idx)
                current_duration += duration
            else:
                if current_group:
                    # print(
                    #     f"✓ Group {len(groups)}: {len(current_group)} samples, duration: {current_duration:.2f}s"
                    # )
                    groups.append(current_group)
                current_group = [idx]
                current_duration = duration

        if current_group:
            groups.append(current_group)

        return groups

    def __len__(self):
        if self.concat_samples:
            return len(self.grouped_indices)
        return len(self.dataset)

    def pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if len(audio) > self.max_audio_length:
            return audio[: self.max_audio_length]
        return F.pad(audio, (0, self.max_audio_length - len(audio)), value=0)

    def __getitem__(self, idx):
        if not self.concat_samples:
            # Get single sample
            example = self.dataset[idx]

            # Process audio
            samples = torch.tensor(example["audio"]["array"], dtype=torch.float32)
            if samples.dim() == 2:
                samples = samples.mean(0)

            # Resample if needed
            if example["audio"]["sampling_rate"] != 16000:
                resampler = torchaudio.transforms.Resample(
                    example["audio"]["sampling_rate"], 16000
                )
                samples = resampler(samples)

            # Normalize audio
            if samples.abs().max() > 0:
                samples = samples / samples.abs().max()

            # Pad audio
            samples = self.pad_audio(samples)

            # Create mask for attention
            mask = torch.zeros(30 * 16000 // 320, dtype=torch.bool)
            audio_frames = min(len(samples), self.max_audio_length) // 320
            mask[:audio_frames] = 1

            # Process text tokens
            tokens = list(
                self.tokenizer.sot_sequence_including_notimestamps
            ) + self.tokenizer.encode(example[self.txt_label])

            # Pad tokens
            rpad = self.max_tokens - len(tokens)

            in_ttoks = F.pad(
                torch.tensor(tokens, dtype=torch.long),
                (0, rpad),
                value=self.tokenizer.eot,
            )
            out_ttoks = F.pad(
                torch.tensor(tokens[1:] + [self.tokenizer.eot], dtype=torch.long),
                (0, rpad),
                value=-100,
            )

            return samples, mask, in_ttoks, out_ttoks

        # Get the group of samples to concatenate
        group = self.grouped_indices[idx]

        # Process and concatenate audio
        audio_samples = []
        texts = []
        total_frames = 0

        for sample_idx in group:
            example = self.dataset[sample_idx]
            # Process audio
            samples = torch.tensor(example["audio"]["array"], dtype=torch.float32)
            if samples.dim() == 2:
                samples = samples.mean(0)

            # Resample if needed
            if example["audio"]["sampling_rate"] != 16000:
                resampler = torchaudio.transforms.Resample(
                    example["audio"]["sampling_rate"], 16000
                )
                samples = resampler(samples)

            # Normalize audio
            if samples.abs().max() > 0:
                samples = samples / samples.abs().max()

            audio_samples.append(samples)
            texts.append(example[self.txt_label])
            total_frames += len(samples)

        # Concatenate audio samples
        concatenated_audio = torch.cat(audio_samples)
        concatenated_text = " ".join(texts)

        # Pad if necessary
        concatenated_audio = self.pad_audio(concatenated_audio)

        # Create mask for attention
        mask = torch.zeros(30 * 16000 // 320, dtype=torch.bool)
        audio_frames = min(len(concatenated_audio), self.max_audio_length) // 320
        mask[:audio_frames] = 1

        # Process text tokens
        tokens = list(
            self.tokenizer.sot_sequence_including_notimestamps
        ) + self.tokenizer.encode(concatenated_text)

        # Pad tokens
        rpad = self.max_tokens - len(tokens)

        in_ttoks = F.pad(
            torch.tensor(tokens, dtype=torch.long),
            (0, rpad),
            value=self.tokenizer.eot,
        )
        out_ttoks = F.pad(
            torch.tensor(tokens[1:] + [self.tokenizer.eot], dtype=torch.long),
            (0, rpad),
            value=-100,
        )

        return concatenated_audio, mask, in_ttoks, out_ttoks

    def _print_stats_summary(self):
        """Print summary statistics and save to file"""
        if not self.stats:
            return

        print("\n=== Dataset Statistics ===")
        print(f"Total groups: {len(self.grouped_indices)}")

        # Calculate averages
        avg_samples = sum(s["num_samples"] for s in self.stats) / len(self.stats)
        avg_frames = sum(s["total_frames"] for s in self.stats) / len(self.stats)
        avg_tokens = sum(s["token_length"] for s in self.stats) / len(self.stats)

        print(f"Average samples per group: {avg_samples:.2f}")
        print(f"Average frames per group: {avg_frames:.2f}")
        print(f"Average tokens per group: {avg_tokens:.2f}")

        # Save to file
        with open("dataset_stats.txt", "w") as f:
            f.write("Group Index,Num Samples,Total Frames,Text Length,Token Length\n")
            for stat in self.stats:
                f.write(
                    f"{stat['group_idx']},{stat['num_samples']},"
                    f"{stat['total_frames']},{stat['text_length']},"
                    f"{stat['token_length']}\n"
                )
        print(f"Statistics saved to dataset_stats.txt")


def load_whisper_dataset(
    dataset_dir: str,
    txt_label: str = "transcription",
    language: str = "vi",
    validation: bool = False,
    num_samples: Optional[int] = None,
    weight: float = 1.0,
    concat_samples: bool = True,
    max_tokens: int = 200,
) -> WeightedDataset:
    """Load dataset with weight"""
    split = "validation" if validation else "train"
    concat_mode = False if validation else concat_samples

    dataset = WhisperDataset(
        dataset_dir=dataset_dir,
        split=split,
        txt_label=txt_label,
        language=language,
        num_samples=num_samples,
        concat_samples=concat_mode,
        max_tokens=max_tokens,
    )
    return WeightedDataset(dataset, weight)


def load_multiple_datasets(
    dataset_configs: List[dict],
    validation: bool = False,
) -> ConcatDataset:
    """Load multiple datasets with their weights"""
    datasets = []
    for config in dataset_configs:
        dataset = load_whisper_dataset(validation=validation, **config)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def load_test_dataset(
    dataset_dir: str,
    txt_label: str = "transcription",
    language: str = "vi",
    num_samples: Optional[int] = None,
) -> WhisperDataset:
    return WhisperDataset(
        dataset_dir=dataset_dir,
        split="test",
        txt_label=txt_label,
        language=language,
        num_samples=num_samples,
        concat_samples=False,
    )