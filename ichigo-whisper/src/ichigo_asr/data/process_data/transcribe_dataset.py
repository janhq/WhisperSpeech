import torch
from datasets import load_dataset, Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_whisper_model():
    # Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3"
    ).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="vi", task="transcribe"
    )
    return model, processor, device


def process_batch(batch, model, processor, device):
    # Process audio inputs
    input_features = processor(
        [x["array"] for x in batch["audio"]],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    ).input_features

    # Pad or truncate to expected length (3000)
    batch_size, _, seq_len = input_features.shape
    target_length = 3000

    if seq_len < target_length:
        # Pad
        padding = torch.zeros(batch_size, 128, target_length - seq_len)
        input_features = torch.cat([input_features, padding], dim=2)
    elif seq_len > target_length:
        # Truncate
        input_features = input_features[:, :, :target_length]

    input_features = input_features.to(device)

    # Generate transcriptions
    with torch.no_grad():
        generated_ids = model.generate(input_features, max_length=225)

    # Decode transcriptions
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(transcriptions)
    return transcriptions


def process_dataset_split(dataset_split, model, processor, device, batch_size=64):
    # Create a custom collate function to handle None values
    def collate_fn(batch):
        return {"audio": [item["audio"] for item in batch if item["audio"] is not None]}

    # Use the custom collate function in the DataLoader
    dataloader = DataLoader(dataset_split, batch_size=batch_size, collate_fn=collate_fn)
    whisper_transcriptions = []

    for batch in tqdm(dataloader):
        # Skip empty batches
        if not batch["audio"]:
            continue
        transcriptions = process_batch(batch, model, processor, device)
        whisper_transcriptions.extend(transcriptions)

    # Add new column to dataset
    new_dataset = dataset_split.add_column(
        "whisper_transcription", whisper_transcriptions
    )
    return new_dataset


def main():
    # Load model
    model, processor, device = load_whisper_model()

    # Load dataset
    dataset = load_dataset("linhtran92/viet_bud500")

    # Process each split
    processed_splits = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        processed_splits[split] = process_dataset_split(
            dataset[split], model, processor, device
        )

    # Push to hub
    processed_dataset = Dataset.from_dict(processed_splits)
    processed_dataset.push_to_hub("jan-hq/viet_bud500_high_quality")


if __name__ == "__main__":
    main()
