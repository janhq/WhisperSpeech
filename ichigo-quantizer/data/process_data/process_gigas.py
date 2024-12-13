import gc
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value, Audio
from tqdm.auto import tqdm
from typing import Dict


# Create index
def create_audio_id_index(tsv_dataset) -> Dict[str, str]:
    return {
        row["audio_id"]: row["transcription"]
        for row in tqdm(tsv_dataset, desc="Creating index")
    }


# Load TSV dataset
tsv_dataset = load_dataset(
    "csv",
    data_files="https://huggingface.co/datasets/speechcolab/gigaspeech2/resolve/main/data/vi/train_raw.tsv",
    delimiter="\t",
    column_names=["audio_id", "transcription"],
    split="train",
)

audio_id_map = create_audio_id_index(tsv_dataset)


# Process audio files and convert to Dataset
def process_audio_to_dataset(audio_dataset):
    data = {"audio_id": [], "transcript": [], "audio": []}

    for example in tqdm(audio_dataset, desc="Processing audio files"):
        audio_id = example["__key__"].split("/")[-1]
        transcript = audio_id_map.get(audio_id)

        if transcript:
            data["audio_id"].append(audio_id)
            data["transcript"].append(transcript)
            data["audio"].append(example["wav"])

    features = Features(
        {
            "audio_id": Value("string"),
            "transcript": Value("string"),
            "audio": Audio(),  # This might preserve the audio structure better
        }
    )

    return Dataset.from_dict(data, features=features)


# Initialize an empty dataset
final_dataset = None

# Process each tar.gz file and append to the final dataset
tar_files = [
    "data/vi/train/0.tar.gz",
    "data/vi/train/1.tar.gz",
]  # Add more files as needed
for data_file in tqdm(tar_files, desc="Processing tar files"):
    # Load the audio dataset for this file
    audio_dataset = load_dataset(
        "speechcolab/gigaspeech2", data_files={"train": data_file}, split="train"
    )

    # Process this batch into a dataset
    batch_dataset = process_audio_to_dataset(audio_dataset)

    # Append to final dataset
    if final_dataset is None:
        final_dataset = batch_dataset
    else:
        final_dataset = concatenate_datasets([final_dataset, batch_dataset])

    # Clean up memory
    del audio_dataset
    del batch_dataset
    gc.collect()

# Print out
print(f"Final dataset size: {len(final_dataset)} examples")
print(final_dataset.features)

# Push to hub
final_dataset.push_to_hub("jan-hq/gigaspeech-vie-sub")
