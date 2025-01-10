from huggingface_hub import HfApi, hf_hub_download


def download_hf(
    file_name, repo_id="homebrewltd/ichigo-whisper", local_dir="checkpoints"
):
    hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=local_dir)


def upload_hf(folder_path, commit_message, repo_id="homebrewltd/ichigo-whisper"):
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    print("Checkpoint pushed to Hugging Face Hub.")
