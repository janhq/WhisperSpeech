from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="jan-hq/ichigo-quantizer", exist_ok=True)
api.upload_folder(
    folder_path="/root/WhisperSpeech/ichigo-quantizer/epoch=21-step=5478-val",
    repo_id="jan-hq/ichigo-quantizer",
    commit_message="Upload Quantizer Checkpoint",
)
print("Checkpoint pushed to Hugging Face Hub.")
