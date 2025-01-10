import os

import torch
from huggingface_hub import hf_hub_download

from ichigo_whisper.config.vq_config import VQConfig
from ichigo_whisper.models.factory import make_vq_model


def load_model(
    ref,
    size: str,
    repo_id=None,
    filename=None,
    local_dir=None,
    local_filename=None,
):
    """Load model from file or Hugging Face Hub.

    Args:
        ref (str): Either a local path or "repo_id:filename" format
        repo_id (str, optional): Hugging Face repository ID
        filename (str, optional): Filename in the repository
        local_dir (str, optional): Local directory for downloads
        local_filename (str, optional): Direct path to local file

    Returns:
        RQBottleneckTransformer: Loaded model instance
    """
    # Parse reference string
    if repo_id is None and filename is None and local_filename is None:
        if ":" in ref:
            repo_id, filename = ref.split(":", 1)
        else:
            local_filename = ref

    # Download or use local file
    if not os.path.exists(f"{local_filename}"):
        local_filename = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=local_dir
        )

    # Load and validate spec
    spec = torch.load(local_filename)
    model_state_dict = {
        k.replace("model.", ""): v for k, v in spec["state_dict"].items()
    }
    vq_config = VQConfig()
    ichigo_model = make_vq_model(size=size, config=vq_config)
    ichigo_model.load_state_dict(model_state_dict)
    ichigo_model.eval()
    return ichigo_model
