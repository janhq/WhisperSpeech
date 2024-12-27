import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=FutureWarning)

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import argparse

from config.trainer_config import TrainerConfig
from config.vq_config import VQConfig
from data.dataset import load_test_dataset
from models.factory import make_vq_model
from trainer.lightning_module import WhisperVQModule
from trainer.trainer import WhisperVQTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--basevq-path",
        type=str,
        help="Path to base WhisperVQ checkpoint",
    )
    parser.add_argument(
        "--test-data", type=str, required=True, help="Path to test dataset"
    )
    parser.add_argument("--model-size", type=str, required=True, help="VQ Model")
    parser.add_argument("--whisper-name", type=str, required=True, help="Whisper model")
    parser.add_argument(
        "--language", type=str, default="vi", help="Language of the data"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of samples"
    )
    return parser.parse_args()


def merge_codebooks(checkpoint_512_path, checkpoint_2048_path, save_local=False):
    """Merge codebooks from two checkpoints, putting 2049 first then 512"""

    checkpoint_512 = torch.load(checkpoint_512_path)
    checkpoint_2048 = torch.load(checkpoint_2048_path)

    state_dict_512 = (
        checkpoint_512["state_dict"]
        if "state_dict" in checkpoint_512
        else checkpoint_512
    )
    state_dict_2048 = checkpoint_2048["state_dict"]

    state_dict_512 = {k.replace("model.", ""): v for k, v in state_dict_512.items()}
    state_dict_2048 = {k.replace("model.", ""): v for k, v in state_dict_2048.items()}

    # Get codebooks
    codebook_512 = state_dict_512["rq.layers.0._codebook.embed"][
        :, :-1, :
    ]  # Remove mask token [1, 512, 64]
    codebook_2048 = state_dict_2048["rq.layers.0._codebook.embed"]  # [1, 2049, 64]

    # Create new merged state dict - start with 2048 as it has all the necessary keys
    merged_state_dict = state_dict_2048.copy()

    # Calculate new size: 2048 (with mask) + 512 (without mask) = 2561
    new_size = codebook_2048.shape[1] + codebook_512.shape[1]

    # ! Reset codebook usage to zeros
    if "_codebook_usage" in merged_state_dict:
        merged_state_dict["_codebook_usage"] = torch.zeros(
            new_size,
            dtype=merged_state_dict["_codebook_usage"].dtype,
            device=merged_state_dict["_codebook_usage"].device,
        )

    # ! Merge codebook embed
    new_codebook = torch.empty(
        (codebook_2048.shape[0], new_size, codebook_2048.shape[2]),
        dtype=codebook_2048.dtype,
        device=codebook_2048.device,
    )
    # First copy 2048 (including mask token)
    new_codebook[:, : codebook_2048.shape[1], :] = codebook_2048
    # Then copy all from 512 (excluding mask token)
    new_codebook[:, codebook_2048.shape[1] :, :] = codebook_512
    merged_state_dict["rq.layers.0._codebook.embed"] = new_codebook

    # ! Merge cluster_size
    old_cluster_size_2048 = state_dict_2048["rq.layers.0._codebook.cluster_size"]
    old_512_cluster_size = state_dict_512["rq.layers.0._codebook.cluster_size"][
        :, :-1
    ]  # Remove mask token
    new_cluster_size = torch.zeros(
        (old_cluster_size_2048.shape[0], new_size),
        dtype=old_cluster_size_2048.dtype,
        device=old_cluster_size_2048.device,
    )
    new_cluster_size[:, : old_cluster_size_2048.shape[1]] = old_cluster_size_2048
    new_cluster_size[:, old_cluster_size_2048.shape[1] :] = old_512_cluster_size
    merged_state_dict["rq.layers.0._codebook.cluster_size"] = new_cluster_size

    # ! Merge embed_avg
    old_embed_avg_2048 = state_dict_2048["rq.layers.0._codebook.embed_avg"]
    old_512_embed_avg = state_dict_512["rq.layers.0._codebook.embed_avg"][
        :, :-1, :
    ]  # Remove mask token
    new_embed_avg = torch.zeros(
        (old_embed_avg_2048.shape[0], new_size, old_embed_avg_2048.shape[2]),
        dtype=old_embed_avg_2048.dtype,
        device=old_embed_avg_2048.device,
    )
    new_embed_avg[:, : old_embed_avg_2048.shape[1], :] = old_embed_avg_2048
    new_embed_avg[:, old_embed_avg_2048.shape[1] :, :] = old_512_embed_avg
    merged_state_dict["rq.layers.0._codebook.embed_avg"] = new_embed_avg

    if save_local:
        checkpoint_2560 = checkpoint_2048.copy()
        merged_state_dict = {f"model.{k}": v for k, v in merged_state_dict.items()}
        checkpoint_2560["state_dict"] = merged_state_dict
        torch.save(checkpoint_2560, "merge-medium-vi-2d-2560c-dim64.pth")

    return merged_state_dict


def main():
    args = parse_args()

    # Create configs
    vq_config = VQConfig()
    trainer_config = TrainerConfig(
        task="evaluation", batch_size=args.batch_size, vq_config=vq_config
    )

    test_dataset = load_test_dataset(
        dataset_dir=args.test_data,
        language=args.language,
        num_samples=args.num_samples,
    )

    # ! Merge codebooks
    merged_state_dict = merge_codebooks(
        args.basevq_path,
        args.model_path,
    )  # TODO: comment this line if load merged ckpt directly

    # Load model
    model = make_vq_model(args.model_size, config=vq_config)
    model.load_state_dict(merged_state_dict)
    lightning_module = WhisperVQModule(model, trainer_config)
    model = lightning_module.model

    # TODO: uncomment this block if load merged ckpt directly
    # model = make_vq_model(args.model_size, config=vq_config)
    # lightning_module = WhisperVQModule(model, trainer_config)
    # lightning_module.load_from_checkpoint(args.model_path)
    # model = lightning_module.model

    model.setup(device="cuda", language=args.language)

    # Create trainer and get predictions
    trainer = WhisperVQTrainer(trainer_config)
    predictions_df = trainer.get_predictions(
        model, test_dataset, args.whisper_name, args.language
    )


if __name__ == "__main__":
    main()
