import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import argparse

import torch

from config.trainer_config import TrainerConfig
from config.vq_config import VQConfig
from data.dataset import load_multiple_datasets
from models.factory import make_vq_model
from trainer.trainer import WhisperVQTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Training phase (1 or 2). Phase 1 uses KL loss, Phase 2 disables it",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional: Override epochs with fixed iterations",
    )
    parser.add_argument("--tunables", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--validate-every-n-steps",
        type=int,
        default=100,
    )
    parser.add_argument("--resume-from", type=str)
    parser.add_argument(
        "--load-checkpoint", type=str, help="Path to checkpoint to load"
    )
    parser.add_argument("--wandb-task-name", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=200)
    return parser.parse_args()


def load_state_dict_flexible(model, state_dict):
    """Load state dict with flexible matching"""
    model_state = model.state_dict()

    # # TODO: only for v2 -> v3 transition
    # # Map old keys to new keys
    # key_mapping = {
    #     "out_blocks.": "_out_blocks.",  # Map v2 to v3 naming
    # }

    # # Update keys based on mapping
    # for old_key in list(state_dict.keys()):
    #     for old_pattern, new_pattern in key_mapping.items():
    #         if old_pattern in old_key:
    #             new_key = old_key.replace(old_pattern, new_pattern)
    #             state_dict[new_key] = state_dict.pop(old_key)
    #             print(f"Remapping key: {old_key} -> {new_key}")

    # Remove unexpected keys
    for key in list(state_dict.keys()):
        if key not in model_state:
            print(f"Removing unexpected key: {key}")
            del state_dict[key]

    # Initialize missing keys with model's current values
    for key in model_state:
        if key not in state_dict:
            print(f"Initializing missing key: {key}")
            state_dict[key] = model_state[key]
        elif state_dict[key].shape != model_state[key].shape:
            print(
                f"Reshaping key {key}: {state_dict[key].shape} -> {model_state[key].shape}"
            )
            if "codebook" in key:
                # Handle codebook resizing
                old_size = state_dict[key].shape[1]  # 512
                new_size = model_state[key].shape[1]  # 1024
                if new_size > old_size:
                    # Create new empty tensor with target shape
                    new_tensor = torch.empty_like(model_state[key])

                    # Calculate how many times to repeat the old codebook
                    num_repeats = (
                        new_size + old_size - 1
                    ) // old_size  # ceiling division

                    # Repeat the old codebook entries with noise
                    for i in range(num_repeats):
                        start_idx = i * old_size
                        end_idx = min((i + 1) * old_size, new_size)
                        new_tensor[:, start_idx:end_idx, ...] = state_dict[key][
                            :, : end_idx - start_idx, ...
                        ]

                        # Add small noise for better learning
                        if i > 0:
                            noise = (
                                torch.randn_like(new_tensor[:, start_idx:end_idx, ...])
                                * 0.01
                            )
                            new_tensor[:, start_idx:end_idx, ...] += noise

                    state_dict[key] = new_tensor
                else:
                    # Truncate if new size is smaller
                    state_dict[key] = state_dict[key][:, :new_size, ...]
            else:
                # For non-codebook tensors, use model's initialization
                state_dict[key] = model_state[key]

    # Load the modified state dict
    model.load_state_dict(state_dict)


def main():
    args = parse_args()

    # Parse task and model size
    task_name, model_size = args.task.split()

    # Create VQ config with tunables
    vq_config = VQConfig(
        rope="--rope" in args.tunables,
        mask_embs="--mask_embs" in args.tunables,
        downsample_mean="--downsample_mean" in args.tunables,
    )

    # Create trainer config
    trainer_config = TrainerConfig(
        task=task_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        iterations=args.iterations,
        vq_config=vq_config,
        wandb_task_name=args.wandb_task_name,
        run_name=args.run_name,
        validate_every_n_steps=args.validate_every_n_steps,
        num_gpus=args.num_gpus,
        resume_from=args.resume_from,
        checkpoint_dir=f"checkpoints/{task_name}",
        phase=args.phase,
    )

    # Create model
    model = make_vq_model(model_size, config=vq_config)

    # Load checkpoint and extend codebook if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint)
        load_state_dict_flexible(model, checkpoint["state_dict"])
        model.train()

    dataset_configs = [
        {
            "dataset_dir": "capleaf/viVoice",
            "language": "vi",
            "weight": 0.1,
            "concat_samples": False,
            "max_tokens": args.max_tokens,
        },
        {
            "dataset_dir": "parler-tts/libritts_r_filtered",
            "language": "en",
            "weight": 0.9,
            "concat_samples": False,
            "max_tokens": args.max_tokens,
        },
    ]

    # Create weighted datasets
    train_dataset = load_multiple_datasets(dataset_configs, validation=False)
    val_dataset = load_multiple_datasets(dataset_configs, validation=True)

    # Create and run trainer
    trainer = WhisperVQTrainer(trainer_config)
    trainer.train(model, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
