import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import argparse
import torch
from config.trainer_config import TrainerConfig
from config.vq_config import VQConfig
from models.factory import make_vq_model
from trainer.trainer import WhisperVQTrainer
from data.whisper_dataset import load_whisper_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--training-data", type=str, required=True)
    parser.add_argument("--validation-data", type=str, required=True)
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
    parser.add_argument(
        "--extend-codebook", type=int, help="New size for codebook extension"
    )
    parser.add_argument("--wandb-task-name", type=str, default=None)
    return parser.parse_args()


def load_state_dict_flexible(model, state_dict):
    """Load state dict with flexible matching"""
    model_state = model.state_dict()

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
                old_size = state_dict[key].shape[1]
                new_size = model_state[key].shape[1]
                if new_size > old_size:
                    # Extend existing codebook
                    new_tensor = torch.empty_like(model_state[key])
                    new_tensor[:, :old_size, ...] = state_dict[key]
                    # Initialize new entries
                    mean = state_dict[key].mean(dim=1, keepdim=True)
                    std = state_dict[key].std(dim=1, keepdim=True)
                    noise = torch.randn_like(new_tensor[:, old_size:, ...]) * std * 0.1
                    new_tensor[:, old_size:, ...] = mean + noise
                    state_dict[key] = new_tensor
                else:
                    # Truncate
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
        iterations=args.iterations,
        training_data=[args.training_data],
        validation_data=[args.validation_data],
        vq_config=vq_config,
        wandb_task_name=args.wandb_task_name,
        validate_every_n_steps=args.validate_every_n_steps,
        num_gpus=args.num_gpus,
        resume_from=args.resume_from,
        checkpoint_dir=f"checkpoints/{task_name}",
        strategy="ddp_find_unused_parameters_true",
    )

    # Adjust extend_codebook size if using mask_embs
    if args.extend_codebook and vq_config.mask_embs:
        args.extend_codebook += 1  # Add 1 for mask token
        print(f"Adjusting codebook size to {args.extend_codebook} for mask embedding")

    # Create model
    model = make_vq_model(model_size, config=vq_config)

    # Load checkpoint and extend codebook if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint)

        if args.extend_codebook:
            print(f"Extending codebook to size {args.extend_codebook}")
            model.extend_codebook(new_size=args.extend_codebook)

        load_state_dict_flexible(model, checkpoint["state_dict"])

        model.train()

    # Create datasets
    train_dataset = load_whisper_dataset(
        dataset_dir="linhtran92/viet_bud500",
        language="vi",
        model="medium",
    )

    val_dataset = load_whisper_dataset(
        dataset_dir="linhtran92/viet_bud500",
        language="vi",
        validation=True,
        model="medium",
    )

    # Create and run trainer
    trainer = WhisperVQTrainer(trainer_config)
    trainer.train(model, train_dataset, [val_dataset])


if __name__ == "__main__":
    main()
