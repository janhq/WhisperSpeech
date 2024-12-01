import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import argparse
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
        help="Run validation every N training steps",
    )
    parser.add_argument("--wandb-task-name", type=str, default=None)
    return parser.parse_args()


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
    )

    # Create model
    model = make_vq_model(model_size, config=vq_config)

    # Create datasets
    train_dataset = load_whisper_dataset(
        dataset_dir="linhtran92/viet_bud500",
        language="vi",
        model="large-v3",
    )

    val_dataset = load_whisper_dataset(
        dataset_dir="linhtran92/viet_bud500",
        language="vi",
        validation=True,
        model="large-v3",
    )

    # Create and run trainer
    trainer = WhisperVQTrainer(trainer_config)
    trainer.train(model, train_dataset, [val_dataset])


if __name__ == "__main__":
    main()
