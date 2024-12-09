import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import argparse
from config.trainer_config import TrainerConfig
from config.vq_config import VQConfig
from models.factory import make_vq_model
from trainer.trainer import WhisperVQTrainer
from data.dataset import load_test_dataset
from trainer.lightning_module import WhisperVQModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model checkpoint"
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Create configs
    vq_config = VQConfig()
    trainer_config = TrainerConfig(
        task="evaluation", batch_size=args.batch_size, vq_config=vq_config
    )

    # Load model
    model = make_vq_model(args.model_size, config=vq_config)
    lightning_module = WhisperVQModule(model, trainer_config)
    lightning_module.load_from_checkpoint(args.model_path)
    model = lightning_module.model

    model.setup(device="cuda")

    test_dataset = load_test_dataset(
        dataset_dir=args.test_data,
        language=args.language,
    )

    # Create trainer and get predictions
    trainer = WhisperVQTrainer(trainer_config)
    predictions_df = trainer.get_predictions(
        model, test_dataset, args.whisper_name, args.language
    )


if __name__ == "__main__":
    main()
