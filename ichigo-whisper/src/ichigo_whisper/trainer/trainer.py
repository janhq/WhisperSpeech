import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import whisper
from evaluate import load
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import pipeline

import wandb
from config.trainer_config import TrainerConfig
from trainer.lightning_module import WhisperVQModule
from trainer.utils import clean_whisper_text


class WhisperVQTrainer:
    """
    Trainer class for WhisperVQ model using PyTorch Lightning.

    This trainer handles the complete training pipeline including:
    - Environment setup
    - Weights & Biases logging
    - Checkpoint management
    - Multi-GPU training
    - Model saving

    Args:
        config (TrainerConfig): Configuration object containing training parameters
            including learning rates, batch sizes, and training iterations.

    Attributes:
        config (TrainerConfig): Stored configuration object
        run_name (str): Unique name for the training run
        wandb_logger (WandbLogger): Weights & Biases logger instance
        callbacks (list): List of PyTorch Lightning callbacks
        trainer (pl.Trainer): PyTorch Lightning trainer instance
    """

    def __init__(self, config: TrainerConfig):
        """
        Initialize the WhisperVQ trainer with the given configuration.

        Args:
            config (TrainerConfig): Training configuration object
        """
        self.config = config
        self._setup_environment()
        self._setup_wandb()
        self._setup_callbacks()
        self._setup_trainer()

    def _setup_environment(self):
        """
        Configure PyTorch environment settings.
        Sets float32 matmul precision to "medium" for better performance.
        """
        torch.set_float32_matmul_precision("medium")

    def _setup_wandb(self):
        """
        Initialize Weights & Biases logging.

        Sets up the project name with optional suffix and generates a unique run name.
        The project name format is: WhisperSpeech-{task_name}[-{suffix}]
        Logs configuration parameters for experiment tracking.
        """
        project = f"{self.config.wandb_task_name or self.config.task}"
        if self.config.wandb_suffix:
            project += f"-{self.config.wandb_suffix}"

        self.run_name = self.config.run_name
        self.wandb_logger = WandbLogger(project=project, name=self.run_name)

        # Log configuration parameters
        config_dict = {
            # Training parameters
            "training": self.config.to_hparams(),
            # VQ specific parameters
            "vq": {
                "init_std": self.config.vq_config.init_std,
                "embeddings_std": self.config.vq_config.embeddings_std,
                "embeddings_lr_scale": self.config.vq_config.embeddings_lr_scale,
                "query_mult": self.config.vq_config.query_mult,
                "rope": self.config.vq_config.rope,
                "mask_embs": self.config.vq_config.mask_embs,
                "output_mult": self.config.vq_config.output_mult,
                "downsample_conv": self.config.vq_config.downsample_conv,
                "downsample_mean": self.config.vq_config.downsample_mean,
                "codebook_dim": self.config.vq_config.codebook_dim,
                "codebook_decay": self.config.vq_config.codebook_decay,
            },
            # Dataset parameters
            "dataset": {
                "training_data": self.config.training_data,
                "validation_data": self.config.validation_data,
                "dataset_config": self.config.dataset_config,
            },
            # Hardware settings
            "hardware": {
                "num_workers": self.config.num_workers,
                "precision": self.config.precision,
                "torch_compile": self.config.torch_compile,
                "strategy": self.config.strategy,
                "num_gpus": self.config.num_gpus,
            },
        }

        # self.wandb_logger.experiment.config.update(config_dict) #TODO: trick bypass ddp

    def _setup_callbacks(self):
        """
        Initialize PyTorch Lightning callbacks.

        Sets up:
        1. ModelCheckpoint for best validation loss
        2. ModelCheckpoint for periodic epoch saves
        3. LearningRateMonitor
        4. EarlyStopping
        """
        self.callbacks = [
            # Best checkpoint based on validation loss
            ModelCheckpoint(
                dirpath=self.config.checkpoint_dir,
                filename=f"{self.config.task}/{self.run_name}/best-{{epoch}}-{{step}}-{{val/epoch_accuracy:.5f}}",
                monitor="val/epoch_accuracy",
                save_top_k=1,
                mode="max",
                save_on_train_epoch_end=False,
                verbose=True if rank_zero_only.rank == 0 else False,
            ),
            # Periodic checkpoint every epoch
            ModelCheckpoint(
                dirpath=self.config.checkpoint_dir,
                filename=f"{self.config.task}/{self.run_name}/epoch-{{epoch}}-{{step}}-{{val/epoch_accuracy:.5f}}",
                save_top_k=-1,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            # EarlyStopping(
            #     monitor="val/epoch_accuracy",
            #     patience=self.config.early_stopping_patience,
            #     mode="max",
            #     verbose=True if rank_zero_only.rank == 0 else False,
            # ),
        ]

    def _setup_trainer(self):
        """Initialize PyTorch Lightning trainer."""
        trainer_kwargs = {
            "strategy": self.config.strategy,
            "accelerator": "gpu",
            "precision": self.config.precision,
            "gradient_clip_val": self.config.vq_config.clip_gradient_norm,
            "accumulate_grad_batches": self.config.accumulate_grad_batches,
            "logger": self.wandb_logger,
            "callbacks": self.callbacks,
            "num_nodes": int(os.environ.get("SLURM_NNODES", 1)),
            "devices": int(self.config.num_gpus),
            "log_every_n_steps": 1,
        }

        # Configure validation frequency based on training mode
        if self.config.iterations:
            # Iteration-based training
            trainer_kwargs["max_steps"] = self.config.iterations
            if self.config.validate_every_n_steps:
                # Use check_val_every_n_epoch=None to enable step-based validation
                trainer_kwargs["check_val_every_n_epoch"] = None
                trainer_kwargs["val_check_interval"] = (
                    self.config.validate_every_n_steps
                )
        else:
            # Epoch-based training
            trainer_kwargs["max_epochs"] = self.config.epochs
            # Validate once per epoch (default behavior)
            trainer_kwargs["check_val_every_n_epoch"] = 1

        self.trainer = pl.Trainer(**trainer_kwargs)

    def train(self, model, train_dataset, val_datasets):
        """
        Train the WhisperVQ model.

        Args:
            model: The WhisperVQ model to train
            train_dataset: Dataset for training
            val_datasets: Dataset(s) for validation

        The method:
        1. Sets up data loaders
        2. Wraps the model in a Lightning module
        3. Executes the training
        4. Saves the final model (on rank 0 only)

        #!  What happens during training example:
        1. Total samples = 700K (all samples are included)
        2. For each training step:
        - Vietnamese samples have 0.7 probability of being selected
        - English samples have 0.3 probability of being selected

        # Approximate sampling distribution:
        - Vietnamese: (500K * 0.7) / (500K * 0.7 + 200K * 0.3) ≈ 85% chance
        - English: (200K * 0.3) / (500K * 0.7 + 200K * 0.3) ≈ 15% chance
        """
        # Add statistics printing at the start
        if isinstance(train_dataset, ConcatDataset):
            total_samples = sum(
                len(dataset.dataset) for dataset in train_dataset.datasets
            )
            if rank_zero_only.rank == 0:
                print("\n=== Dataset Statistics ===")
                for i, dataset in enumerate(train_dataset.datasets):
                    weight = dataset.weight
                    size = len(dataset.dataset)
                    effective_ratio = (size * weight) / sum(
                        len(d.dataset) * d.weight for d in train_dataset.datasets
                    )
                    print(f"Dataset {i}:")
                    print(f"  - Size: {size:,} samples")
                    print(f"  - Weight: {weight}")
                    print(f"  - Effective sampling ratio: {effective_ratio:.1%}")
                print(f"Total samples available: {total_samples:,}")
                print("=====================\n")

        # Train DataLoader
        if isinstance(train_dataset, ConcatDataset):
            weights = []
            dataset_sizes = []
            for dataset in train_dataset.datasets:
                weight = getattr(dataset, "weight", 1.0)
                size = len(dataset)
                weights.extend([weight] * size)
                dataset_sizes.append(size)

            weights = torch.DoubleTensor(weights)
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(weights), replacement=True
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        # Test DataLoader
        if isinstance(val_datasets, (list, tuple)):
            val_loaders = [
                DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                )
                for val_dataset in val_datasets
            ]
        else:
            val_loaders = DataLoader(
                val_datasets,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        train_dataset_size = len(train_dataset)

        lightning_module = WhisperVQModule(
            model,
            self.config,
            train_dataset_size=train_dataset_size,
            phase=self.config.phase,
        )

        # Phase 2: Load state dict directly if resuming
        if self.config.phase == 2 and self.config.resume_from:
            lightning_module.load_state_dict(
                torch.load(self.config.resume_from)["state_dict"], strict=False
            )

        self.trainer.fit(
            model=lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loaders,
            ckpt_path=(
                self.config.resume_from
                if self.config.phase == 1
                and self.config.resume_from
                is not None  # TODO: extend this later to resume training from a checkpoint
                else None
            ),
        )

        if rank_zero_only.rank == 0:
            self._save_model(model)

    def get_predictions(self, model, test_dataset, whisper_name, language):
        # ! Whisper Medium
        whisper_model = whisper.load_model(whisper_name)
        whisper_model.to("cuda")

        # ! PhoWhisper
        phowhisper = pipeline(
            "automatic-speech-recognition",
            model="vinai/PhoWhisper-large",
            device="cuda",
        )

        # ! Quantizer
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        model.eval()
        model = model.cuda()

        # ! Tracking
        columns = [
            "audio_id",
            "ground_truth",
            "predicted_output",
            "phowhisper_output",
            "whisper_output",
            "model_wer",
            "phowhisper_wer",
            "whisper_wer",
        ]
        predictions_table = wandb.Table(columns=columns)

        results = []
        wer_metric = load("wer")

        progress_bar = tqdm(
            total=len(test_dataset), desc="Generating predictions", unit="samples"
        )

        # Store all tokens for distribution analysis
        all_tokens = []

        with torch.no_grad():
            for batch_idx, (samples, output_toks) in enumerate(test_loader):
                samples = samples.cuda()
                decoded_results = model.inference(samples)

                # Collect tokens
                all_tokens.append(model.stoks_id.flatten())

                for i in range(len(samples)):
                    gt_tokens = output_toks[i][output_toks[i] != -100]
                    ground_truth = model.tokenizer.decode(gt_tokens.tolist())
                    ground_truth = clean_whisper_text(ground_truth)

                    # ! Process model predictions
                    pred_text = clean_whisper_text(decoded_results[i].text)

                    #! Process Whisper predictions
                    audio_sample = samples[i].cpu().numpy()
                    whisper_result = whisper_model.transcribe(
                        audio_sample,
                        language=language,
                        task="transcribe",
                        fp16=False,
                    )
                    whisper_text = clean_whisper_text(whisper_result["text"])

                    #! Process PhoWhisper
                    phowhisper_text = clean_whisper_text(
                        phowhisper(audio_sample)["text"]
                    )

                    # ! Calculate WER
                    model_wer = wer_metric.compute(
                        references=[ground_truth], predictions=[pred_text]
                    )
                    phowhisper_wer = wer_metric.compute(
                        references=[ground_truth], predictions=[phowhisper_text]
                    )
                    whisper_wer = wer_metric.compute(
                        references=[ground_truth], predictions=[whisper_text]
                    )

                    result_dict = {
                        "audio_id": f"audio_{batch_idx * self.config.batch_size + i}",
                        "ground_truth": ground_truth,
                        "predicted_output": pred_text,
                        "phowhisper_output": phowhisper_text,
                        "whisper_output": whisper_text,
                        "model_wer": model_wer,
                        "phowhisper_wer": phowhisper_wer,
                        "whisper_wer": whisper_wer,
                    }

                    results.append(result_dict)
                    print(result_dict, "\n")

                    predictions_table.add_data(
                        result_dict["audio_id"],
                        result_dict["ground_truth"],
                        result_dict["predicted_output"],
                        result_dict["phowhisper_output"],
                        result_dict["whisper_output"],
                        result_dict["model_wer"],
                        result_dict["phowhisper_wer"],
                        result_dict["whisper_wer"],
                    )

                    progress_bar.update(1)

        progress_bar.close()

        # Create token distribution histogram
        all_tokens = torch.cat(all_tokens).numpy()

        # Create histogram table
        token_data = [[int(token)] for token in all_tokens]
        token_df = pd.DataFrame(token_data, columns=["token_index"])
        token_df.to_csv("all_tokens.csv", index=False)

        # WER chart
        avg_model_wer = sum(r["model_wer"] for r in results) / len(results)
        avg_whisper_wer = sum(r["whisper_wer"] for r in results) / len(results)
        avg_phowhisper_wer = sum(r["phowhisper_wer"] for r in results) / len(results)

        wer_data = [
            [label, val]
            for (label, val) in [
                ("Ichigo Quantizer", avg_model_wer),
                ("PhoWhisper Large", avg_phowhisper_wer),
                ("Whisper Medium", avg_whisper_wer),
            ]
        ]
        wer_chart = wandb.plot.bar(
            wandb.Table(data=wer_data, columns=["Model", "WER"]),
            "Model",
            "WER",
            title="Word Error Rate Comparison",
        )

        metrics = {
            "predictions": predictions_table,
            "avg_model_wer": avg_model_wer,
            "avg_whisper_wer": avg_whisper_wer,
            "avg_phowhisper_wer": avg_phowhisper_wer,
            "wer_comparison": wer_chart,
        }
        self.wandb_logger.experiment.log(metrics)

        return pd.DataFrame(results)

    def _save_model(self, model):
        Path(self.config.task).mkdir(exist_ok=True, parents=True)
        fname = f"{self.config.task}/{self.run_name}.model"
        print(f"Saving: {fname}")
        model.save_model(fname)
