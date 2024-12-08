import os
import datetime
from pathlib import Path
import torch
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.fabric.utilities.rank_zero import rank_zero_only
from config.trainer_config import TrainerConfig
from trainer.lightning_module import WhisperVQModule
from trainer.utils import clean_whisper_text
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import whisper
from tqdm import tqdm
import wandb


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
        1. ModelCheckpoint: For saving model checkpoints based on validation metrics
        2. LearningRateMonitor: For tracking learning rate changes
        """
        self.callbacks = [
            ModelCheckpoint(
                dirpath=self.config.checkpoint_dir,
                filename=f"{self.config.task}/{self.run_name}/{{epoch}}-{{step}}-{{val/loss:.2f}}",
                monitor=self.config.monitored_metric,
                save_top_k=1,
                mode="min",
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval="step"),
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
        """
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

        if rank_zero_only.rank == 0:
            print(f"🥹 Training dataset size: {train_dataset_size}")

        lightning_module = WhisperVQModule(
            model, self.config, train_dataset_size=train_dataset_size
        )

        self.trainer.fit(
            model=lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loaders,
            ckpt_path=self.config.resume_from,
        )

        if rank_zero_only.rank == 0:
            self._save_model(model)

    def get_predictions(self, model, test_dataset, whisper_name, language):
        whisper_model = whisper.load_model(whisper_name)
        whisper_model.to("cuda")

        # W&B Table to store the results
        columns = ["audio_id", "ground_truth", "predicted_output", "whisper_output"]
        predictions_table = wandb.Table(columns=columns)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        results = []
        model.eval()
        model = model.cuda()

        audio_id_counter = 0

        total_samples = len(test_dataset)

        progress_bar = tqdm(
            total=total_samples, desc="Generating predictions", unit="samples"
        )

        with torch.no_grad():
            for batch_idx, (samples, mask, input_toks, output_toks) in enumerate(
                test_loader
            ):
                samples = samples.cuda()
                mask = mask.cuda()
                input_toks = input_toks.cuda()
                output_toks = output_toks.cuda()

                # Get predictions from your model
                _, logits, _ = model(samples, mask, input_toks, output_toks)

                # Process each sample in the batch
                for i in range(len(samples)):
                    # Your model predictions
                    pred_tokens = logits[i].argmax(dim=-1)
                    pred_text = model.tokenizer.decode(pred_tokens.tolist())
                    pred_text = clean_whisper_text(pred_text)

                    # Ground truth
                    ground_truth = model.tokenizer.decode(
                        output_toks[i][output_toks[i] != -100].tolist()
                    )
                    ground_truth = clean_whisper_text(ground_truth)

                    # Get Whisper model prediction
                    audio_sample = samples[i].cpu().numpy()
                    whisper_result = whisper_model.transcribe(
                        audio_sample,
                        language=language,
                        task="transcribe",
                        fp16=False,
                    )
                    whisper_text = clean_whisper_text(whisper_result["text"])

                    result_dict = {
                        "audio_id": f"audio_{audio_id_counter}",
                        "ground_truth": ground_truth,
                        "predicted_output": pred_text,
                        "whisper_output": whisper_text,
                    }

                    predictions_table.add_data(
                        result_dict["audio_id"],
                        result_dict["ground_truth"],
                        result_dict["predicted_output"],
                        result_dict["whisper_output"],
                    )

                    # print(result_dict)
                    results.append(result_dict)
                    audio_id_counter += 1
                    progress_bar.update(1)
        progress_bar.close()

        self.wandb_logger.experiment.log({"predictions": predictions_table})

        return pd.DataFrame(results)

    def _save_model(self, model):
        Path(self.config.task).mkdir(exist_ok=True, parents=True)
        fname = f"{self.config.task}/{self.run_name}.model"
        print(f"Saving: {fname}")
        model.save_model(fname)
