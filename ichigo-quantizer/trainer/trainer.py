import os
import datetime
from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.fabric.utilities.rank_zero import rank_zero_only
import webdataset as wds
from config.trainer_config import TrainerConfig
from trainer.lightning_module import WhisperVQModule
from trainer.utils import generate_run_name, setup_dataloaders


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

        self.run_name = generate_run_name()
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
                filename=f"{self.config.task}-{self.run_name}-{{step}}-{{val_loss:.2f}}",
                monitor=self.config.monitored_metric,
                save_top_k=3,
                train_time_interval=datetime.timedelta(minutes=14),
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

    def _setup_trainer(self):
        """
        Initialize PyTorch Lightning trainer.

        Configures training parameters including:
        - Distributed training strategy
        - Hardware acceleration
        - Precision settings
        - Gradient clipping
        - Validation frequency
        - Multi-node training support
        """
        self.trainer = pl.Trainer(
            strategy=self.config.strategy,
            max_steps=self.config.iterations,
            accelerator="gpu",
            precision=self.config.precision,
            gradient_clip_val=self.config.vq_config.clip_gradient_norm,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            val_check_interval=self.config.validate_every_n_steps,
            logger=self.wandb_logger,
            callbacks=self.callbacks,
            num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
            devices=int(self.config.num_gpus),
            log_every_n_steps=1,
        )

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
        train_loader, val_loaders = setup_dataloaders(
            train_dataset, val_datasets, self.config
        )

        lightning_module = WhisperVQModule(model, self.config)

        self.trainer.fit(
            model=lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loaders,
            ckpt_path=self.config.resume_from,
        )

        if rank_zero_only.rank == 0:
            self._save_model(model)

    def test(self, model, test_dataset):
        """
        Evaluate model on test dataset and save results.

        Args:
            model: The WhisperVQ model to evaluate
            test_dataset: Dataset for testing

        Returns:
            dict: Test metrics including WER and entropy
        """
        try:
            dataset_length = len(test_dataset)
        except Exception:
            temp_loader = wds.WebLoader(test_dataset).unbatched()
            dataset_length = sum(1 for _ in temp_loader)

        num_batches = dataset_length // self.config.batch_size
        if dataset_length % self.config.batch_size != 0:
            num_batches += 1

        test_loader = (
            wds.WebLoader(
                test_dataset,
                num_workers=self.config.num_workers,
                batch_size=None,
                persistent_workers=self.config.num_workers > 0,
            )
            .unbatched()
            .batched(self.config.batch_size)
            .with_length(num_batches)
        )

        # Log test dataset info
        if rank_zero_only.rank == 0:
            print(f"\nTest Dataset Info:")
            print(f"Total samples: {dataset_length}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Number of batches: {num_batches}")

        lightning_module = WhisperVQModule(model, self.config)

        results = self.trainer.test(
            model=lightning_module, dataloaders=test_loader, verbose=True
        )

        if rank_zero_only.rank == 0:
            print("\nTest Results:")
            print(f"Loss: {results[0]['test_loss']:.4f}")
            print(f"WER: {results[0]['test_wer']:.2%}")
            print(f"Entropy: {results[0]['test_entropy']:.4f}")

            # Save detailed test results if available
            if hasattr(lightning_module, "test_results"):
                import pandas as pd

                Path(self.config.task).mkdir(exist_ok=True, parents=True)
                fname = f"{self.config.task}/{self.run_name}_test_results.csv"
                final_results = pd.concat(
                    lightning_module.test_results, ignore_index=True
                )
                final_results.to_csv(fname, index=False)
                print(f"Saved detailed test results to: {fname}")

        return results[0] if results else None

    def _save_model(self, model):
        Path(self.config.task).mkdir(exist_ok=True, parents=True)
        fname = f"{self.config.task}/{self.run_name}.model"
        print(f"Saving: {fname}")
        model.save_model(fname)
