import dataclasses
from pathlib import Path
from typing import List, Optional

from .vq_config import VQConfig


@dataclasses.dataclass
class TrainerConfig:
    """Configuration class for training settings and hyperparameters.

    This class handles all configuration aspects of the training process, including
    hardware settings, dataset configuration, model parameters, and experiment tracking.

    Attributes:
        task (str): Identifier for the training task/experiment.

        # Training Loop Parameters
        iterations (int): Total number of training iterations.
        batch_size (int): Number of samples per batch.
        accumulate_grad_batches (int): Number of batches to accumulate gradients over.
        validate_every_n_steps (int): Frequency of validation in training steps.

        # Hardware/Performance
        num_workers (int): Number of data loading worker processes.
        precision (str): Numerical precision for training (e.g., "16-mixed", "32").
        torch_compile (bool): Whether to use torch.compile() for optimization.
        strategy (str): Training strategy (e.g., "ddp" for DistributedDataParallel).

        # Dataset
        training_data (List[str]): Paths to training data files.
        validation_data (List[str]): Paths to validation data files.
        dataset_config (str): Additional dataset configuration parameters.

        # Model & Optimization
        vq_config (VQConfig): Vector quantization model configuration.
        lr_schedule (str): Learning rate schedule type.
        monitored_metric (str): Metric to monitor for model selection.

        # Checkpointing
        resume_from (Path, optional): Path to checkpoint to resume training from.
        load_from (Path, optional): Path to model weights to load.
        checkpoint_dir (str): Directory to save checkpoints.

        # Experiment Tracking
        wandb_suffix (str, optional): Suffix for Weights & Biases run name.
        wandb_task_name (str, optional): Task name for Weights & Biases.
    """

    # Task identifier
    task: str
    run_name: Optional[str] = None

    # Training phase
    phase: int = 1

    # Training loop parameters
    epochs: int = 100
    iterations: Optional[int] = None
    batch_size: int = 16
    accumulate_grad_batches: int = 1
    validate_every_n_steps: int = 500
    early_stopping_patience: int = 100  # TODO: fix threshold later

    # Hardware/Performance settings
    num_workers: int = 8
    precision: str = "16-mixed"
    torch_compile: bool = False
    strategy: str = "ddp"
    num_gpus: int = 1

    # Dataset configuration
    training_data: List[str] = None
    validation_data: List[str] = None
    dataset_config: str = ""

    # Model and optimization parameters
    vq_config: VQConfig = None
    lr_schedule: str = "linear"

    # Checkpoint handling
    resume_from: Optional[Path] = None
    load_from: Optional[Path] = None
    checkpoint_dir: str = "checkpoints"

    # Experiment tracking
    wandb_suffix: Optional[str] = None
    wandb_task_name: Optional[str] = None

    def __post_init__(self):
        """Initialize default VQConfig if none is provided.

        This method is automatically called after the dataclass is initialized.
        It ensures that vq_config is always set, using default values if necessary.
        """
        if self.vq_config is None:
            self.vq_config = VQConfig()

    def to_hparams(self) -> dict:
        """Convert configuration to a hyperparameters dictionary for training.

        This method creates a simplified dictionary containing the essential
        hyperparameters needed during the training process.

        Returns:
            dict: A dictionary containing training hyperparameters with the following keys:
                - iterations: Total number of training iterations
                - batch_size: Batch size for training
                - accumulate_grad_batches: Gradient accumulation steps
                - validate_every_n_steps: Validation frequency
                - strategy: Training strategy
                - precision: Numerical precision
                - torch_compile: Compilation flag
                - lr_schedule: Learning rate schedule type
                - lr0: Initial learning rate
                - warmup_steps: Number of warmup steps
                - weight_decay: Weight decay value
                - clip_gradient_norm: Gradient clipping norm
        """
        return {
            # Training parameters
            "phase": self.phase,
            "iterations": self.iterations,
            "batch_size": self.batch_size,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "validate_every_n_steps": self.validate_every_n_steps,
            "early_stopping_patience": self.early_stopping_patience,
            # Hardware settings
            "strategy": self.strategy,
            "precision": self.precision,
            "torch_compile": self.torch_compile,
            # Optimization parameters
            "lr_schedule": self.lr_schedule,
            "lr0": self.vq_config.lr0,
            "warmup_steps": self.vq_config.warmup_steps,
            "weight_decay": self.vq_config.weight_decay,
            "clip_gradient_norm": self.vq_config.clip_gradient_norm,
        }
