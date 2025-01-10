import lightning.pytorch as pl
import torch
import torch._dynamo
from lightning.fabric.utilities.rank_zero import rank_zero_only

from models.vq_transformer import RQBottleneckTransformer


class WhisperVQModule(pl.LightningModule):
    def __init__(
        self, model: RQBottleneckTransformer, config, train_dataset_size=None, phase=1
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.train_dataset_size = train_dataset_size
        self.model.phase = phase
        self.save_hyperparameters(config.to_hparams())

    def on_fit_start(self):
        """
        Called at the beginning of training.
        Sets up the model and applies compilation if configured.
        """
        if hasattr(self.model, "setup"):
            self.model.setup(self.device, is_train=self.training)
        self._maybe_compile_model()

    def on_validation_epoch_end(self):
        metrics = self.model.get_metrics()
        self.log(
            "val/epoch_accuracy",
            metrics["acc_0"],
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
        )

    def _maybe_compile_model(self):
        """
        Conditionally compile the model for performance optimization.
        Disables DDP optimization and applies model-specific training optimizations.
        """
        if self.config.torch_compile:
            torch._dynamo.config.optimize_ddp = False
            if hasattr(self.model, "optimize_training"):
                self.model.optimize_training()

    @rank_zero_only
    def _log_training_setup(self, total_steps, warmup_steps, steps_per_epoch=None):
        """Log training setup information (only on master process)"""
        if self.config.iterations:
            print(f"Training with fixed iterations: {total_steps}")
        else:
            print(f"Dataset size: {self.train_dataset_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Total epochs: {self.config.epochs}")

        print(
            f"Training schedule: {total_steps} total steps with {warmup_steps} warmup steps"
        )

    def configure_optimizers(self):
        """
        Initialize AdamW optimizer with parameter groups.

        Returns:
            list: Configured optimizer and scheduler
        """
        return self._configure_optimizer_and_scheduler()

    def _configure_optimizer_and_scheduler(self):
        """
        Configure optimizer and scheduler with parameter groups.

        Handles:
        - Custom learning rates per module
        - Weight decay exclusions
        - Warmup and decay scheduling

        Returns:
            tuple: ([optimizer], [scheduler_config])
        """
        """Configure optimizer and scheduler with parameter groups"""
        lr = self.config.vq_config.lr0
        weight_decay = self.config.vq_config.weight_decay

        # Collect all parameters
        all_params = set(self.model.parameters())
        customized_params = set()
        groups = []
        group_map = {}

        # Group parameters based on module attributes
        for name, m in self.model.named_modules():
            if hasattr(m, "no_weight_decay") or hasattr(m, "lr_scale"):
                customized_params |= set(m.parameters())
                m_wd = 0 if hasattr(m, "no_weight_decay") else weight_decay
                m_lr = lr * getattr(m, "lr_scale", 1)

                group = group_map.get((m_wd, m_lr), None)
                if not group:
                    group = {
                        "params": [],
                        "names": [],
                        "weight_decay": m_wd,
                        "lr": m_lr,
                    }
                    groups.append(group)
                    group_map[(m_wd, m_lr)] = group
                group["params"].extend(m.parameters())
                group["names"].append(name)

        # Add remaining parameters
        other_params = all_params - customized_params
        param_groups = groups + [
            {
                "params": list(other_params),
                "weight_decay": weight_decay,
                "lr": lr,
                "names": ["other"],
            }
        ]

        # Initialize optimizer
        optimizer = torch.optim.AdamW(params=param_groups, lr=lr, betas=(0.9, 0.95))

        # Calculate total steps based on either iterations or epochs
        if self.config.iterations:
            total_steps = self.config.iterations
            steps_per_epoch = None
        else:
            if self.train_dataset_size is None:
                raise ValueError(
                    "train_dataset_size must be provided for epoch-based training"
                )

            # Calculate steps for epoch-based training
            num_devices = self.trainer.num_devices if self.trainer else 1
            steps_per_epoch = self.train_dataset_size // (
                self.config.batch_size * num_devices
            )
            total_steps = steps_per_epoch * self.config.epochs

        # Calculate warmup steps
        warmup_steps = getattr(
            self.config.vq_config, "warmup_steps", max(1, int(0.05 * total_steps))
        )

        # Log training setup (only on master process)
        self._log_training_setup(total_steps, warmup_steps, steps_per_epoch)

        # Create warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
        )

        # Create main training scheduler
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=lr / 25
        )

        # Combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "train_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch: Tuple of (samples, mask, input_toks, output_toks)
            batch_idx: Index of current batch

        Returns:
            loss: Total loss value for optimization
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            (samples, mask, input_toks, output_toks), _ = batch
        else:
            samples, mask, input_toks, output_toks = batch

        list_loss, logits, loss = self.model(samples, mask, input_toks, output_toks)

        metrics = {
            "loss/total_train": loss.item(),
            "loss/ce_loss": list_loss[0],
            "loss/kl_loss": list_loss[1],
            "loss/commit_loss": list_loss[2],
        }

        if hasattr(self.model, "get_codebook_stats"):
            stats = self.model.get_codebook_stats()
            if stats:
                metrics.update(
                    {
                        "codebook/used_codes": stats["used_codes"],
                        "codebook/utilization": stats["utilization"],
                    }
                )

        self.log_dict(
            metrics,
            sync_dist=True,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        Args:
            batch: Tuple of (samples, mask, input_toks, output_toks) or ((samples, mask, input_toks, output_toks), weight)
            batch_idx: Index of current batch
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            (samples, mask, input_toks, output_toks), _ = batch
        else:
            samples, mask, input_toks, output_toks = batch

        _, logits, loss = self.model(samples, mask, input_toks, output_toks)

        # Valid accuracy
        valid_toks = output_toks != -100
        current_true = (
            (logits.detach().argmax(-1)[valid_toks] == output_toks[valid_toks])
            .float()
            .sum()
        )
        current_total = valid_toks.float().sum()
        current_acc = (current_true / current_total).item()

        metrics = {
            f"val/loss": loss.item(),
            f"val/acc": current_acc,
        }

        # Log all metrics
        self.log_dict(
            metrics,
            sync_dist=True,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

    def _calculate_entropy(self, logits):
        """
        Calculate entropy of predictions to measure uncertainty in model's output distribution.

        Higher entropy = more uncertain/random predictions (max log2(vocab_size))
        Lower entropy = more confident predictions

        Args:
            logits: Raw model outputs [batch_size, sequence_length, vocab_size]

        Returns:
            float: Average entropy across batch
        """
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

        # Calculate entropy: -Σ(p * log2(p))
        # 1e-10 is added for numerical stability to avoid log(0)
        entropy = -torch.sum(
            probs * torch.log2(probs + 1e-10), dim=-1
        )  # [batch, seq_len]

        # Return mean entropy across batch
        return entropy.mean().item()

    def load_from_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
