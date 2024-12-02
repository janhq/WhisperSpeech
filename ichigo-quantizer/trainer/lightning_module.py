import torch
import torch._dynamo
import lightning.pytorch as pl
import pandas as pd
from models.vq_transformer import RQBottleneckTransformer
from trainer.wer_metrics import compute_wer_cer


class WhisperVQModule(pl.LightningModule):
    def __init__(self, model: RQBottleneckTransformer, config):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(config.to_hparams())

    def on_fit_start(self):
        """
        Called at the beginning of training.
        Sets up the model and applies compilation if configured.
        """
        if hasattr(self.model, "setup"):
            self.model.setup(self.device)
        self._maybe_compile_model()

    def _maybe_compile_model(self):
        """
        Conditionally compile the model for performance optimization.
        Disables DDP optimization and applies model-specific training optimizations.
        """
        if self.config.torch_compile:
            torch._dynamo.config.optimize_ddp = False
            if hasattr(self.model, "optimize_training"):
                self.model.optimize_training()

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

        # Configure scheduler
        warmup_steps = self.config.vq_config.warmup_steps
        total_steps = self.config.iterations

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
        samples, mask, input_toks, output_toks = batch
        list_loss, logits, loss = self.model(samples, mask, input_toks, output_toks)

        metrics = {
            # Loss metrics
            "loss/total_train": loss,
            "loss/ce_loss": list_loss[0],
            "loss/kl_loss": list_loss[1],
            "loss/commit_loss": list_loss[2],
        }

        # Add codebook metrics
        if hasattr(self.model, "get_codebook_stats"):
            stats = self.model.get_codebook_stats()
            if stats:
                metrics.update(
                    {
                        "codebook/used_codes": stats["used_codes"],
                        "codebook/utilization": stats["utilization"],
                    }
                )

        # Log all metrics at once
        self.log_dict(
            metrics,
            sync_dist=True,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Perform a validation step.

        Args:
            batch: Tuple of (samples, mask, input_toks, output_toks)
            batch_idx: Index of current batch
            dataloader_idx: Index of dataloader when using multiple validation sets
        """
        samples, mask, input_toks, output_toks = batch
        _, logits, loss = self.model(samples, mask, input_toks, output_toks)

        # Base metrics for all validation dataloaders
        metrics = {
            f"val/loss_{dataloader_idx}": loss.item(),
            f"val/entropy_{dataloader_idx}": self._calculate_entropy(logits),
        }

        # Additional metrics for primary validation set
        if dataloader_idx == 0:
            metrics.update(
                {
                    "val/loss": loss.item(),  # Main validation loss
                    "val/entropy": self._calculate_entropy(logits),
                }
            )

            # Add codebook metrics if available
            if hasattr(self.model, "get_codebook_stats"):
                stats = self.model.get_codebook_stats()
                if stats:
                    metrics.update(
                        {
                            "val/codebook_utilization": stats["utilization"],
                            "val/used_codes": stats["used_codes"],
                        }
                    )

        # Log all metrics
        self.log_dict(
            metrics,
            sync_dist=True,
            prog_bar=True,
            on_step=False,  # Validation metrics typically logged per epoch
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        """
        Perform a test step and collect results in DataFrame.

        Args:
            batch: Tuple of (samples, mask, input_toks, output_toks)
            batch_idx: Index of current batch

        Returns:
            dict: Test metrics including loss, WER, CER, and entropy
        """
        samples, mask, input_toks, output_toks = batch

        # Forward pass
        _, logits, loss = self.model(samples, mask, input_toks, output_toks)
        pred_ids = torch.argmax(logits, dim=-1)

        # Ensure output_toks is 2D
        if output_toks.dim() == 3:
            output_toks = output_toks.squeeze(1)

        # Get predictions and targets as text
        predictions = []
        targets = []
        for pred, target in zip(pred_ids, output_toks):
            valid_mask = target != -100
            pred_clean = pred[valid_mask]
            target_clean = target[valid_mask]

            # Convert to text
            pred_text = self.model.tokenizer.decode(pred_clean)
            target_text = self.model.tokenizer.decode(target_clean)

            predictions.append(pred_text)
            targets.append(target_text)

        # Calculate metrics
        wer, cer = self._calculate_wer_cer(pred_ids, output_toks)
        entropy = self._calculate_entropy(logits)

        # Store detailed results for later analysis
        batch_results = pd.DataFrame(
            {
                "batch_idx": batch_idx,
                "prediction": predictions,
                "target": targets,
                "loss": loss.item(),
                "wer": wer,
                "cer": cer,
                "entropy": entropy,
            }
        )

        # Initialize results list if not exists
        if not hasattr(self, "test_results"):
            self.test_results = []
        self.test_results.append(batch_results)

        # Prepare metrics for logging
        metrics = {
            "test/loss": loss.item(),
            "test/wer": wer,
            "test/cer": cer,
            "test/entropy": entropy,
        }

        # Add codebook metrics if available
        if hasattr(self.model, "get_codebook_stats"):
            stats = self.model.get_codebook_stats()
            if stats:
                metrics.update(
                    {
                        "test/codebook_utilization": stats["utilization"],
                        "test/used_codes": stats["used_codes"],
                    }
                )

        # Log all metrics
        self.log_dict(
            metrics,
            sync_dist=True,
            on_step=False,  # Test metrics typically logged once
            on_epoch=True,
            prog_bar=True,
        )

        return metrics

    def _calculate_wer_cer(self, pred_ids, target_ids):
        """
        Calculate WER & CER for batch of predictions

        Args:
            pred_ids: Tensor of shape [batch_size, seq_len]
            target_ids: Tensor of shape [batch_size, seq_len]

        Returns:
            tuple: (wer, cer) scores
        """
        # Process each sequence in the batch
        predictions = []
        targets = []

        for pred, target in zip(pred_ids, target_ids):
            # Remove padding tokens (-100)
            valid_mask = target != -100
            pred_clean = pred[valid_mask]
            target_clean = target[valid_mask]

            predictions.append(pred_clean)
            targets.append(target_clean)

        return compute_wer_cer(predictions, targets, tokenizer=self.model.tokenizer)

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
