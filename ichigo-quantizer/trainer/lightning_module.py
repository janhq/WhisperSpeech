import torch
import lightning.pytorch as pl
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
            import torch._dynamo

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
        if self.config.lr_schedule == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=lr / 25
            )
        else:  # linear schedule
            main_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=1 / 25,
                total_iters=total_steps - warmup_steps,
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
        """
        samples, mask, input_toks, output_toks = batch
        _, logits, loss = self.model(samples, mask, input_toks, output_toks)

        # Log standard metrics
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)

        # Log codebook utilization
        if hasattr(self.model, "get_codebook_stats"):
            stats = self.model.get_codebook_stats()
            if stats:
                self.log(
                    "codebook/used_codes",
                    stats["used_codes"],
                    sync_dist=True,
                    prog_bar=True,
                )
                self.log(
                    "codebook/utilization",
                    stats["utilization"],
                    sync_dist=True,
                    prog_bar=True,
                )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        samples, mask, input_toks, output_toks = batch
        _, logits, loss = self.model(samples, mask, input_toks, output_toks)

        metrics = {
            f"val_loss_{dataloader_idx}": loss.item(),
            f"val_entropy_{dataloader_idx}": self._calculate_entropy(logits),
        }

        if dataloader_idx == 0:
            metrics["val_loss"] = loss.item()

            # Add codebook utilization metrics for validation
            if hasattr(self.model, "get_codebook_stats"):
                stats = self.model.get_codebook_stats()
                if stats:
                    metrics["val_codebook_utilization"] = stats["utilization"]
                    metrics["val_used_codes"] = stats["used_codes"]

        # Log all metrics
        self.log_dict(metrics, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Perform a test step and collect results in DataFrame.
        """
        import pandas as pd

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

        # Store results in DataFrame
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

        # Log metrics as before
        metrics = {
            "test_loss": loss.item(),
            "test_wer": wer,
            "test_cer": cer,
            "test_entropy": entropy,
        }

        if hasattr(self.model, "get_codebook_stats"):
            stats = self.model.get_codebook_stats()
            if stats:
                metrics["test_codebook_utilization"] = stats["utilization"]
                metrics["test_used_codes"] = stats["used_codes"]

        self.log_dict(metrics, sync_dist=True)
        print(metrics)
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
        """Calculate entropy of predictions"""
        probs = torch.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1).mean().item()

    def load_from_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
