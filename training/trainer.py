"""A resumable, mixed-precision trainer built around Colab's constraints.

Colab sessions die — on idle, on time limits, on the daily quota. So the trainer
checkpoints after *every* epoch to a directory you point at Google Drive, and on
startup resumes from the last checkpoint automatically. A run that gets killed at
epoch 12 picks up at 13 next session instead of starting over.

Model selection is driven by a clinical metric (default AUC), not training loss,
and the best checkpoint is kept separately from the latest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from core.enums import TaskType
from core.types import HeadOutput
from evaluation.metrics import ClassificationEvaluator
from .losses import MultiTaskLoss, _classification_logits


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    mixed_precision: bool = True
    grad_clip: float | None = 1.0
    monitor: str = "auc"          # validation metric to select the best model on
    checkpoint_dir: str = "checkpoints"
    resume: bool = True
    # ── CUDA speed flags ──────────────────────────────────────────────────────
    # cudnn_benchmark: let cuDNN profile and cache the fastest conv kernel for
    #   your exact input shape. Always True for fixed-size inputs (chest 320×320).
    cudnn_benchmark: bool = True
    # channels_last: store 2-D feature maps in NHWC order (N,H,W,C) instead of
    #   NCHW. Tensor Cores on L4/A100 are 20-40% faster in this layout for 2-D
    #   CNNs (DenseNet, ResNet, EfficientNet). Set False for 3-D/volumetric models.
    channels_last: bool = False
    # tf32: enable TF32 for fp32 matmul on Ampere+ GPUs (L4, A100). Near-lossless
    #   10-bit mantissa vs 23-bit, ~3× faster for large linear layers.
    tf32: bool = True


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        evaluator,  # ClassificationEvaluator or MultilabelEvaluator — same interface
        config: TrainConfig,
        loss_fn: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: "torch.optim.lr_scheduler._LRScheduler | None" = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config
        self.use_amp = config.mixed_precision and self.device.type == "cuda"

        # ── Global CUDA tuning (applied once, affects all subsequent ops) ──
        if self.device.type == "cuda":
            if config.tf32:
                # TF32: free ~3× speedup for fp32 matmuls on Ampere+ (L4, A100).
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if config.cudnn_benchmark:
                # Profile conv kernels once for your exact input shape and cache
                # the fastest variant. Essential for fixed-size 2-D inputs.
                torch.backends.cudnn.benchmark = True

        # ── Model placement and memory format ──
        self.channels_last = config.channels_last and self.device.type == "cuda"
        mem_fmt = torch.channels_last if self.channels_last else torch.preserve_format
        self.model = model.to(self.device, memory_format=mem_fmt)

        self.evaluator = evaluator
        self.loss_fn = loss_fn or MultiTaskLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = scheduler
        # torch.amp.GradScaler (2.3+) is the new home; fall back to the cuda-specific
        # variant for the rare case someone pins an older torch.
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)  # <2.3 compat
        self.start_epoch = 0
        self.best_metric = float("-inf")

    # -- checkpointing -------------------------------------------------------
    def _path(self, name: str) -> str:
        return os.path.join(self.config.checkpoint_dir, name)

    def _save(self, name: str, epoch: int) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        # Unwrap compiled model so checkpoint keys never carry _orig_mod. prefix.
        base = getattr(self.model, "_orig_mod", self.model)
        torch.save(
            {
                "epoch": epoch,
                "model": base.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_metric": self.best_metric,
                "class_names": self.evaluator.class_names,
            },
            self._path(name),
        )

    def _maybe_resume(self) -> None:
        last = self._path("last.pt")
        if not (self.config.resume and os.path.isfile(last)):
            return
        ckpt = torch.load(last, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.best_metric = ckpt.get("best_metric", float("-inf"))
        self.start_epoch = ckpt["epoch"] + 1
        print(f"[resume] continuing from epoch {self.start_epoch} (best={self.best_metric:.4f})")

    # -- loop ----------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
        self._maybe_resume()
        history: list[dict] = []
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self._train_one_epoch(train_loader)
            metrics = self._validate(val_loader)

            score = self._monitored_score(metrics)
            self._save("last.pt", epoch)
            if score > self.best_metric:
                self.best_metric = score
                self._save("best.pt", epoch)

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    import torch.optim.lr_scheduler as _sched
                    if isinstance(self.scheduler, _sched.ReduceLROnPlateau):
                        self.scheduler.step(score)
                    else:
                        self.scheduler.step()

            record = {
                "epoch": epoch, "train_loss": train_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                **metrics,
            }
            history.append(record)
            self._log(record, score)
        return history

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running, n = 0.0, 0
        for x, targets in loader:
            # non_blocking=True: H→D transfer runs in a CUDA stream while the CPU
            # continues preparing the next batch. Only effective when the DataLoader
            # uses pin_memory=True (which the Colab notebook does).
            x = x.to(self.device, non_blocking=True)
            if self.channels_last:
                x = x.to(memory_format=torch.channels_last)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(x)
                loss, _ = self.loss_fn(outputs, targets)
            self.scaler.scale(loss).backward()
            if self.config.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running += float(loss.detach()) * x.size(0)
            n += x.size(0)
        return running / max(n, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        self.evaluator.reset()
        running, n = 0.0, 0
        for x, targets in loader:
            x = x.to(self.device, non_blocking=True)
            if self.channels_last:
                x = x.to(memory_format=torch.channels_last)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
            # autocast here too: validation forward should run in fp16 just like training.
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(x)
                loss, _ = self.loss_fn(outputs, targets)
            running += float(loss) * x.size(0)
            n += x.size(0)
            logits = _classification_logits(outputs)
            if logits is not None and "label" in targets:
                self.evaluator.update(logits, targets["label"])
        metrics = self.evaluator.compute()
        metrics["val_loss"] = running / max(n, 1)
        return metrics

    def _monitored_score(self, metrics: dict) -> float:
        value = metrics.get(self.config.monitor)
        if value is None:  # metric undefined this epoch (e.g. AUC with one class)
            value = metrics.get("macro_sensitivity") or metrics.get("accuracy", 0.0)
        return float(value)

    @staticmethod
    def _log(record: dict, score: float) -> None:
        auc = record.get("auc")
        auc_str = f"{auc:.4f}" if auc is not None else "n/a"
        sens = record.get("sensitivity")
        sens_str = f"{sens:.4f}" if sens is not None else "n/a"
        print(
            f"epoch {record['epoch']:3d} | "
            f"train {record['train_loss']:.4f} | "
            f"val {record.get('val_loss', float('nan')):.4f} | "
            f"lr {record.get('lr', 0):.2e} | "
            f"sens {sens_str} | "
            f"auc {auc_str} | best {score:.4f}"
        )
