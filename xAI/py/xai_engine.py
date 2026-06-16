"""
Modified training engine for xAI pretraining-as-preadaptation experiments.

Based on PlantCLEF2022/engine_finetune.py but with:
- Callback support for mid-epoch validation, checkpointing, and representation extraction
- Per-batch gradient norm logging
- Phase-aware training (frozen backbone vs full model)
"""

import math
import sys
from typing import Iterable, Optional, Dict, Callable, Tuple

import torch
import torch.nn as nn

from timm.data import Mixup

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'PlantCLEF2022'))
import util.misc as misc
import util.lr_sched as lr_sched
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def freeze_backbone(model):
    """Freeze all parameters except the classification head."""
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("head."):
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
            frozen += param.numel()
    print(f"Frozen: {frozen:,} params, Trainable (head only): {trainable:,} params")
    return trainable


def unfreeze_all(model):
    """Unfreeze all parameters."""
    total = 0
    for param in model.parameters():
        param.requires_grad_(True)
        total += param.numel()
    print(f"All parameters unfrozen: {total:,} trainable")
    return total


def get_head_parameters(model):
    """Return list of head parameters (for Phase 1 optimizer)."""
    return [p for name, p in model.named_parameters() if name.startswith("head.") and p.requires_grad]


def compute_gradient_norms(model, layers_of_interest=None):
    """Compute L2 gradient norms for the full model and specific layers.

    Args:
        model: The model after backward pass
        layers_of_interest: dict mapping layer names to parameter name prefixes
            e.g. {"head": "head.", "last_block": "blocks.23.", "first_block": "blocks.0."}

    Returns:
        dict with "total" and per-layer gradient norms
    """
    norms = {}

    # Total gradient norm
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    norms["total"] = math.sqrt(total_norm_sq)

    # Per-layer norms
    if layers_of_interest:
        for layer_name, prefix in layers_of_interest.items():
            layer_norm_sq = 0.0
            for name, p in model.named_parameters():
                if name.startswith(prefix) and p.grad is not None:
                    layer_norm_sq += p.grad.data.norm(2).item() ** 2
            norms[layer_name] = math.sqrt(layer_norm_sq)

    return norms


def train_one_epoch_with_callbacks(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    args=None,
    callbacks: Dict[str, Tuple[Callable, int]] = None,
    global_step_offset: int = 0,
    gradient_layers: Dict[str, str] = None,
):
    """Train one epoch with periodic callbacks for validation, checkpointing, etc.

    Args:
        model, criterion, data_loader, optimizer, device, epoch, loss_scaler, max_norm,
        mixup_fn, args: Same as original train_one_epoch

        callbacks: Dict mapping callback names to (function, frequency) tuples.
            Callback functions receive (model, global_step, epoch, batch_idx, train_loss).
            Example: {"val": (val_fn, 100), "ckpt": (ckpt_fn, 240), "repr": (repr_fn, 500)}

        global_step_offset: Running count of total batches across all epochs.
            Updated as: global_step = global_step_offset + batch_idx

        gradient_layers: Dict mapping layer names to param name prefixes for gradient norm logging.
            Example: {"head": "head.", "last_block": "blocks.23.", "first_block": "blocks.0."}

    Returns:
        dict with:
            "stats": averaged training stats for the epoch
            "global_step": final global step count after this epoch
            "gradient_norms": list of per-batch gradient norm dicts
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter if hasattr(args, 'accum_iter') else 1
    optimizer.zero_grad()

    callbacks = callbacks or {}
    gradient_norms = []
    num_batches = len(data_loader)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = global_step_offset + data_iter_step

        # Per-iteration LR scheduling
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_batches + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # --- Gradient norm logging (every batch) ---
        grad_norms = compute_gradient_norms(model, gradient_layers)
        gradient_norms.append(grad_norms)

        # Log total gradient norm for every batch (GuildAI format)
        print(f"grad_norm: {grad_norms['total']:.6f}")
        print(f"train_loss_batch: {loss_value:.6f}")
        print(f"global_step: {global_step}")

        metric_logger.update(loss=loss_value)
        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        # --- Callbacks ---
        for cb_name, (cb_fn, cb_freq) in callbacks.items():
            if cb_freq > 0 and (data_iter_step + 1) % cb_freq == 0:
                # Switch to eval mode for callback
                model.eval()
                cb_fn(model, global_step, epoch, data_iter_step, loss_value)
                model.train(True)

    # Gather stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    final_global_step = global_step_offset + num_batches

    return {
        "stats": {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        "global_step": final_global_step,
        "gradient_norms": gradient_norms,
    }


@torch.no_grad()
def fast_validate(model, val_loader, criterion, device):
    """Fast validation on a subset — returns raw predictions and targets for R-side metric computation.

    Returns:
        dict with "outputs" (sigmoid probabilities), "targets", and "val_loss"
    """
    model.eval()
    all_outputs = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[-1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs)
        all_outputs.append(probs.cpu())
        all_targets.append(targets.cpu())
        total_loss += loss.item()
        n_batches += 1

    return {
        "outputs": torch.cat(all_outputs, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "val_loss": total_loss / max(n_batches, 1),
    }


@torch.no_grad()
def extract_features(model, data_loader, device):
    """Extract penultimate layer features for a subset of images.

    Returns:
        torch.Tensor of shape (N, embed_dim) — the CLS token features from the last layer.
    """
    model.eval()
    all_features = []

    for batch in data_loader:
        images = batch[0].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            features = model.forward_features(images)

        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)
