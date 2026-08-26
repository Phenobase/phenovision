#!/usr/bin/env python3
"""
xAI Training: Pretraining as Preadaptation Experiment

Two-phase training for ecological fitting decomposition:
  Phase 1 (Equalization): Frozen backbone, train head only -> reveals true starting fitness
  Phase 2 (Evolution): Full model unfrozen -> measures evolvability

Collects: frequent validation (AUC-ROC, AUC-PR, loss), gradient norms (Phase 2 only),
          latent representations, and checkpoints with hybrid frequency.

Usage:
    python xAI/py/xai_train.py --pretrained_model mae
    python xAI/py/xai_train.py --pretrained_model plantclef
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'PlantCLEF2022'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import timm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import util.misc as misc
import util.lr_sched as lr_sched
from models_vit import vit_large_patch16
from util.pos_embed import interpolate_pos_embed
from py.custom_dataset import PhenoDataset

# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="xAI Pretraining-as-Preadaptation Training")
    p.add_argument("--pretrained_model", type=str, default="mae", choices=["mae", "plantclef"])
    p.add_argument("--num_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=768, help="Batch size for Phase 1 (frozen backbone)")
    p.add_argument("--phase2_batch_size", type=int, default=384, help="Batch size for Phase 2 (full model, needs more VRAM)")
    p.add_argument("--test_batch_size", type=int, default=None, help="Override ALL batch sizes for testing on smaller GPU")
    p.add_argument("--val_every_n", type=int, default=100, help="Validate every N training batches")
    p.add_argument("--repr_every_n", type=int, default=500, help="Extract representations every N batches")
    p.add_argument("--val_subset_size", type=int, default=10000)
    p.add_argument("--repr_subset_size", type=int, default=1000)
    p.add_argument("--train_csv", type=str, default="data/inat/train_v1.1.0.csv")
    p.add_argument("--val_csv", type=str, default="data/inat/val_v1.1.0.csv")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--blr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--layer_decay", type=float, default=0.65)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--plateau_window", type=int, default=3)
    p.add_argument("--plateau_threshold", type=float, default=0.002)
    p.add_argument("--plateau_patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    p.add_argument("--max_train_samples", type=int, default=None, help="Limit training data for testing")
    p.add_argument("--resume_phase2", type=str, default=None,
                   help="Path to phase1_final checkpoint to skip Phase 1 and jump to Phase 2")
    return p.parse_args()


# =============================================================================
# Data Loading
# =============================================================================

def load_csv_data(csv_path):
    """Load training/validation CSV and return (file_names, labels_array)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    file_names = df["file_name"].tolist()
    labels = df[["fruiting", "flowering"]].values.astype(np.float32)
    return file_names, labels


def create_data_loaders(args):
    """Create train, val_subset, and repr_subset data loaders."""
    print("Loading data...")
    train_files, train_labels = load_csv_data(args.train_csv)
    val_files, val_labels = load_csv_data(args.val_csv)

    # Optionally limit training data (for testing on small GPU)
    if args.max_train_samples and args.max_train_samples < len(train_files):
        rng_train = np.random.RandomState(args.seed + 1)
        idx = rng_train.choice(len(train_files), args.max_train_samples, replace=False)
        train_files = [train_files[i] for i in idx]
        train_labels = train_labels[idx]
        print(f"  [TEST MODE] Limited training to {len(train_files)} samples")

    # Fixed random subsets
    rng = np.random.RandomState(args.seed)
    val_idx = rng.choice(len(val_files), min(args.val_subset_size, len(val_files)), replace=False)
    repr_idx = val_idx[:min(args.repr_subset_size, len(val_idx))]

    val_sub_files = [val_files[i] for i in val_idx]
    val_sub_labels = val_labels[val_idx]
    repr_sub_files = [val_files[i] for i in repr_idx]
    repr_sub_labels = val_labels[repr_idx]

    print(f"  Training images: {len(train_files)}")
    print(f"  Validation subset: {len(val_sub_files)}")
    print(f"  Representation subset: {len(repr_sub_files)}")

    bs = args.test_batch_size or args.batch_size
    nw = args.num_workers
    if nw is None:
        nw = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

    # Transforms
    transform_train = timm.data.create_transform(
        input_size=224, is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25, re_mode="pixel", re_count=1,
    )
    transform_val = timm.data.create_transform(input_size=224, is_training=False)

    # Datasets
    train_ds = PhenoDataset(train_files, train_labels)
    train_ds.transform = transform_train
    val_sub_ds = PhenoDataset(val_sub_files, val_sub_labels)
    val_sub_ds.transform = transform_val
    repr_sub_ds = PhenoDataset(repr_sub_files, repr_sub_labels)
    repr_sub_ds.transform = transform_val

    # Loaders
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_sub_dl = torch.utils.data.DataLoader(
        val_sub_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True, drop_last=False,
    )
    repr_sub_dl = torch.utils.data.DataLoader(
        repr_sub_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True, drop_last=False,
    )

    # Save indices
    os.makedirs(args.output_dir, exist_ok=True)
    np.savetxt(os.path.join(args.output_dir, "val_subset_indices.csv"), val_idx, fmt="%d")
    np.savetxt(os.path.join(args.output_dir, "repr_subset_indices.csv"), repr_idx, fmt="%d")

    return train_dl, val_sub_dl, repr_sub_dl


# =============================================================================
# Model Initialization
# =============================================================================

def init_model(args, device):
    """Initialize ViT-L/16 with specified pretrained weights."""
    print(f"Initializing model: {args.pretrained_model}")
    model = vit_large_patch16(num_classes=2)

    if args.pretrained_model == "plantclef":
        model_path = "models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"PlantCLEF base model not found: {model_path}\n"
                "Please recover this file and place it in models/"
            )
        checkpoint = torch.load(model_path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        # Remove head weights (we reinitialize)
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model:
                del checkpoint_model[k]
        model.load_state_dict(checkpoint_model, strict=False)
        interpolate_pos_embed(model, checkpoint_model)

    elif args.pretrained_model == "mae":
        timm_model = timm.create_model("vit_large_patch16_224.mae", pretrained=True)
        model.load_state_dict(timm_model.state_dict(), strict=False)

    else:
        raise ValueError(f"Unknown pretrained_model: {args.pretrained_model}")

    # Random classification head
    nn.init.trunc_normal_(model.head.weight, std=1e-5)
    nn.init.zeros_(model.head.bias)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    return model


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Fast validation: AUC-ROC, AUC-PR, val_loss on subset."""
    model.eval()
    all_probs, all_targets = [], []
    total_loss, n = 0.0, 0

    for batch in val_loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[-1].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        probs = torch.sigmoid(outputs)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        total_loss += loss.item()
        n += 1

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    val_loss = total_loss / max(n, 1)

    metrics = {"val_loss": val_loss}
    for i, cls in enumerate(["fruit", "flower"]):
        try:
            metrics[f"auc_roc_{cls}"] = roc_auc_score(targets[:, i], probs[:, i])
        except ValueError:
            metrics[f"auc_roc_{cls}"] = float("nan")
        try:
            metrics[f"auc_pr_{cls}"] = average_precision_score(targets[:, i], probs[:, i])
        except ValueError:
            metrics[f"auc_pr_{cls}"] = float("nan")

    metrics["auc_roc_mean"] = np.nanmean([metrics["auc_roc_fruit"], metrics["auc_roc_flower"]])
    metrics["auc_pr_mean"] = np.nanmean([metrics["auc_pr_fruit"], metrics["auc_pr_flower"]])

    return metrics


@torch.no_grad()
def extract_features(model, data_loader, device):
    """Extract CLS token features from the penultimate layer."""
    model.eval()
    all_features = []
    for batch in data_loader:
        images = batch[0].to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            features = model.forward_features(images)
        all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)


# =============================================================================
# Gradient norms
# =============================================================================

GRADIENT_LAYERS = {
    "head": "head.",
    "last_block": "blocks.23.",
    "first_block": "blocks.0.",
}

def compute_gradient_norms(model):
    """Compute total and per-layer gradient L2 norms."""
    norms = {}
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.data.norm(2).item() ** 2
    norms["grad_norm_total"] = math.sqrt(total_sq)

    for layer_name, prefix in GRADIENT_LAYERS.items():
        layer_sq = 0.0
        for name, p in model.named_parameters():
            if name.startswith(prefix) and p.grad is not None:
                layer_sq += p.grad.data.norm(2).item() ** 2
        norms[f"grad_norm_{layer_name}"] = math.sqrt(layer_sq)

    return norms


# =============================================================================
# Checkpoint schedule
# =============================================================================

def should_save_checkpoint(phase2_epoch, batch_idx, batches_per_epoch):
    """Return True if a checkpoint should be saved at this point in Phase 2."""
    if phase2_epoch <= 5:
        return (batch_idx + 1) % 240 == 0
    elif phase2_epoch <= 10:
        return (batch_idx + 1) % 500 == 0
    else:
        return (batch_idx + 1) == batches_per_epoch  # end of epoch only


def save_checkpoint(model, path, global_step, epoch, phase, extra=None):
    state = {
        "model_state_dict": model.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "phase": phase,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"  CKPT: {path}")


# =============================================================================
# Metrics CSV logger
# =============================================================================

class MetricsLogger:
    def __init__(self, path):
        self.path = path
        self.rows = []
        self.fieldnames = None

    def log(self, row):
        self.rows.append(row)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


# =============================================================================
# Training phases
# =============================================================================

def run_phase1(model, train_dl, val_dl, criterion, device, args, metrics_logger):
    """Phase 1: Frozen backbone, train head only until plateau."""
    print("\n" + "=" * 60)
    print("PHASE 1: Frozen Backbone (Equalization)")
    print("=" * 60)

    # Freeze backbone
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("head."))

    head_params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in head_params)
    print(f"  Trainable parameters: {trainable:,} (head only)")

    optimizer = torch.optim.Adam(head_params, lr=1e-3)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    # Simple args for Phase 1
    phase1_args = argparse.Namespace(
        accum_iter=1, warmup_epochs=0, lr=1e-3, min_lr=1e-4, epochs=args.num_epochs,
    )

    global_step = 0
    plateau_losses = deque(maxlen=args.plateau_window)
    plateau_count = 0
    phase1_converged = False
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")

    for epoch in range(args.num_epochs):
        model.train()
        print(f"\n--- Phase 1, Epoch {epoch} ---")
        epoch_loss = 0.0
        n_batches = len(train_dl)

        for batch_idx, (samples, targets) in enumerate(train_dl):
            # LR schedule
            lr_sched.adjust_learning_rate(optimizer, batch_idx / n_batches + epoch, phase1_args)

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_value = loss.item()
            epoch_loss += loss_value

            loss_scaler(loss, optimizer, parameters=head_params, create_graph=False, update_grad=True)
            optimizer.zero_grad()
            torch.cuda.synchronize()

            global_step += 1

            # Periodic validation (no gradient logging in Phase 1)
            if global_step % args.val_every_n == 0:
                val_metrics = validate(model, val_dl, criterion, device)
                row = {
                    "global_step": global_step, "epoch": epoch, "batch": batch_idx,
                    "phase": "frozen", "train_loss": loss_value, **val_metrics,
                }
                metrics_logger.log(row)
                print(f"  VAL step={global_step} val_loss={val_metrics['val_loss']:.4f} "
                      f"auc_roc_mean={val_metrics['auc_roc_mean']:.4f}")

                # Plateau detection (based on AUC-ROC, not val_loss)
                plateau_losses.append(val_metrics["auc_roc_mean"])
                if len(plateau_losses) == args.plateau_window:
                    improvement = plateau_losses[-1] - plateau_losses[0]  # AUC: higher is better
                    if improvement < args.plateau_threshold:
                        plateau_count += 1
                        print(f"  Plateau ({plateau_count}/{args.plateau_patience}) "
                              f"auc_roc improvement={improvement:.4f}")
                        if plateau_count >= args.plateau_patience:
                            phase1_converged = True
                    else:
                        plateau_count = 0

                model.train()

            if phase1_converged:
                break

            # Print progress
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch} [{batch_idx}/{n_batches}] "
                      f"loss={loss_value:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

        # End-of-epoch checkpoint
        save_checkpoint(model, os.path.join(ckpt_dir, f"phase1_epoch{epoch}.pt"),
                        global_step, epoch, "frozen")

        # Check convergence (may already be set mid-epoch by inner break)
        if phase1_converged:
            print(f"\n*** Phase 1 converged at epoch {epoch}, step {global_step} ***")
            break
    else:
        print(f"\n*** Phase 1 did not converge in {args.num_epochs} epochs — proceeding ***")

    # Final validation
    val_metrics = validate(model, val_dl, criterion, device)
    row = {
        "global_step": global_step, "epoch": epoch, "batch": -1,
        "phase": "frozen_final", "train_loss": float("nan"), **val_metrics,
    }
    metrics_logger.log(row)

    # Save phase1_final checkpoint (the reusable starting point for Phase 2)
    save_checkpoint(model, os.path.join(ckpt_dir, "phase1_final.pt"),
                    global_step, epoch, "frozen_final",
                    extra={"val_loss": val_metrics["val_loss"]})

    print(f"\nPHASE_TRANSITION: step={global_step} epoch={epoch} "
          f"val_loss={val_metrics['val_loss']:.4f} "
          f"auc_roc_mean={val_metrics['auc_roc_mean']:.4f}")

    return global_step, epoch


def run_phase2(model, train_dl, val_dl, repr_dl, criterion, device, args,
               metrics_logger, start_global_step, start_epoch):
    """Phase 2: Full model unfrozen — the evolutionary experiment."""
    print("\n" + "=" * 60)
    print("PHASE 2: Full Model Unfrozen (Evolution)")
    print("=" * 60)

    # Rebuild train loader with Phase 2 batch size (smaller to fit gradients in VRAM)
    p2_bs = args.test_batch_size or args.phase2_batch_size
    if p2_bs != (args.test_batch_size or args.batch_size):
        print(f"  Rebuilding train loader: batch_size {args.batch_size} -> {p2_bs} for Phase 2")
        train_dl = torch.utils.data.DataLoader(
            train_dl.dataset,
            batch_size=p2_bs,
            shuffle=True,
            num_workers=train_dl.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad_(True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  All parameters unfrozen: {total_params:,} trainable")

    # Optimizer with layer-wise LR decay
    bs = p2_bs
    effective_lr = args.blr * bs / 256
    from util.lr_decay import param_groups_lrd
    param_groups = param_groups_lrd(
        model,
        weight_decay=args.weight_decay,
        no_weight_decay_list=model.no_weight_decay() if hasattr(model, 'no_weight_decay') else [],
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=effective_lr)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    phase2_args = argparse.Namespace(
        accum_iter=1, warmup_epochs=args.warmup_epochs,
        lr=effective_lr, min_lr=args.min_lr, epochs=args.num_epochs,
    )

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    repr_dir = os.path.join(args.output_dir, "representations")

    global_step = start_global_step
    phase2_start_epoch = start_epoch + 1
    n_batches = len(train_dl)

    # Save phase2_start checkpoint
    save_checkpoint(model, os.path.join(ckpt_dir, "phase2_start.pt"),
                    global_step, phase2_start_epoch, "phase2_start")
    phase2_start_step = global_step

    # Open gradient norms CSV for streaming writes
    grad_csv_path = os.path.join(args.output_dir, "metrics", "gradient_norms.csv")
    grad_csv_file = open(grad_csv_path, "w", newline="")
    grad_fields = ["global_step", "epoch", "batch", "train_loss",
                    "grad_norm_total", "grad_norm_head", "grad_norm_last_block", "grad_norm_first_block"]
    grad_writer = csv.DictWriter(grad_csv_file, fieldnames=grad_fields)
    grad_writer.writeheader()

    for epoch in range(phase2_start_epoch, args.num_epochs):
        model.train()
        phase2_ep = epoch - phase2_start_epoch
        print(f"\n--- Phase 2, Epoch {epoch} (P2 epoch {phase2_ep}) ---")

        for batch_idx, (samples, targets) in enumerate(train_dl):
            # LR schedule
            lr_sched.adjust_learning_rate(optimizer, batch_idx / n_batches + epoch, phase2_args)

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping")
                sys.exit(1)

            # loss_scaler does: backward, unscale, clip/get_norm, step, update
            # It returns the total gradient norm (computed after unscale, before step)
            total_grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                          create_graph=False, update_grad=True)

            # Compute per-layer gradient norms BEFORE zero_grad clears them
            # (gradients still exist after optimizer.step, cleared by zero_grad)
            grad_norms = compute_gradient_norms(model)
            # Override total with the scaler's more accurate value (pre-clip)
            if total_grad_norm is not None:
                norm_val = total_grad_norm.item() if hasattr(total_grad_norm, 'item') else float(total_grad_norm)
                # GradScaler produces inf when it needs to adjust scale — replace with NaN for clean data
                if math.isinf(norm_val):
                    norm_val = float("nan")
                grad_norms["grad_norm_total"] = norm_val
            # Also clean inf in per-layer norms (same cause)
            for k in grad_norms:
                if math.isinf(grad_norms[k]):
                    grad_norms[k] = float("nan")

            optimizer.zero_grad()
            torch.cuda.synchronize()

            global_step += 1

            # --- Log gradient norms ---
            grad_row = {
                "global_step": global_step, "epoch": epoch, "batch": batch_idx,
                "train_loss": loss_value, **grad_norms,
            }
            grad_writer.writerow(grad_row)
            grad_csv_file.flush()

            # --- Adaptive frequency based on Phase 2 batches elapsed ---
            phase2_batches = global_step - phase2_start_step
            if phase2_batches <= 500:
                val_freq = 20
                repr_freq = 50
            elif phase2_batches <= 1000:
                val_freq = 50
                repr_freq = 100
            else:
                val_freq = args.val_every_n   # 100
                repr_freq = args.repr_every_n  # 500

            # --- Periodic validation ---
            if phase2_batches % val_freq == 0:
                val_metrics = validate(model, val_dl, criterion, device)
                row = {
                    "global_step": global_step, "epoch": epoch, "batch": batch_idx,
                    "phase": "unfrozen", "train_loss": loss_value, **val_metrics,
                }
                metrics_logger.log(row)
                print(f"  VAL step={global_step} (p2_batch={phase2_batches}, freq={val_freq}) "
                      f"val_loss={val_metrics['val_loss']:.4f} "
                      f"auc_roc_mean={val_metrics['auc_roc_mean']:.4f}")
                model.train()

            # --- Periodic representation extraction ---
            if phase2_batches % repr_freq == 0:
                features = extract_features(model, repr_dl, device)
                feat_path = os.path.join(repr_dir, f"features_step{global_step:06d}.pt")
                torch.save(features, feat_path)
                print(f"  REPR: {feat_path} shape={tuple(features.shape)}")
                model.train()

            # --- Checkpoints ---
            if should_save_checkpoint(phase2_ep, batch_idx, n_batches):
                ckpt_path = os.path.join(ckpt_dir, f"phase2_step{global_step:06d}.pt")
                save_checkpoint(model, ckpt_path, global_step, epoch, "unfrozen")

            # Progress
            if batch_idx % 20 == 0:
                print(f"  [{batch_idx}/{n_batches}] loss={loss_value:.4f} "
                      f"grad={grad_norms['grad_norm_total']:.4f} "
                      f"g_head={grad_norms['grad_norm_head']:.4f} "
                      f"g_last={grad_norms['grad_norm_last_block']:.4f} "
                      f"g_first={grad_norms['grad_norm_first_block']:.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # End-of-epoch checkpoint
        save_checkpoint(model, os.path.join(ckpt_dir, f"phase2_epoch{epoch}.pt"),
                        global_step, epoch, "unfrozen")

        # Flush gradient CSV
        grad_csv_file.flush()

    grad_csv_file.close()
    return global_step


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join("xAI/output", f"{args.pretrained_model}_run")

    for subdir in ["checkpoints", "representations", "metrics"]:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    print("=" * 60)
    print("xAI Training: Pretraining as Preadaptation")
    print("=" * 60)
    print(f"  Pretrained model: {args.pretrained_model}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.test_batch_size or args.batch_size}")
    print(f"  Val every: {args.val_every_n} batches")
    print(f"  Output: {args.output_dir}")
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_dl, val_sub_dl, repr_sub_dl = create_data_loaders(args)

    # Model
    model = init_model(args, device)

    # Loss
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Metrics logger
    metrics_logger = MetricsLogger(os.path.join(args.output_dir, "metrics", "validation_metrics.csv"))

    t0 = time.time()

    if args.resume_phase2:
        # Skip Phase 1 — load from checkpoint
        print(f"\nResuming from Phase 1 checkpoint: {args.resume_phase2}")
        ckpt = torch.load(args.resume_phase2, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        global_step = ckpt["global_step"]
        phase1_end_epoch = ckpt["epoch"]
        print(f"  Loaded: step={global_step}, epoch={phase1_end_epoch}")
        t1 = time.time()
    else:
        # Save init checkpoint
        save_checkpoint(model, os.path.join(args.output_dir, "checkpoints", "epoch0_init.pt"),
                        0, 0, "init")

        # Phase 1
        global_step, phase1_end_epoch = run_phase1(
            model, train_dl, val_sub_dl, criterion, device, args, metrics_logger
        )
        t1 = time.time()
        print(f"\nPhase 1 took {(t1-t0)/60:.1f} minutes")

    # Phase 2
    global_step = run_phase2(
        model, train_dl, val_sub_dl, repr_sub_dl, criterion, device, args,
        metrics_logger, global_step, phase1_end_epoch
    )
    t2 = time.time()
    print(f"\nPhase 2 took {(t2-t1)/60:.1f} minutes")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {(t2-t0)/60:.1f} minutes")
    print(f"  Total steps: {global_step}")
    print(f"  Phase 1 ended at epoch {phase1_end_epoch}")
    print(f"  Output: {args.output_dir}")

    # Save summary
    summary = {
        "pretrained_model": args.pretrained_model,
        "phase1_end_epoch": phase1_end_epoch,
        "total_steps": global_step,
        "total_time_min": (t2 - t0) / 60,
        "phase1_batch_size": args.test_batch_size or args.batch_size,
        "phase2_batch_size": args.test_batch_size or args.phase2_batch_size,
        "num_epochs": args.num_epochs,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
    }
    with open(os.path.join(args.output_dir, "run_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
