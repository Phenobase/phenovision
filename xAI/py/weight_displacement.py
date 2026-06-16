#!/usr/bin/env python3
"""
Compute weight displacement between checkpoints.

Measures the L2 distance in parameter space between two model states,
decomposed by layer group. This is the straight-line "evolutionary distance"
(displacement) as opposed to the path length (cumulative gradient norms).

Usage:
    python xAI/py/weight_displacement.py
    python xAI/py/weight_displacement.py --ckpt_a path/to/a.pt --ckpt_b path/to/b.pt
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import torch


# Layer group definitions for ViT-L/16 (24 blocks)
LAYER_GROUPS = {
    "head": lambda name: name.startswith("head."),
    "last_block": lambda name: name.startswith("blocks.23."),
    "second_last_block": lambda name: name.startswith("blocks.22."),
    "first_block": lambda name: name.startswith("blocks.0."),
    "second_block": lambda name: name.startswith("blocks.1."),
    "patch_embed": lambda name: name.startswith("patch_embed."),
    "pos_embed": lambda name: "pos_embed" in name,
    "cls_token": lambda name: "cls_token" in name,
    "norm": lambda name: name.startswith("norm.") or name.startswith("fc_norm."),
}


def classify_param(name):
    """Classify a parameter name into a layer group."""
    for group, check_fn in LAYER_GROUPS.items():
        if check_fn(name):
            return group
    # Middle blocks
    for i in range(2, 23):
        if name.startswith(f"blocks.{i}."):
            return f"middle_blocks"
    return "other"


def compute_displacement(ckpt_a_path, ckpt_b_path):
    """Compute L2 weight displacement between two checkpoints.

    Returns:
        dict with "total" and per-group displacement values,
        plus parameter counts per group.
    """
    print(f"Loading: {ckpt_a_path}")
    a = torch.load(ckpt_a_path, map_location="cpu", weights_only=False)
    print(f"Loading: {ckpt_b_path}")
    b = torch.load(ckpt_b_path, map_location="cpu", weights_only=False)

    sd_a = a.get("model_state_dict", a)
    sd_b = b.get("model_state_dict", b)

    group_sq = defaultdict(float)
    group_params = defaultdict(int)
    total_sq = 0.0
    total_params = 0

    for key in sd_a:
        if key not in sd_b:
            print(f"  Warning: {key} not in checkpoint B, skipping")
            continue

        diff = sd_a[key].float() - sd_b[key].float()
        dist_sq = (diff ** 2).sum().item()
        n_params = diff.numel()

        total_sq += dist_sq
        total_params += n_params

        group = classify_param(key)
        group_sq[group] += dist_sq
        group_params[group] += n_params

    result = {
        "total": total_sq ** 0.5,
        "total_params": total_params,
    }

    for group in sorted(group_sq.keys()):
        result[f"{group}_dist"] = group_sq[group] ** 0.5
        result[f"{group}_params"] = group_params[group]
        # Normalized: displacement per parameter (mean per-param change)
        result[f"{group}_per_param"] = (group_sq[group] / group_params[group]) ** 0.5 if group_params[group] > 0 else 0.0

    return result


def compute_all_displacements(ckpt_dir, start_ckpt, output_csv):
    """Compute displacement from start checkpoint to all other checkpoints in a directory.

    Useful for tracking how far the model has moved from initialization over time.
    """
    checkpoints = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".pt") and f != os.path.basename(start_ckpt)
    ])

    print(f"Computing displacement from {start_ckpt} to {len(checkpoints)} checkpoints...")

    # Load start checkpoint once
    start_path = start_ckpt if os.path.isabs(start_ckpt) else os.path.join(ckpt_dir, os.path.basename(start_ckpt))
    start = torch.load(start_path, map_location="cpu", weights_only=False)
    sd_start = start.get("model_state_dict", start)

    rows = []
    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"  {ckpt_name}...", end=" ")

        other = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd_other = other.get("model_state_dict", other)

        group_sq = defaultdict(float)
        total_sq = 0.0

        for key in sd_start:
            if key not in sd_other:
                continue
            diff = sd_start[key].float() - sd_other[key].float()
            dist_sq = (diff ** 2).sum().item()
            total_sq += dist_sq
            group = classify_param(key)
            group_sq[group] += dist_sq

        total_dist = total_sq ** 0.5
        print(f"dist={total_dist:.4f}")

        row = {
            "checkpoint": ckpt_name,
            "global_step": other.get("global_step", -1),
            "epoch": other.get("epoch", -1),
            "phase": other.get("phase", "unknown"),
            "total_displacement": total_dist,
        }
        for group in sorted(group_sq.keys()):
            row[f"{group}_displacement"] = group_sq[group] ** 0.5

        rows.append(row)

        # Free memory
        del other, sd_other

    # Write CSV
    if rows:
        fieldnames = rows[0].keys()
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {output_csv}")

    return rows


def main():
    p = argparse.ArgumentParser(description="Compute weight displacement between checkpoints")
    p.add_argument("--mode", choices=["pair", "trajectory"], default="trajectory",
                   help="'pair' for two checkpoints, 'trajectory' for all from start")
    p.add_argument("--ckpt_a", type=str, help="First checkpoint (for pair mode)")
    p.add_argument("--ckpt_b", type=str, help="Second checkpoint (for pair mode)")
    p.add_argument("--output_dir", type=str, default="xAI/output",
                   help="Output directory for CSVs")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "pair":
        if not args.ckpt_a or not args.ckpt_b:
            p.error("--ckpt_a and --ckpt_b required for pair mode")
        result = compute_displacement(args.ckpt_a, args.ckpt_b)
        print("\nResults:")
        for k, v in sorted(result.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    elif args.mode == "trajectory":
        # Compute displacement trajectories for both models
        for model_name in ["plantclef", "mae"]:
            run_dir = f"xAI/output/{model_name}_run"
            ckpt_dir = os.path.join(run_dir, "checkpoints")

            if not os.path.isdir(ckpt_dir):
                print(f"Skipping {model_name}: {ckpt_dir} not found")
                continue

            start_ckpt = os.path.join(ckpt_dir, "phase2_start.pt")
            if not os.path.exists(start_ckpt):
                print(f"Skipping {model_name}: phase2_start.pt not found")
                continue

            print(f"\n{'='*60}")
            print(f"  {model_name.upper()}: Displacement from Phase 2 start")
            print(f"{'='*60}")

            output_csv = os.path.join(args.output_dir, f"{model_name}_weight_displacement.csv")
            compute_all_displacements(ckpt_dir, start_ckpt, output_csv)

        # Also compute the pair displacement: start -> approximate peak
        print(f"\n{'='*60}")
        print("  SUMMARY: Start to Peak Displacement")
        print(f"{'='*60}")

        for model_name, peak_ckpt_name in [
            ("plantclef", "phase2_epoch10.pt"),  # approximate peak epoch
            ("mae", "phase2_epoch13.pt"),
        ]:
            start = f"xAI/output/{model_name}_run/checkpoints/phase2_start.pt"
            peak = f"xAI/output/{model_name}_run/checkpoints/{peak_ckpt_name}"
            if os.path.exists(start) and os.path.exists(peak):
                result = compute_displacement(start, peak)
                print(f"\n{model_name.upper()} (start -> {peak_ckpt_name}):")
                print(f"  Total displacement: {result['total']:.4f}")
                for k, v in sorted(result.items()):
                    if k.endswith("_dist"):
                        group = k.replace("_dist", "")
                        per_param = result.get(f"{group}_per_param", 0)
                        n_params = result.get(f"{group}_params", 0)
                        print(f"  {group:20s}: dist={v:.4f}  per_param={per_param:.6f}  n_params={n_params:>12,}")
            else:
                print(f"  {model_name}: checkpoint not found, skipping")


if __name__ == "__main__":
    main()
