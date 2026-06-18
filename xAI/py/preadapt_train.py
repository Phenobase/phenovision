#!/usr/bin/env python3
"""
preadapt_train.py — two-phase trainer for the PhenoVision
"pretraining-as-preadaptation" experiment (plan component C3).

One process trains exactly ONE (condition x optimizer-variant x seed) run, e.g.
``plantclef x stable_evo x s42``. The producer side of Plan A: it writes
checkpoints on the adaptive motion-driven schedule of briefing Part II §3 and logs
the cheap per-step stream of §5 inline. It never blocks on the extractor (the
collector owns deletion; this trainer only writes + ``.done``).

Two phases (briefing Part I §3, plan C3):

  PHASE 1 — Equalization (ecological fitting). Freeze the ENTIRE backbone, train
  the HEAD ONLY with Adam until the validation AUC plateaus. The converged
  frozen-head metric *is* the ecological-fitting readout for this condition (the
  linear-readout ceiling on the starting representation). Identical across the two
  optimizer variants — it precedes the variant-specific optimizer. Logged to the
  ScalarStore as ``ecological_fitting_auc_pr`` / ``ecological_fitting_val_loss`` at
  the PHASE_TRANSITION.

  PHASE 2 — Evolution. Unfreeze everything, then re-freeze the input stage
  (``patch_embed.*`` + ``pos_embed``); build a UNIFORM-LR param grouping that
  EXCLUDES the frozen stage; build the variant optimizer (AdamW or
  StableEvolutionSOAP + tuning-free demographic noise); train in bf16 autocast with
  NO GradScaler (precision must not be a confound between variants — and the fp16
  GradScaler corrupts StableEvo's generative preconditioner). The adaptive sampler
  emits checkpoints (full model + optimizer state) and the inline §5 stream is
  logged every step.

This module imports its foundation (do NOT reimplement) from:
  * preadapt_common.py  — checkpoint IO, retention ladder, ScalarStore, projection,
                          trainable_named_parameters, manifest.
  * preadapt_models.py  — init_model, freeze_input_stage, build_uniform_param_groups,
                          head_only_param_groups, input_stage_state,
                          assert_frozen_input_stage.
  * xai_train.py        — create_data_loaders, validate, compute_gradient_norms.
  * PlantCLEF2022/util/lr_sched.py — adjust_learning_rate (uniform when no lr_scale).

Conventions (xAI/two_noise/CONVENTIONS.md §5): every stochastic op threads an
explicit torch.Generator. The demographic-noise generator is seeded ``seed+9973``;
the projection seed is a fixed run-INDEPENDENT constant (1234) so all runs of the
experiment share one coordinate system (recorded in the manifest).

Arch: ViT-L/16, D=1024, 24 blocks, 16 heads, head_dim 64, patch16/224.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import re
import sys
import time
from collections import deque, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- project paths (mirror xai_train.py so PlantCLEF2022 + py modules import cleanly) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLANTCLEF = os.path.join(_HERE, "..", "..", "PlantCLEF2022")
_PROJECT_ROOT = os.path.join(_HERE, "..", "..")
# The StableEvolutionSOAP optimizer lives in the two_noise subproject; it must be on
# sys.path BEFORE importing the optimizer (`from optim.stable_evolution_optimizer import ...`).
_TWO_NOISE = "/blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise"
for _p in (_PLANTCLEF, _PROJECT_ROOT, _TWO_NOISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PIL import ImageFile  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

import util.lr_sched as lr_sched  # noqa: E402  (PlantCLEF2022/util/lr_sched.py)

# Foundation modules (do NOT reimplement) ------------------------------------------------
import preadapt_models as pm  # noqa: E402
import preadapt_common as pc  # noqa: E402
from preadapt_common import (  # noqa: E402
    ScalarStore,
    RetentionLadder,
    build_train_ckpt,
    save_checkpoint_atomic,
    write_manifest,
    trainable_named_parameters,
    SparseRandomProjection,
    optimizer_geometry_fields,
)

# Reuse the validated data / validation / grad-norm helpers from the reference trainer.
import xai_train as xt  # noqa: E402
from xai_train import create_data_loaders, validate, compute_gradient_norms  # noqa: E402


# =============================================================================
# Constants
# =============================================================================

#: Fixed, run-INDEPENDENT projection seed so EVERY run of the experiment shares one
#: low-dimensional coordinate system (briefing §4 invariant 2 / §7 manifest). Recorded
#: in the manifest; consumed by the extractor §6.2 and the d90 sweep §9.1.
PROJECTION_SEED = 1234
#: Default projected dimension for the shared random projection (manifest-recorded).
PROJECTION_DIM = 8192
#: Demographic-noise generator seed offset (briefing-mandated: seed + 9973).
DEMO_GEN_SEED_OFFSET = 9973

#: Per-epoch deterministic-shuffle seed multiplier. The Phase-2 train DataLoader's shuffle
#: generator is reseeded each epoch to ``seed * EPOCH_SHUFFLE_MULT + epoch`` so resuming at the
#: START of epoch E reproduces the EXACT data order of epoch E (a pure function of seed+epoch,
#: independent of how far the global RNG has advanced). Large odd prime to decorrelate seeds.
EPOCH_SHUFFLE_MULT = 100003

#: Version tag stamped into every saved ``rng_state`` so a future loader can detect a
#: schema change. Bump if the set of saved RNG streams changes.
RNG_VERSION = 1

#: Velocity-autocorrelation lags accumulated inline for §6.3(c) (small lags only).
VELOCITY_LAGS: Tuple[int, ...] = (1, 2, 4, 8)

#: Disk-watchdog PAUSE sentinel (created by xAI/scripts/disk_watchdog.sh when the /blue group quota
#: crosses the HARD free-space threshold). The trainer's emit path polls this file BEFORE writing a
#: checkpoint and DEFERS emission while it exists; the disk watchdog removes it once free space
#: recovers past the RELEASE threshold. Training itself NEVER blocks — only checkpoint emission is
#: gated, so the disk can never be filled by a runaway producer while the watchdog holds the brake.
#: Path is relative to the project root (xAI/output/preadapt/), matching the watchdog + collector.
DISK_PAUSE_SENTINEL = os.path.normpath(
    os.path.join(_PROJECT_ROOT, "xAI", "output", "preadapt", "_DISK_PAUSE"))

#: Max time (seconds) the emit path will sleep-poll for the PAUSE sentinel to clear before GIVING
#: UP on this one emission (logging a SKIP) and returning to TRAINING. We never block training
#: indefinitely on disk pressure: if the pause outlasts this budget the checkpoint for this step is
#: simply skipped (the adaptive sampler / epoch-end will emit the next opportunity once disk frees).
DISK_PAUSE_MAX_WAIT_S = 30 * 60  # 30 minutes

#: Poll interval (seconds) while waiting for the PAUSE sentinel to clear.
DISK_PAUSE_POLL_S = 30

# =============================================================================
# Collector backpressure (REDESIGN: collector throughput is the limiter)
# =============================================================================
#
# In the redesigned pipeline the two L4 collectors are the bottleneck, not the B200
# trainer. The trainer emits checkpoints on the dense motion sampler; if it emitted
# freely the work queue (<output_dir>/checkpoints, a PURE WORK QUEUE — kept checkpoints
# are MOVED to <run>/kept/ by the collector) would grow without bound and fill /blue.
#
# BACKPRESSURE: after each emit the trainer checks the PENDING depth
# (preadapt_common.count_pending: ready+unclaimed *.pt in the watch dir). If it EXCEEDS
# the high-water mark it PAUSES — it stops training/emitting and sleep-polls until the
# depth drops BELOW the low-water mark, then resumes. The hysteresis (high>low) avoids
# thrashing. This is what makes COLLECTOR THROUGHPUT the limiter: early Phase-2 training,
# where weights change fast and the sampler emits densely (>=25 ckpts in epoch 1), will
# CRAWL because the trainer constantly waits for collectors to drain the queue; late
# training, where weight change slows and emissions are rare, runs at full B200 speed
# (the queue stays well below the high mark, so backpressure never engages).
#
# This is orthogonal to the disk-watchdog pause: BOTH can pause emission. The disk pause
# is a hard safety brake (group quota); backpressure is the steady-state flow control.

#: Default PENDING high-water mark: pause when count_pending(watch_dir) EXCEEDS this.
BACKPRESSURE_HIGH_DEFAULT = 6
#: Default PENDING low-water mark: resume once count_pending(watch_dir) drops BELOW this.
BACKPRESSURE_LOW_DEFAULT = 2
#: Poll interval (seconds) while paused on backpressure waiting for the queue to drain.
BACKPRESSURE_POLL_S = 15


# =============================================================================
# RNG-state capture / restore (resume support)
# =============================================================================

def epoch_shuffle_generator(seed: int, epoch: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded deterministically from ``(seed, epoch)`` for the
    Phase-2 train DataLoader's per-epoch shuffle.

    Reseeding the sampler generator from ``seed * EPOCH_SHUFFLE_MULT + epoch`` at the START of
    each epoch makes the data order for epoch E a pure function of ``(seed, epoch)`` — it does
    NOT depend on how many random draws happened before (augmentation, noise, validation, etc.).
    That is exactly what bit-exact resume needs: re-entering ``run_phase2`` at the start of
    epoch E reproduces the identical batch order of epoch E regardless of the resume path.
    """
    g = torch.Generator()
    g.manual_seed((seed * EPOCH_SHUFFLE_MULT + int(epoch)) & 0x7FFFFFFFFFFFFFFF)
    return g


def capture_rng_state(demo_gen: Optional[torch.Generator] = None,
                      optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Capture every RNG stream needed to resume bit-exactly (briefing reproducibility / §5).

    Saves the global torch CPU RNG, all CUDA device RNGs, the numpy global state, the python
    ``random`` state, and — when given — the StableEvolutionSOAP demographic generator. On CUDA
    the optimizer derives a device-matched generator (``_demo_gen_cache``) seeded ONCE from the
    CPU ``demographic_generator``; that cached device generator carries the live noise stream, so
    we save BOTH the CPU generator state and every cached device generator's state. All generator
    states are CPU ``ByteTensor``s (picklable by ``torch.save``).
    """
    rng: Dict[str, Any] = {
        "rng_version": RNG_VERSION,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if demo_gen is not None:
        rng["demo_gen"] = demo_gen.get_state()
    # Cached device-matched demographic generators (the live CUDA noise stream).
    if optimizer is not None:
        cache = getattr(optimizer, "_demo_gen_cache", None)
        if cache:
            demo_cache: Dict[str, Any] = {}
            for dev, g in cache.items():
                demo_cache[str(dev)] = {"index": _device_index(dev), "state": g.get_state()}
            rng["demo_gen_cache"] = demo_cache
    return rng


def _device_index(dev: Any) -> Optional[int]:
    """Best-effort CUDA index for a device key used in ``_demo_gen_cache`` (a torch.device)."""
    try:
        d = torch.device(dev)
    except Exception:
        return None
    if d.type != "cuda":
        return None
    return 0 if d.index is None else int(d.index)


def restore_rng_state(rng: Dict[str, Any], demo_gen: Optional[torch.Generator] = None,
                      optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    """Restore every RNG stream captured by :func:`capture_rng_state` in place.

    Restores torch CPU + CUDA RNGs, numpy, python ``random``, the demographic generator (CPU),
    and rebuilds the optimizer's ``_demo_gen_cache`` device generators with their saved state so
    the CUDA noise stream continues bit-identically. Tolerant of a state saved on a machine with
    a different CUDA-device count (only restores the streams it can).
    """
    if rng.get("rng_version") != RNG_VERSION:
        print(f"[preadapt_train] WARNING: rng_version mismatch "
              f"(saved={rng.get('rng_version')} expected={RNG_VERSION}); restoring best-effort.")
    torch.set_rng_state(_as_byte_tensor(rng["torch"]))
    if torch.cuda.is_available() and rng.get("cuda"):
        saved = [_as_byte_tensor(s) for s in rng["cuda"]]
        n = torch.cuda.device_count()
        if len(saved) == n:
            torch.cuda.set_rng_state_all(saved)
        else:
            for i in range(min(n, len(saved))):
                torch.cuda.set_rng_state(saved[i], device=i)
    np.random.set_state(rng["numpy"])
    random.setstate(_as_python_random_state(rng["python"]))
    if demo_gen is not None and "demo_gen" in rng:
        demo_gen.set_state(_as_byte_tensor(rng["demo_gen"]))
    if optimizer is not None and rng.get("demo_gen_cache"):
        cache = getattr(optimizer, "_demo_gen_cache", None)
        if cache is None:
            cache = {}
            optimizer._demo_gen_cache = cache
        for _dev_str, entry in rng["demo_gen_cache"].items():
            idx = entry.get("index")
            if idx is None or not torch.cuda.is_available() or idx >= torch.cuda.device_count():
                continue
            dev = torch.device(f"cuda:{idx}")
            g = torch.Generator(device=dev)
            g.set_state(_as_byte_tensor(entry["state"]))
            cache[dev] = g


def _as_byte_tensor(x: Any) -> torch.Tensor:
    """A ``torch.Generator`` state round-trips through ``torch.save`` as a uint8 tensor; make
    sure it is a CPU ByteTensor before handing it back to ``set_rng_state`` / ``set_state``."""
    if isinstance(x, torch.Tensor):
        return x.cpu().to(torch.uint8)
    return torch.as_tensor(x, dtype=torch.uint8)


def _as_python_random_state(x: Any) -> tuple:
    """``random.getstate()`` returns a tuple-of-tuples; ``torch.save`` may round-trip the inner
    sequence as a list. ``random.setstate`` requires the inner sequence be a tuple of ints."""
    version, internalstate, gauss_next = x
    return (version, tuple(int(v) for v in internalstate), gauss_next)


def build_train_ckpt_with_rng(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    demo_gen: Optional[torch.Generator],
    **meta: Any,
) -> Dict[str, Any]:
    """Wrap :func:`preadapt_common.build_train_ckpt`, additionally embedding a full ``rng_state``
    sub-dict (torch CPU + CUDA, numpy, python, demographic generator) captured at call time.

    The caller passes the usual metadata (``step``/``epoch``/``phase``/``condition``/``variant``/
    ``seed``/``run_id``/...); ``global_step``, ``epoch``, ``phase``, ``seed`` and ``rng_version``
    are guaranteed present on the returned dict so a resume can read them back without parsing
    the filename. ``global_step`` mirrors ``step`` (resume reads ``global_step``)."""
    state = build_train_ckpt(model, optimizer=optimizer, **meta)
    state["rng_state"] = capture_rng_state(demo_gen=demo_gen, optimizer=optimizer)
    # Resume-critical scalars, normalized so the loader never has to parse the filename.
    state.setdefault("global_step", int(meta.get("step", 0)))
    state["rng_version"] = RNG_VERSION
    return state


def resolve_resume_checkpoint(resume: Optional[str], ckpt_dir: str) -> Optional[str]:
    """Resolve ``--resume`` to a concrete checkpoint path (or ``None`` for a fresh run).

    ``None``        -> fresh run (returns None).
    explicit path   -> returned as-is if it exists (else raises).
    ``'auto'``      -> the latest ``step*.pt`` in ``ckpt_dir`` by embedded step number; if none,
                       ``phase1_final.pt`` if present; else None (fresh)."""
    if resume is None:
        return None
    if resume != "auto":
        if not os.path.exists(resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume}")
        return resume
    steps = []
    for path in glob.glob(os.path.join(ckpt_dir, "step*.pt")):
        m = re.search(r"step(\d+)\.pt$", os.path.basename(path))
        if m:
            steps.append((int(m.group(1)), path))
    if steps:
        steps.sort()
        latest = steps[-1][1]
        print(f"[preadapt_train] --resume auto -> latest Phase-2 checkpoint {os.path.basename(latest)}")
        return latest
    p1 = os.path.join(ckpt_dir, "phase1_final.pt")
    if os.path.exists(p1):
        print("[preadapt_train] --resume auto -> phase1_final.pt (no step*.pt found); "
              "skipping Phase 1, starting Phase 2 fresh.")
        return p1
    print("[preadapt_train] --resume auto -> no checkpoint found; starting a FRESH run.")
    return None


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PhenoVision pretraining-as-preadaptation two-phase trainer "
                    "(one condition x one variant per process)."
    )
    # --- experiment identity ---
    p.add_argument("--condition", required=True, choices=list(pm.CONDITIONS),
                   help="starting 'evolutionary history': imagenet | mae | plantclef (VT) | naive")
    # Accept both --optimizer and --variant as aliases for the variant choice.
    p.add_argument("--optimizer", "--variant", dest="variant", default="stable_evo",
                   choices=["adamw", "stable_evo"],
                   help="optimizer variant for Phase 2: adamw (baseline) | stable_evo")
    p.add_argument("--shared-tokenizer", dest="shared_tokenizer", default="imagenet",
                   choices=["imagenet", "mae", "none"],
                   help="which frozen input stage to share across conditions. "
                        "'imagenet' (default) => exact-control with ImageNet stage; "
                        "'none' => keep each condition's own stage (no exact-control); "
                        "'mae' => overwrite ALL conditions' patch_embed+pos_embed with the MAE stage.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None,
                   help="run output dir; defaults to xAI/output/preadapt/<run_id>")

    # --- resume (bit-exact, epoch-boundary) ---
    p.add_argument("--resume", type=str, default=None,
                   help="resume from a checkpoint. A path to a step*.pt / phase1_final.pt, or the "
                        "literal 'auto' to pick the latest step*.pt in the run's checkpoints dir "
                        "(falling back to phase1_final.pt, else a fresh run). A Phase-2 checkpoint "
                        "rebuilds the Phase-2 optimizer + restores its state and all RNG streams "
                        "and CONTINUES from the next epoch boundary (bit-exact via the per-epoch "
                        "deterministic shuffle); phase1_final.pt skips Phase 1 and starts Phase 2 fresh.")

    # --- training schedule / data ---
    p.add_argument("--num_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=768,
                   help="Phase-1 batch size (frozen backbone)")
    p.add_argument("--phase2_batch_size", type=int, default=384,
                   help="Phase-2 batch size (full model)")
    p.add_argument("--lr", type=float, default=None,
                   help="explicit uniform Phase-2 lr (overrides --blr if set)")
    p.add_argument("--blr", type=float, default=5e-4,
                   help="base lr; effective Phase-2 lr = blr * eff_batch / 256 if --lr unset")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=5)
    # v2 (evolutionary-realism) knobs. Defaults preserve legacy behavior so the OLD-config runs and
    # the live collectors that re-invoke this code are unaffected; the v2 grid sets them explicitly.
    p.add_argument("--lr_schedule", choices=["cosine", "fixed"], default="cosine",
                   help="Phase-2 LR schedule. 'cosine' (legacy) = warmup then half-cycle cosine to "
                        "--min_lr. 'fixed' = warmup then HOLD at the base LR (no decay) — removes the "
                        "imposed-rate confound for the 'when does evolution stop' measurement.")
    p.add_argument("--beta1", type=float, default=None,
                   help="override optimizer momentum (exp_avg) beta1 for BOTH variants. None = variant "
                        "default (stable_evo 0.95, adamw 0.9). v2 sets 0.0 (momentum off: breeder's-eq "
                        "R=h2S has no cross-generation velocity memory; also de-confounds the velocity "
                        "autocorrelation / straightness drift signal). beta2 (the preconditioner / "
                        "G-matrix) is kept.")
    p.add_argument("--beta2", type=float, default=None,
                   help="override beta2 (2nd-moment EMA) for BOTH variants. None = variant default "
                        "(stable_evo 0.95, adamw 0.999).")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="limit training data (for short pilots / smoke runs)")
    p.add_argument("--train_csv", type=str, default="data/inat/train_v1.1.0.csv")
    p.add_argument("--val_csv", type=str, default="data/inat/val_v1.1.0.csv")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--val_subset_size", type=int, default=10000)
    p.add_argument("--repr_subset_size", type=int, default=1000)
    p.add_argument("--test_batch_size", type=int, default=None,
                   help="override ALL batch sizes (small-GPU testing)")

    # --- Phase-1 plateau detection (mirror xai_train) ---
    p.add_argument("--plateau_window", type=int, default=3)
    p.add_argument("--plateau_threshold", type=float, default=0.002)
    p.add_argument("--plateau_patience", type=int, default=2)

    # --- Phase-2 TRAIN-LOSS plateau early stop (v2; default OFF = legacy run-all-epochs) ---
    # "Evolution has stopped" = selection pressure (the train gradient) is exhausted. Mirrors the
    # Phase-1 plateau logic but on a smoothed TRAIN loss at each val_every_n: stop when the loss
    # improvement over the window is < threshold for `patience` consecutive checks. num_epochs stays
    # a hard backstop. (Train-loss-plateau is the safer stop than weight-straightness — it protects
    # held-out fitness, since fitness keeps creeping up while train loss still drops.)
    p.add_argument("--phase2_early_stop", action="store_true",
                   help="enable Phase-2 train-loss-plateau early stop (v2). Default off = legacy.")
    p.add_argument("--phase2_plateau_window", type=int, default=8,
                   help="number of val_every_n checkpoints in the train-loss plateau window.")
    p.add_argument("--phase2_plateau_threshold", type=float, default=0.003,
                   help="min train-loss improvement across the window to count as still-descending; "
                        "below it for --phase2_plateau_patience consecutive windows => converged.")
    p.add_argument("--phase2_plateau_patience", type=int, default=3)

    # --- adaptive sampler (briefing §3) ---
    p.add_argument("--sampler_delta", type=float, default=0.10,
                   help="scale-free per-layer net-displacement trigger threshold (sum over layers)")
    p.add_argument("--sampler_min_step", type=int, default=20,
                   help="min-step floor k: never emit more often than every k Phase-2 steps")
    p.add_argument("--sampler_max_step", type=int, default=500,
                   help="max-step cap K: force an emission at least every K Phase-2 steps")

    # --- demographic noise (stable_evo only) ---
    p.add_argument("--demo_kappa", type=float, default=1.0,
                   help="demographic_kappa (1.0 => N_e == sampling population)")
    p.add_argument("--demo_warmup", type=int, default=200,
                   help="demographic_warmup: steps before noise injection begins")

    # --- preconditioner size cap (stable_evo only; AdamW ignores it) ---
    p.add_argument("--max_precond_dim", type=int, default=2048,
                   help="StableEvolutionSOAP max Kronecker-factor dimension: factors with a "
                        "dimension above this fall back to diagonal Adam (no QL/QR/L/R). Caps "
                        "optimizer-state checkpoint size + extractor host RAM. Ignored by AdamW.")

    # --- logging / validation cadence ---
    p.add_argument("--val_every_n", type=int, default=100,
                   help="validate (and log AUC) every N Phase-2 steps")

    # --- projection (manifest-recorded; shared coordinate system) ---
    p.add_argument("--projection_dim", type=int, default=PROJECTION_DIM)
    p.add_argument("--retention_latest_keep", type=int, default=2)

    # --- collector backpressure (REDESIGN: collector is the bottleneck) ---
    p.add_argument("--backpressure-high", dest="backpressure_high", type=int,
                   default=BACKPRESSURE_HIGH_DEFAULT,
                   help="pause emission/training when PENDING checkpoints in the watch dir EXCEED "
                        "this (the high-water mark). Pairs with --backpressure-low (hysteresis).")
    p.add_argument("--backpressure-low", dest="backpressure_low", type=int,
                   default=BACKPRESSURE_LOW_DEFAULT,
                   help="resume training once PENDING checkpoints drop BELOW this (the low-water "
                        "mark). Must be < --backpressure-high.")

    return p.parse_args(argv)


# =============================================================================
# Run identity / paths
# =============================================================================

def make_run_id(condition: str, variant: str, seed: int) -> str:
    """run_id format mandated by the frozen interface: ``{condition}__{variant}__s{seed}``."""
    return f"{condition}__{variant}__s{seed}"


def resolve_output_dir(args: argparse.Namespace, run_id: str) -> str:
    if args.output_dir:
        return args.output_dir
    return os.path.join(_PROJECT_ROOT, "xAI", "output", "preadapt", run_id)


# =============================================================================
# Shared-tokenizer helper (briefing §I.3/§I.4 + plan C3)
# =============================================================================

def apply_mae_input_stage(model: nn.Module) -> None:
    """Overwrite ``patch_embed.proj.weight/bias`` and ``pos_embed`` in ``model`` with the MAE
    stage (timm ``vit_large_patch16_224.mae``), for ``--shared-tokenizer mae``.

    This is the MAE analog of ``preadapt_models.init_model(exact_control_tokenizer=True)``'s
    ImageNet copy: it makes the frozen input stage byte-identical across all conditions, but
    sourced from the MAE checkpoint instead of ImageNet. pos_embed is interpolated to the
    model grid first if the MAE grid differs (it does not for vit_large_patch16_224, but the
    guard is cheap and matches preadapt_models' handling).
    """
    from util.pos_embed import interpolate_pos_embed  # noqa: E402

    mae_sd = pm.load_source_state_dict("mae")
    # Align pos_embed grid to the model's before copying (no-op when grids match).
    interpolate_pos_embed(model, mae_sd)
    pm._copy_input_stage(model, mae_sd, source_name="mae (shared-tokenizer)")
    print("[preadapt_train] applied MAE shared input stage (patch_embed + pos_embed).")


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Init the condition's model with the requested shared-tokenizer policy.

    --shared-tokenizer:
      * 'imagenet' (default): init_model(exact_control_tokenizer=True) — ImageNet stage in all.
      * 'none'              : init_model(exact_control_tokenizer=False) — each condition keeps its own.
      * 'mae'               : init_model(exact_control_tokenizer=False), then overwrite the input
                              stage with the MAE stage in every condition.
    """
    head_gen = torch.Generator().manual_seed(args.seed + 4242)
    exact_control = (args.shared_tokenizer == "imagenet")
    model = pm.init_model(
        args.condition,
        num_classes=2,
        device="cpu",  # build on CPU, copy stages, then move (avoids GPU churn during copies)
        exact_control_tokenizer=exact_control,
        head_generator=head_gen,
    )
    if args.shared_tokenizer == "mae":
        apply_mae_input_stage(model)
    model = model.to(device)
    return model


# =============================================================================
# Per-layer grouping for displacement / sampler (briefing §3, §5)
# =============================================================================

def layer_key(param_name: str) -> str:
    """Group a trainable parameter name by its 'layer' for per-layer displacement and the
    adaptive sampler (briefing §3 sums *relative* per-layer displacement over layers).

    Leading-key grouping:
      * ``blocks.N.*`` -> ``blocks.N``
      * ``head.*``     -> ``head``
      * ``norm.*`` / ``fc_norm.*`` -> ``norm`` / ``fc_norm``
      * ``cls_token``  -> ``cls_token``
      * anything else  -> the first dotted component (or the whole name).
    Frozen-stage params never reach here (they are excluded upstream by
    trainable_named_parameters), so patch_embed/pos_embed contribute nothing — as required.
    """
    if param_name.startswith("blocks."):
        parts = param_name.split(".")
        return f"blocks.{parts[1]}"
    if param_name.startswith("head."):
        return "head"
    if param_name.startswith("fc_norm."):
        return "fc_norm"
    if param_name.startswith("norm."):
        return "norm"
    if param_name == "cls_token":
        return "cls_token"
    return param_name.split(".")[0]


def grouped_trainable_layers(model: nn.Module) -> "OrderedDict[str, List[Tuple[str, nn.Parameter]]]":
    """Return an ordered mapping ``layer_key -> [(param_name, param), ...]`` over the
    trainable parameters (frozen input stage excluded via trainable_named_parameters)."""
    groups: "OrderedDict[str, List[Tuple[str, nn.Parameter]]]" = OrderedDict()
    for name, p in trainable_named_parameters(model):
        groups.setdefault(layer_key(name), []).append((name, p))
    return groups


@torch.no_grad()
def snapshot_trainable(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """Detached CPU clones of every trainable parameter (frozen stage excluded), keyed by
    parameter name — used as theta_0 and as the per-checkpoint 'saved' reference for the
    adaptive sampler."""
    snap: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in trainable_named_parameters(model):
        snap[name] = p.detach().to("cpu", torch.float32).clone()
    return snap


@torch.no_grad()
def per_layer_relative_displacement(
    model: nn.Module,
    reference: "OrderedDict[str, torch.Tensor]",
    layer_members: "OrderedDict[str, List[Tuple[str, nn.Parameter]]]",
) -> Dict[str, float]:
    """Per-layer ``||theta_l(t) - theta_l(ref)|| / ||theta_l(ref)||`` (briefing §3).

    Relative (not raw-norm) so the largest layer does not dominate and the threshold means
    the same thing across conditions — the very-naive baseline starts its downstream weights
    at a different scale than ImageNet, and a raw-norm threshold would sample it differently.
    Returns ``{layer_key: relative_displacement}``.
    """
    out: Dict[str, float] = {}
    for lk, members in layer_members.items():
        num_sq = 0.0
        den_sq = 0.0
        for name, p in members:
            ref = reference[name]
            cur = p.detach().to("cpu", torch.float32)
            num_sq += float(torch.sum((cur - ref) ** 2).item())
            den_sq += float(torch.sum(ref ** 2).item())
        den = math.sqrt(den_sq)
        out[lk] = (math.sqrt(num_sq) / den) if den > 0 else 0.0
    return out


@torch.no_grad()
def per_layer_displacement_from_init(
    model: nn.Module,
    init_snap: "OrderedDict[str, torch.Tensor]",
    layer_members: "OrderedDict[str, List[Tuple[str, nn.Parameter]]]",
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Return ``(global_disp, per_layer_disp, per_layer_relative_disp)`` of ``theta_t`` from
    ``theta_0`` over trainable params (briefing §5). Global is the L2 norm of the full
    trainable displacement; per-layer is the raw L2 per layer; per-layer-relative divides by
    the layer's init norm (the sampler signal, also logged here)."""
    per_layer: Dict[str, float] = {}
    per_layer_rel: Dict[str, float] = {}
    total_sq = 0.0
    for lk, members in layer_members.items():
        num_sq = 0.0
        den_sq = 0.0
        for name, p in members:
            ref = init_snap[name]
            cur = p.detach().to("cpu", torch.float32)
            d_sq = float(torch.sum((cur - ref) ** 2).item())
            num_sq += d_sq
            den_sq += float(torch.sum(ref ** 2).item())
        per_layer[lk] = math.sqrt(num_sq)
        den = math.sqrt(den_sq)
        per_layer_rel[lk] = (math.sqrt(num_sq) / den) if den > 0 else 0.0
        total_sq += num_sq
    return math.sqrt(total_sq), per_layer, per_layer_rel


# =============================================================================
# Velocity ring buffer (briefing §5 / §6.3c)
# =============================================================================

class VelocityTracker:
    """Accumulate per-step velocity quantities cheaply (briefing §5 / §6.3): the projected
    step velocity ``v_t = P^T(theta_t - theta_{t-1})`` (the dimension-reducing map, so the
    autocorrelations are computed in the shared low-dim coordinate system at O(d) memory),
    a small ring buffer of recent ``v_t`` for lags ``tau in VELOCITY_LAGS``, running
    ``sum_t v_t . v_{t+tau}`` for each lag, and the accumulated path length ``sum ||v_t||``.

    Everything threads through the shared :class:`SparseRandomProjection` so the velocity
    coordinates live in the same frame as the extractor's §6.2 projected weights.
    """

    def __init__(self, projection: SparseRandomProjection, lags: Tuple[int, ...] = VELOCITY_LAGS):
        self.proj = projection
        self.lags = lags
        self._buf: deque = deque(maxlen=max(lags) + 1)
        self.dot_sums: Dict[int, float] = {tau: 0.0 for tau in lags}
        self.dot_counts: Dict[int, int] = {tau: 0 for tau in lags}
        self.path_length: float = 0.0
        self.sq_norm_sum: float = 0.0     # sum_t ||v_t||^2 (autocorr denominator)
        self.n_steps: int = 0

    @torch.no_grad()
    def update(self, flat_delta: torch.Tensor) -> float:
        """Push the projected velocity for a step given the *full* trainable displacement
        ``flat_delta = theta_t - theta_{t-1}`` (1-D tensor on any device). Returns ``||v_t||``
        in the projected frame. Updates path length and the running lagged dot products."""
        v = self.proj.rmatvec(flat_delta).to("cpu", torch.float32)
        vnorm = float(v.norm().item())
        self.path_length += vnorm
        self.sq_norm_sum += vnorm * vnorm
        self.n_steps += 1
        # Running lagged dot products against earlier velocities still in the buffer.
        for tau in self.lags:
            if len(self._buf) >= tau:
                v_prev = self._buf[-tau]
                self.dot_sums[tau] += float(torch.dot(v_prev, v).item())
                self.dot_counts[tau] += 1
        self._buf.append(v)
        return vnorm

    def autocorr(self, tau: int) -> float:
        """Normalized velocity autocorrelation at lag ``tau`` (briefing §6.3c):
        ``mean_t (v_t . v_{t+tau}) / mean_t ||v_t||^2``. NaN until enough steps."""
        if self.dot_counts.get(tau, 0) == 0 or self.n_steps == 0:
            return float("nan")
        mean_dot = self.dot_sums[tau] / self.dot_counts[tau]
        mean_sq = self.sq_norm_sum / self.n_steps
        return (mean_dot / mean_sq) if mean_sq > 0 else float("nan")


# =============================================================================
# Inline global §6.2 projection (collector-speed optimization)
# =============================================================================

@torch.no_grad()
def compute_global_proj_coords(
    model: nn.Module, projection: SparseRandomProjection,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute the GLOBAL §6.2 projected coordinates ``P^T theta(t) in R^d`` for this
    checkpoint and the projection meta, to be stored INLINE in the checkpoint.

    ``theta(t) = flatten_trainable(model)`` is the trainable parameter vector (frozen input
    stage EXCLUDED, ``preadapt_common.trainable_named_parameters`` order); ``projection`` is the
    SAME run-shared :class:`SparseRandomProjection` instance the trainer already uses for the
    velocity tracker, so the coordinates live in the experiment's one shared frame (manifest
    ``projection`` / seed ``PROJECTION_SEED``) — bit-identical to what the extractor would have
    recomputed. The result is a small ``R^d`` vector (d=PROJECTION_DIM=8192 -> ~32 KB float32),
    stored on CPU.

    WHY INLINE: this single full-D ``rmatvec`` over the ~302M-coord trainable vector is the
    DOMINANT per-checkpoint cost of the collector's §6.2 ``block_weights`` extract (it scatters
    the whole 302M coords). Computing it once HERE — where ``theta`` is already flattened on the
    GPU each step for the velocity tracker — and saving the tiny ``R^d`` result in the checkpoint
    lets the collector read ``ckpt['proj_coords']`` instead of re-projecting 302M coordinates,
    removing that dominant rmatvec from every per-checkpoint extraction (target T_ex ~3 min).

    Returns ``(coords_cpu_float32_1d, proj_meta)`` where ``proj_meta = projection.to_meta()``
    (method/seed/d/D_trainable/density) lets the consumer sanity-check seed/d against its own
    projection. On any failure returns ``(None, None)`` so emission never breaks on a projection
    hiccup (the collector then falls back to recomputing).
    """
    try:
        theta = pc.flatten_trainable(model)  # 1-D trainable vector (frozen stage excluded)
        if theta.numel() != projection.D_trainable:
            print(f"[preadapt_train] inline proj skipped: D_trainable mismatch "
                  f"(model {theta.numel()} vs projection {projection.D_trainable}).")
            return None, None
        coords = projection.rmatvec(theta)  # R^D -> R^d (L2-preserving reducing direction)
        coords_cpu = coords.detach().to("cpu", torch.float32)
        return coords_cpu, projection.to_meta()
    except Exception as e:  # never break emission on a projection hiccup
        print(f"[preadapt_train] inline proj skipped: {e!r}")
        return None, None


# =============================================================================
# Optimizer scalars (variant-aware, briefing §5)
# =============================================================================

@torch.no_grad()
def global_update_norm(model: nn.Module) -> float:
    """L2 norm of the gradients over trainable params, as a proxy for the global update norm
    available every step (briefing §5). Computed before zero_grad."""
    total_sq = 0.0
    for _, p in trainable_named_parameters(model):
        if p.grad is not None:
            total_sq += float(p.grad.detach().norm(2).item()) ** 2
    return math.sqrt(total_sq)


def optimizer_scalars(optimizer: torch.optim.Optimizer, variant: str) -> Dict[str, float]:
    """Variant-aware optimizer scalars for the inline §5 stream.

    Common: current lr (param-group 0). For stable_evo additionally: ``mean_exponent()``,
    ``demo_T()``, ``demo_trace()``, ``loss_tax()`` (the optimizer's own readers). AdamW
    exposes none of these — only lr is recorded.
    """
    scalars: Dict[str, float] = {}
    if optimizer.param_groups:
        scalars["lr"] = float(optimizer.param_groups[0].get("lr", float("nan")))
    if variant == "stable_evo":
        for name, fn in (
            ("mean_exponent", getattr(optimizer, "mean_exponent", None)),
            ("demo_T", getattr(optimizer, "demo_T", None)),
            ("demo_trace", getattr(optimizer, "demo_trace", None)),
            ("loss_tax", getattr(optimizer, "loss_tax", None)),
        ):
            if callable(fn):
                try:
                    scalars[name] = float(fn())
                except Exception:
                    scalars[name] = float("nan")
    return scalars


# =============================================================================
# Disk-watchdog PAUSE hook (briefing C5 disk safety — WARN + AUTO-PAUSE)
# =============================================================================

def await_disk_pause_clear(
    sentinel: str = DISK_PAUSE_SENTINEL,
    max_wait_s: float = DISK_PAUSE_MAX_WAIT_S,
    poll_s: float = DISK_PAUSE_POLL_S,
    *,
    sleep_fn=time.sleep,
    exists_fn=os.path.exists,
    now_fn=time.monotonic,
) -> bool:
    """Gate checkpoint emission on the disk watchdog's PAUSE sentinel (plan C5 auto-pause).

    Returns ``True`` if it is OK to emit a checkpoint now (the sentinel is absent, or it cleared
    within ``max_wait_s``); returns ``False`` if the sentinel is STILL present after waiting up to
    ``max_wait_s`` — in which case the caller SKIPS this one emission and keeps TRAINING (we never
    block training indefinitely on disk pressure).

    Behavior:
      * No sentinel -> return True immediately (the common, fast path; one cheap ``os.path.exists``).
      * Sentinel present -> log once, then sleep-poll every ``poll_s`` seconds until it clears
        (return True) or ``max_wait_s`` elapses (log + return False).

    The ``sleep_fn`` / ``exists_fn`` / ``now_fn`` seams keep the helper unit-testable without a real
    filesystem clock (the smoke test injects a pre-created sentinel + a 1s budget).
    """
    if not exists_fn(sentinel):
        return True
    print(f"  DISK_PAUSE active — deferring checkpoint emission (sentinel={sentinel}; "
          f"polling every {poll_s:.0f}s, max wait {max_wait_s / 60:.0f} min). Training continues.")
    deadline = now_fn() + max_wait_s
    while exists_fn(sentinel):
        if now_fn() >= deadline:
            print(f"  DISK_PAUSE still active after {max_wait_s / 60:.0f} min — SKIPPING this "
                  f"checkpoint emission and continuing to TRAIN (will retry at the next emit point).")
            return False
        remaining = deadline - now_fn()
        sleep_fn(min(poll_s, max(0.0, remaining)))
    print("  DISK_PAUSE cleared — resuming checkpoint emission.")
    return True


# =============================================================================
# Collector backpressure hook (REDESIGN flow control)
# =============================================================================

def await_backpressure(
    watch_dir: str,
    high: int = BACKPRESSURE_HIGH_DEFAULT,
    low: int = BACKPRESSURE_LOW_DEFAULT,
    *,
    poll_s: float = BACKPRESSURE_POLL_S,
    sleep_fn=time.sleep,
    count_fn=pc.count_pending,
) -> None:
    """Block training while the collector work queue is full (REDESIGN flow control).

    Reads the PENDING depth of ``watch_dir`` via :func:`preadapt_common.count_pending`
    (ready, unclaimed ``*.pt`` checkpoints awaiting a collector). If that depth is
    ``> high`` the trainer has out-run its collectors: log once, then sleep-poll every
    ``poll_s`` seconds until the depth falls ``< low``, and log on resume. Returns
    immediately (a single ``count_pending`` call) when the queue is below the high mark —
    the common case in late training.

    The high/low hysteresis (``high > low``) prevents thrashing: we only pause once the
    queue is genuinely backed up and only resume once it has drained with real headroom.
    Unlike the disk-pause gate this has NO timeout and never SKIPS the emission — the
    whole point is to throttle the trainer to collector throughput, so it waits as long
    as it takes. Training itself is what blocks here (this is called between emits in the
    Phase-2 loop), which deliberately makes early, dense-emission training crawl and lets
    late, sparse-emission training run at full speed.

    ``sleep_fn`` / ``count_fn`` are injectable seams for unit testing without a real
    filesystem or clock.
    """
    pending = count_fn(watch_dir)
    if pending <= high:
        return
    print(f"  BACKPRESSURE: {pending} pending > high ({high}), pausing training "
          f"(waiting for collectors to drain below {low}; polling every {poll_s:.0f}s).")
    while count_fn(watch_dir) >= low:
        sleep_fn(poll_s)
    print(f"  BACKPRESSURE cleared — {count_fn(watch_dir)} pending < low ({low}); "
          f"resuming training.")


# =============================================================================
# Phase 1 — Equalization (ecological fitting)
# =============================================================================

def run_phase1(
    model: nn.Module,
    train_dl,
    val_dl,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    scalar: ScalarStore,
    run_id: str,
    t_start: float,
) -> Tuple[int, int, Dict[str, float]]:
    """Phase 1: freeze the entire backbone, train the head only (Adam) until the AUC
    plateaus. Returns ``(global_step, end_epoch, ecological_fitting_metrics)``.

    Mirrors ``xai_train.run_phase1`` (plateau on auc_roc_mean, PHASE_TRANSITION log,
    phase1_final via save_checkpoint_atomic) but logs to the shared ScalarStore and records
    the converged frozen-head metric explicitly as the ecological-fitting readout.
    Phase 1 is identical across the two optimizer variants.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Frozen Backbone (Equalization = ecological fitting)")
    print("=" * 60)

    # Freeze everything except the head.
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("head."))
    head_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters (head only): {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.Adam(head_params, lr=1e-3)
    phase1_args = argparse.Namespace(
        accum_iter=1, warmup_epochs=0, lr=1e-3, min_lr=1e-4, epochs=args.num_epochs,
    )

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    plateau_metrics = deque(maxlen=args.plateau_window)
    plateau_count = 0
    converged = False
    epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        n_batches = len(train_dl)
        print(f"\n--- Phase 1, Epoch {epoch} ---")
        for batch_idx, batch in enumerate(train_dl):
            samples = batch[0].to(device, non_blocking=True)
            targets = batch[-1].to(device, non_blocking=True)
            lr_sched.adjust_learning_rate(optimizer, batch_idx / n_batches + epoch, phase1_args)

            # bf16 autocast, no GradScaler (Phase 1 head-only; keep precision consistent).
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(samples)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_value = float(loss.item())
            global_step += 1

            wall = time.time() - t_start
            scalar.add(args.condition, run_id, global_step, wall, "phase1_train_loss", loss_value)

            if global_step % args.val_every_n == 0:
                vm = validate(model, val_dl, criterion, device)
                _log_val_metrics(scalar, args.condition, run_id, global_step, wall, vm,
                                 prefix="phase1_")
                print(f"  VAL step={global_step} val_loss={vm['val_loss']:.4f} "
                      f"auc_roc_mean={vm['auc_roc_mean']:.4f} auc_pr_mean={vm['auc_pr_mean']:.4f}")
                plateau_metrics.append(vm["auc_roc_mean"])
                if len(plateau_metrics) == args.plateau_window:
                    improvement = plateau_metrics[-1] - plateau_metrics[0]  # AUC: higher better
                    if improvement < args.plateau_threshold:
                        plateau_count += 1
                        print(f"  Plateau ({plateau_count}/{args.plateau_patience}) "
                              f"auc_roc improvement={improvement:.4f}")
                        if plateau_count >= args.plateau_patience:
                            converged = True
                    else:
                        plateau_count = 0
                model.train()

            if converged:
                break
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch} [{batch_idx}/{n_batches}] loss={loss_value:.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if converged:
            print(f"\n*** Phase 1 converged at epoch {epoch}, step {global_step} ***")
            break
    else:
        print(f"\n*** Phase 1 did not converge in {args.num_epochs} epochs — proceeding ***")

    # Final validation = the ecological-fitting readout (linear-readout ceiling on the start).
    eco = validate(model, val_dl, criterion, device)
    wall = time.time() - t_start
    _log_val_metrics(scalar, args.condition, run_id, global_step, wall, eco, prefix="phase1_final_")
    # Explicit ecological-fitting quantities (briefing Part I §7 / plan C3).
    scalar.add(args.condition, run_id, global_step, wall,
               "ecological_fitting_auc_pr", float(eco["auc_pr_mean"]))
    scalar.add(args.condition, run_id, global_step, wall,
               "ecological_fitting_auc_roc", float(eco["auc_roc_mean"]))
    scalar.add(args.condition, run_id, global_step, wall,
               "ecological_fitting_val_loss", float(eco["val_loss"]))
    scalar.flush()

    # Save phase1_final (reusable Phase-2 start) atomically, with full RNG state.
    state = build_train_ckpt_with_rng(
        model, optimizer=None, demo_gen=None, step=global_step, epoch=epoch,
        phase="phase1_final", condition=args.condition, variant=args.variant, seed=args.seed,
        run_id=run_id, val_loss=float(eco["val_loss"]),
        ecological_fitting_auc_pr=float(eco["auc_pr_mean"]),
    )
    save_checkpoint_atomic(state, os.path.join(ckpt_dir, "phase1_final.pt"))

    print(f"\nPHASE_TRANSITION: step={global_step} epoch={epoch} "
          f"val_loss={eco['val_loss']:.4f} auc_roc_mean={eco['auc_roc_mean']:.4f} "
          f"auc_pr_mean={eco['auc_pr_mean']:.4f} "
          f"(ECOLOGICAL_FITTING: auc_pr={eco['auc_pr_mean']:.4f}, val_loss={eco['val_loss']:.4f})")

    return global_step, epoch, eco


def _log_val_metrics(scalar: ScalarStore, condition: str, run_id: str, step: int,
                     wall: float, metrics: Dict[str, float], prefix: str = "") -> None:
    """Log a validate() metrics dict to the ScalarStore (one row per metric)."""
    for k, v in metrics.items():
        scalar.add(condition, run_id, step, wall, f"{prefix}{k}", float(v))


# =============================================================================
# Phase 2 — Evolution
# =============================================================================

def build_phase2_optimizer(
    model: nn.Module, args: argparse.Namespace, eff_batch: int, lr: float,
) -> Tuple[torch.optim.Optimizer, Optional[torch.Generator]]:
    """Build the Phase-2 optimizer over UNIFORM-LR param groups (frozen input stage excluded).

    Returns ``(optimizer, demo_gen)`` where ``demo_gen`` is the CPU ``torch.Generator`` handle
    to the StableEvolutionSOAP demographic-noise generator (``None`` for AdamW). The handle is
    threaded out so its RNG state can be saved/restored for bit-exact resume (the generator is
    NOT recoverable from ``optimizer.state_dict()`` alone — it lives in the param-group config).

    adamw: torch.optim.AdamW(param_groups, lr).
    stable_evo: StableEvolutionSOAP with the tuning-free demographic-noise config from the
    plan (betas (.95,.95), alpha_max .9 / alpha_min .5, kappa .4, damping 1e-2,
    max_update_norm 1.0, max_precond_dim=args.max_precond_dim (caps Kronecker-factor size so
    full-ViT-L optimizer-state checkpoints stay small; default 2048), precondition_frequency 10,
    demographic_noise + match_grad, demographic_kappa=demo_kappa, demographic_batch=eff_batch,
    demographic_warmup=demo_warmup, demographic_generator seeded seed+9973). No precond_power,
    no M — the constant-isotropic canalization-floor case (briefing §6.1 / plan C4).
    """
    param_groups = pm.build_uniform_param_groups(model, weight_decay=args.weight_decay)
    if args.variant == "adamw":
        # default AdamW betas (0.9, 0.999); v2 sets --beta1 0.0 (momentum off).
        b1 = 0.9 if args.beta1 is None else args.beta1
        b2 = 0.999 if args.beta2 is None else args.beta2
        return torch.optim.AdamW(param_groups, lr=lr, betas=(b1, b2)), None

    from optim.stable_evolution_optimizer import StableEvolutionSOAP
    demo_gen = torch.Generator().manual_seed(args.seed + DEMO_GEN_SEED_OFFSET)
    # default StableEvo betas (0.95, 0.95); v2 sets --beta1 0.0 (drop momentum, keep the
    # 2nd-moment/preconditioner = the metric/G-matrix).
    b1 = 0.95 if args.beta1 is None else args.beta1
    b2 = 0.95 if args.beta2 is None else args.beta2
    optimizer = StableEvolutionSOAP(
        param_groups,
        lr=lr,
        betas=(b1, b2),
        weight_decay=args.weight_decay,
        alpha_max=0.9,
        alpha_min=0.5,
        kappa=0.4,
        damping=1e-2,
        max_update_norm=1.0,
        max_precond_dim=args.max_precond_dim,
        precondition_frequency=10,
        demographic_noise=True,
        demographic_match_grad=True,
        demographic_kappa=args.demo_kappa,
        demographic_batch=eff_batch,
        demographic_warmup=args.demo_warmup,
        demographic_generator=demo_gen,
    )
    return optimizer, demo_gen


@torch.no_grad()
def probe_function_state(model: nn.Module, repr_dl, device: torch.device):
    """Snapshot the model's FUNCTION on the fixed probe set (briefing §6 / v2 C3): per-(example,
    class) sigmoid probabilities + per-example BCE loss. Used to measure FUNCTION-space change
    between checkpoints (the phenotype), distinct from weight-space drift and from the scalar loss
    — it catches iso-loss output rearrangement (re-weighting which examples are fit) that the mean
    loss hides. Returns ``(probs[N,C], perex_loss[N])`` on CPU, or ``None`` if repr_dl is None."""
    if repr_dl is None:
        return None
    was_training = model.training
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    probs_chunks, loss_chunks = [], []
    for batch in repr_dl:
        x = batch[0].to(device, non_blocking=True)
        y = batch[-1].to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        probs_chunks.append(torch.sigmoid(logits).cpu())
        loss_chunks.append(bce(logits, y).mean(dim=1).cpu())   # per-example mean over classes
    if was_training:
        model.train()
    return torch.cat(probs_chunks), torch.cat(loss_chunks)


def function_space_change(prev, cur) -> Dict[str, float]:
    """Function-space change between two ``probe_function_state`` snapshots (prev->cur):
      * ``fn_pred_churn`` — fraction of (example,class) decisions (prob>0.5) that flipped,
      * ``fn_prob_l1`` — mean |Δ sigmoid prob| (smooth output drift),
      * ``fn_perex_loss_rmsd`` — RMS change in the per-example loss vector (iso-loss rearrangement
        is visible here even when MEAN loss is flat).
    Returns {} if either snapshot is None (e.g. first eval, or repr_dl absent)."""
    if prev is None or cur is None:
        return {}
    p_prev, l_prev = prev
    p_cur, l_cur = cur
    churn = float(((p_cur > 0.5) != (p_prev > 0.5)).float().mean().item())
    prob_l1 = float((p_cur - p_prev).abs().mean().item())
    loss_rmsd = float(((l_cur - l_prev) ** 2).mean().sqrt().item())
    return {"fn_pred_churn": churn, "fn_prob_l1": prob_l1, "fn_perex_loss_rmsd": loss_rmsd}


def run_phase2(
    model: nn.Module,
    train_dl,
    val_dl,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    scalar: ScalarStore,
    run_id: str,
    start_global_step: int,
    start_epoch: int,
    input_state0: dict,
    manifest_base: dict,
    t_start: float,
    resume_ckpt: Optional[Dict[str, Any]] = None,
    repr_dl=None,
) -> Tuple[int, Dict[str, Any]]:
    """Phase 2: unfreeze all, re-freeze the input stage, build the variant optimizer, train in
    bf16 (no GradScaler), drive the adaptive sampler + inline §5 logging.

    ``resume_ckpt`` (optional): a loaded Phase-2 checkpoint dict. When given, the model weights
    have ALREADY been loaded by ``main`` (so the projection/displacement references below are
    taken from the resumed weights, not a fresh init); this function additionally restores the
    optimizer's ``optimizer_state_dict`` and the full ``rng_state`` (torch/cuda/numpy/python +
    demographic generator) AFTER building the optimizer, then continues from ``start_epoch + 1``.
    Because the per-epoch shuffle generator is reseeded deterministically from ``(seed, epoch)``
    at the start of every epoch, re-entering at an epoch boundary reproduces that epoch's exact
    data order. A mid-epoch checkpoint resumes from the START of its epoch (a small re-do of the
    steps already taken within that epoch — noted below).

    Returns the final global step.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 2: Evolution  (variant={args.variant})"
          + ("  [RESUME]" if resume_ckpt is not None else ""))
    print("=" * 60)

    # Rebuild the train loader at the Phase-2 batch size (full model needs more VRAM). The
    # shuffle is driven by a per-epoch deterministic generator (see _epoch_train_loader); we
    # build a base loader here only to read dataset/num_workers and to size n_batches.
    p2_bs = args.test_batch_size or args.phase2_batch_size
    train_dataset = train_dl.dataset
    num_workers = train_dl.num_workers
    eff_batch = p2_bs

    def _epoch_train_loader(epoch: int) -> "torch.utils.data.DataLoader":
        """Phase-2 train DataLoader whose shuffle order is a deterministic function of
        ``(seed, epoch)`` (briefing §5 / resume). A RandomSampler with a per-epoch reseeded
        torch.Generator yields the same permutation for the same ``(seed, epoch)`` on every
        process, so resuming at the start of epoch E reproduces E's exact batch order regardless
        of how the global RNG has advanced. ``drop_last=True`` matches the original loader."""
        g = epoch_shuffle_generator(args.seed, epoch)
        sampler = torch.utils.data.RandomSampler(train_dataset, generator=g)
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=p2_bs, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    # Unfreeze everything, then re-freeze the input stage (patch_embed.* + pos_embed). The
    # uniform param groups then EXCLUDE the frozen stage entirely (no lr=0 hack).
    for p in model.parameters():
        p.requires_grad_(True)
    pm.freeze_input_stage(model)

    lr = args.lr if args.lr is not None else (args.blr * eff_batch / 256.0)
    optimizer, demo_gen = build_phase2_optimizer(model, args, eff_batch, lr)
    # FIXED schedule (v2): set min_lr == base lr so adjust_learning_rate's cosine term is constant
    # after warmup (lr = min_lr + (lr-min_lr)*0.5*(1+cos(.)) = lr). Warmup is kept to avoid the
    # Phase-1->2 unfreeze shock. COSINE (legacy) uses the configured --min_lr floor.
    sched_min_lr = lr if args.lr_schedule == "fixed" else args.min_lr
    phase2_args = argparse.Namespace(
        accum_iter=1, warmup_epochs=args.warmup_epochs, lr=lr, min_lr=sched_min_lr,
        epochs=args.num_epochs,
    )
    print(f"  Phase-2 uniform lr={lr:.6g}  eff_batch={eff_batch}  variant={args.variant}  "
          f"lr_schedule={args.lr_schedule} (min_lr={sched_min_lr:.6g})")

    # --- RESUME: restore optimizer state + every RNG stream (model already loaded by main) ---
    if resume_ckpt is not None:
        if resume_ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            print("  [RESUME] optimizer_state_dict loaded.")
        rng = resume_ckpt.get("rng_state")
        if rng is not None:
            restore_rng_state(rng, demo_gen=demo_gen, optimizer=optimizer)
            print("  [RESUME] RNG state restored (torch/cuda/numpy/python"
                  + ("/demographic" if demo_gen is not None else "") + ").")
        else:
            print("  [RESUME] WARNING: checkpoint has no 'rng_state'; resume will NOT be "
                  "bit-exact (legacy checkpoint).")
        print(f"  [RESUME] continuing from global_step={start_global_step}, "
              f"next epoch={start_epoch + 1}.")

    # --- run-shared projection (briefing §4 invariant 2; recorded in manifest) ---
    D_trainable = pc.trainable_param_numel(model)
    projection = SparseRandomProjection(
        D_trainable=D_trainable, d=args.projection_dim, seed=PROJECTION_SEED, method="achlioptas",
    )
    velocity = VelocityTracker(projection)

    # --- references for displacement / sampler ---
    layer_members = grouped_trainable_layers(model)
    init_snap = snapshot_trainable(model)        # theta_0 (Phase-2 start)
    saved_snap = snapshot_trainable(model)       # 'last saved checkpoint' ref for the sampler
    prev_flat = pc.flatten_trainable(model).detach().to("cpu", torch.float32)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- finalize and write the manifest now that the optimizer + projection exist ---
    geom = optimizer_geometry_fields(optimizer)
    manifest = dict(manifest_base)
    manifest.update({
        "phase2_lr": lr,
        "phase2_eff_batch": eff_batch,
        "optimizer_variant_detected": geom["variant"],
        "M_present": bool(geom["M_present"]),
        "optimizer_state_keys": geom["state_keys"],
        "optimizer_group_keys": geom["group_keys"],
        "n_preconditioned": geom["n_preconditioned"],
        "projection": projection.to_meta(),
        "projection_seed": PROJECTION_SEED,
        "D_trainable": D_trainable,
        "sampler": {"delta": args.sampler_delta, "k": args.sampler_min_step,
                    "K": args.sampler_max_step},
        "retention_policy": {"type": "log2+latest", "latest_keep": args.retention_latest_keep},
        "optimizer_config": _optimizer_config_for_manifest(args, lr, eff_batch),
    })
    write_manifest(os.path.join(args.output_dir, "manifest.json"), **manifest)

    # --- save the Phase-2 init checkpoint (step = start_global_step) ---
    # init_model.pt: model_state_dict at the Phase-2 start (plus the optimizer's state_dict — at a
    # fresh start it is empty, on resume it carries the loaded geometric state). On resume this
    # overwrites the original init_model.pt with the SAME weights only when start_epoch is the
    # phase-transition epoch; for a mid-run resume the init checkpoint reflects the resumed start.
    # INLINE §6.2 global projection here too (cheap; the projection exists at the Phase-2 start).
    init_proj_coords, init_proj_meta = compute_global_proj_coords(model, projection)
    init_state = build_train_ckpt_with_rng(
        model, optimizer=optimizer, demo_gen=demo_gen,
        step=start_global_step, epoch=start_epoch,
        phase="phase2_init", condition=args.condition, variant=args.variant, seed=args.seed,
        run_id=run_id,
        param_names=[name for name, _ in trainable_named_parameters(model)],
        proj_coords=init_proj_coords, proj_meta=init_proj_meta,
    )
    if resume_ckpt is None:
        save_checkpoint_atomic(init_state, os.path.join(ckpt_dir, "init_model.pt"))
    else:
        print("  [RESUME] keeping existing init_model.pt (not overwriting on resume).")

    ladder = RetentionLadder(latest_keep=args.retention_latest_keep)

    global_step = start_global_step
    phase2_start_step = global_step
    phase2_start_epoch = start_epoch + 1
    last_emit_step = global_step
    # n_batches is constant across epochs (same dataset / batch_size / drop_last), so read it
    # once from a representative per-epoch loader.
    n_batches = len(_epoch_train_loader(phase2_start_epoch))

    def emit_checkpoint(step: int, epoch: int, reason: str) -> None:
        """Write a full checkpoint (model + optimizer state + full RNG state) atomically; note the
        step on the retention ladder. The trainer NEVER deletes — the collector owns deletion of
        non-ladder checkpoints (briefing §2). The embedded ``rng_state`` makes the checkpoint a
        bit-exact resume point at the epoch boundary.

        DISK SAFETY (plan C5): before writing, consult the disk watchdog's PAUSE sentinel. If the
        watchdog has paused emission (group /blue free space below the HARD threshold), sleep-poll
        until it clears (up to DISK_PAUSE_MAX_WAIT_S). If still paused after that budget, SKIP this
        emission entirely and return — TRAINING is never blocked; only emission is gated, and the
        sampler/epoch-end will emit again at the next opportunity once disk frees. On a SKIP we do
        NOT advance ``last_emit_step`` / ``saved_snap`` so the sampler keeps signalling to emit.
        """
        nonlocal saved_snap, last_emit_step
        # --- disk-watchdog auto-pause gate (briefing C5: WARN + AUTO-PAUSE) ---
        if not await_disk_pause_clear():
            wall_ = time.time() - t_start
            scalar.add(args.condition, run_id, step, wall_, "checkpoint_skipped_disk_pause", 1.0)
            scalar.flush()
            print(f"  EMIT-SKIP step={step} reason={reason} (disk pause did not clear in time); "
                  f"training continues.")
            return
        wall_ = time.time() - t_start
        # INLINE §6.2 global projection (collector-speed): compute the tiny R^d global coords once
        # HERE (the dominant 302M-coord rmatvec) and store them in the checkpoint so the collector
        # reads them instead of re-projecting 302M coords per checkpoint. Uses the SAME run-shared
        # projection as the velocity tracker, so coordinates match the manifest exactly.
        proj_coords, proj_meta = compute_global_proj_coords(model, projection)
        state = build_train_ckpt_with_rng(
            model, optimizer=optimizer, demo_gen=demo_gen,
            step=step, epoch=epoch, phase="phase2",
            condition=args.condition, variant=args.variant, seed=args.seed, run_id=run_id,
            wall_time=wall_, reason=reason,
            param_names=[name for name, _ in trainable_named_parameters(model)],
            proj_coords=proj_coords, proj_meta=proj_meta,
        )
        path = os.path.join(ckpt_dir, f"step{step:08d}.pt")
        save_checkpoint_atomic(state, path)
        keep = ladder.should_keep(step)
        saved_snap = snapshot_trainable(model)   # reset the sampler reference to here
        last_emit_step = step
        scalar.add(args.condition, run_id, step, wall_, "checkpoint_emitted", 1.0)
        scalar.add(args.condition, run_id, step, wall_, "checkpoint_on_ladder", float(keep))
        scalar.add(args.condition, run_id, step, wall_, "pending_after_emit",
                   float(pc.count_pending(ckpt_dir)))
        scalar.flush()
        print(f"  EMIT ckpt step={step} reason={reason} on_ladder={keep} -> {os.path.basename(path)}")
        # --- collector backpressure gate (REDESIGN): pause TRAINING if the watch dir
        # (a PURE WORK QUEUE — kept ckpts are moved to <run>/kept/) has backed up past the
        # high-water mark, until collectors drain it below the low mark. This is what makes
        # COLLECTOR THROUGHPUT the limiter: dense early-epoch emissions force the trainer to
        # wait; sparse late emissions never trip it. Orthogonal to the disk pause above —
        # both can pause emission/training.
        await_backpressure(ckpt_dir, args.backpressure_high, args.backpressure_low)

    # --- step-0 anchors (briefing §5): pre-update loss, ecological-fitting still in scalars ---
    model.train()

    # --- v2 C2: Phase-2 TRAIN-LOSS plateau early stop state (default off) ---
    p2_plateau_metrics: deque = deque(maxlen=args.phase2_plateau_window)
    p2_plateau_count = 0
    p2_window_loss_sum, p2_window_loss_n = 0.0, 0
    early_stop_meta: Dict[str, Any] = {"early_stop": False, "reason": "", "epoch": None, "step": None}
    # --- v2 C3: function-space change reference (probe-set predictions at the previous val eval) ---
    prev_fn_state = probe_function_state(model, repr_dl, device)  # baseline at Phase-2 start

    for epoch in range(phase2_start_epoch, args.num_epochs):
        model.train()
        # Deterministic per-epoch data order: the loader's shuffle is reseeded from (seed, epoch),
        # so re-entering here at the start of epoch E (incl. on resume) reproduces E's exact order.
        epoch_train_dl = _epoch_train_loader(epoch)
        print(f"\n--- Phase 2, Epoch {epoch} (P2 epoch {epoch - phase2_start_epoch}) ---")
        for batch_idx, batch in enumerate(epoch_train_dl):
            samples = batch[0].to(device, non_blocking=True)
            targets = batch[-1].to(device, non_blocking=True)
            lr_sched.adjust_learning_rate(optimizer, batch_idx / n_batches + epoch, phase2_args)

            # bf16 autocast, NO GradScaler — required (fp16 GradScaler corrupts StableEvo's
            # generative preconditioner) and applied to both variants so precision isn't a confound.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(samples)
                loss = criterion(outputs, targets)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping")
                sys.exit(1)

            loss.backward()
            # Grab grad-derived scalars BEFORE the step / zero_grad clears them.
            upd_norm = global_update_norm(model)
            grad_norms = compute_gradient_norms(model)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            wall = time.time() - t_start

            # --- inline §5 stream ---
            scalar.add(args.condition, run_id, global_step, wall, "train_loss", loss_value)
            p2_window_loss_sum += loss_value; p2_window_loss_n += 1  # v2 C2 plateau window accumulator
            scalar.add(args.condition, run_id, global_step, wall, "global_update_norm", upd_norm)
            for gk, gv in grad_norms.items():
                scalar.add(args.condition, run_id, global_step, wall, gk,
                           float(gv) if math.isfinite(gv) else float("nan"))
            for ok, ov in optimizer_scalars(optimizer, args.variant).items():
                scalar.add(args.condition, run_id, global_step, wall, f"opt_{ok}",
                           ov if math.isfinite(ov) else float("nan"))

            # displacement from theta_0 (global + per-layer + per-layer-relative).
            g_disp, pl_disp, pl_rel = per_layer_displacement_from_init(
                model, init_snap, layer_members)
            scalar.add(args.condition, run_id, global_step, wall, "disp_from_init_global", g_disp)
            for lk in layer_members:
                scalar.add(args.condition, run_id, global_step, wall, "disp_from_init",
                           pl_disp[lk], layer=lk)
                scalar.add(args.condition, run_id, global_step, wall, "disp_from_init_rel",
                           pl_rel[lk], layer=lk)

            # velocity ring buffer (projected) — path length + lagged dot products.
            cur_flat = pc.flatten_trainable(model).detach().to("cpu", torch.float32)
            vnorm = velocity.update(cur_flat - prev_flat)
            prev_flat = cur_flat
            scalar.add(args.condition, run_id, global_step, wall, "velocity_norm", vnorm)
            scalar.add(args.condition, run_id, global_step, wall, "path_length",
                       velocity.path_length)
            for tau in VELOCITY_LAGS:
                ac = velocity.autocorr(tau)
                if math.isfinite(ac):
                    scalar.add(args.condition, run_id, global_step, wall,
                               "velocity_autocorr", ac, head=tau)

            # frozen-embedder assertion (must stay ~0; briefing §6.6 correctness check).
            frozen_drift = pm.assert_frozen_input_stage(model, input_state0)
            scalar.add(args.condition, run_id, global_step, wall,
                       "frozen_embedder_drift", float(frozen_drift))
            if frozen_drift > 1e-6:
                print(f"  WARNING: frozen-embedder drift = {frozen_drift:.3e} (should be ~0) "
                      f"— freezing may be broken; ESCALATE per briefing §6.6.")

            # --- adaptive sampler (briefing §3): scale-free per-layer net displacement ---
            rel = per_layer_relative_displacement(model, saved_snap, layer_members)
            sampler_signal = float(sum(rel.values()))
            scalar.add(args.condition, run_id, global_step, wall,
                       "sampler_signal", sampler_signal)
            steps_since_emit = global_step - last_emit_step
            do_emit = False
            reason = ""
            if steps_since_emit >= args.sampler_min_step and sampler_signal > args.sampler_delta:
                do_emit, reason = True, "displacement"
            elif steps_since_emit >= args.sampler_max_step:
                do_emit, reason = True, "max_step_cap"
            if do_emit:
                emit_checkpoint(global_step, epoch, reason)

            # --- periodic validation (AUC) ---
            phase2_steps = global_step - phase2_start_step
            if phase2_steps % args.val_every_n == 0:
                vm = validate(model, val_dl, criterion, device)
                _log_val_metrics(scalar, args.condition, run_id, global_step, wall, vm)
                scalar.add(args.condition, run_id, global_step, wall,
                           "val_auc_pr_mean", float(vm["auc_pr_mean"]))
                print(f"  VAL step={global_step} (p2={phase2_steps}) "
                      f"val_loss={vm['val_loss']:.4f} auc_pr_mean={vm['auc_pr_mean']:.4f}")

                # --- v2 C3: FUNCTION-space change on the probe set (phenotype drift) ---
                cur_fn_state = probe_function_state(model, repr_dl, device)
                for k, v in function_space_change(prev_fn_state, cur_fn_state).items():
                    scalar.add(args.condition, run_id, global_step, wall, k, v)
                if cur_fn_state is not None:
                    prev_fn_state = cur_fn_state
                model.train()

                # --- v2 C2: TRAIN-LOSS plateau early stop (window-mean train loss at val cadence) ---
                if p2_window_loss_n > 0:
                    win_mean = p2_window_loss_sum / p2_window_loss_n
                    p2_window_loss_sum, p2_window_loss_n = 0.0, 0
                    p2_plateau_metrics.append(win_mean)
                    if len(p2_plateau_metrics) == args.phase2_plateau_window:
                        improvement = p2_plateau_metrics[0] - p2_plateau_metrics[-1]  # loss down => +
                        scalar.add(args.condition, run_id, global_step, wall,
                                   "phase2_trainloss_improvement", float(improvement))
                        if improvement < args.phase2_plateau_threshold:
                            p2_plateau_count += 1
                            print(f"  Phase-2 plateau ({p2_plateau_count}/"
                                  f"{args.phase2_plateau_patience}) improvement={improvement:.5f} "
                                  f"< {args.phase2_plateau_threshold}")
                        else:
                            p2_plateau_count = 0
                        if args.phase2_early_stop and p2_plateau_count >= args.phase2_plateau_patience:
                            early_stop_meta = {"early_stop": True, "reason": "trainloss_plateau",
                                               "epoch": epoch, "step": global_step}
                            print(f"*** Phase-2 EARLY STOP (train-loss plateau) at epoch {epoch}, "
                                  f"step {global_step} ***")
                            break

            if batch_idx % 20 == 0:
                scalar.flush()
                print(f"  [{batch_idx}/{n_batches}] loss={loss_value:.4f} "
                      f"upd_norm={upd_norm:.4f} disp={g_disp:.4f} sampler={sampler_signal:.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # End-of-epoch (or early-stop): force an emission so the boundary / converged state is captured.
        emit_checkpoint(global_step, epoch,
                        "early_stop" if early_stop_meta["early_stop"] else "epoch_end")
        scalar.flush()
        if early_stop_meta["early_stop"]:
            break

    scalar.flush()
    return global_step, early_stop_meta


def _optimizer_config_for_manifest(args: argparse.Namespace, lr: float, eff_batch: int) -> dict:
    """Optimizer config recorded in the manifest (briefing §7)."""
    cfg = {"variant": args.variant, "lr": lr, "weight_decay": args.weight_decay,
           "eff_batch": eff_batch}
    if args.variant == "stable_evo":
        cfg.update({
            "betas": [0.95, 0.95], "alpha_max": 0.9, "alpha_min": 0.5, "kappa": 0.4,
            "damping": 1e-2, "max_update_norm": 1.0, "precondition_frequency": 10,
            "max_precond_dim": args.max_precond_dim,
            "demographic_noise": True, "demographic_match_grad": True,
            "demographic_kappa": args.demo_kappa, "demographic_batch": eff_batch,
            "demographic_warmup": args.demo_warmup,
            "demographic_generator_seed": args.seed + DEMO_GEN_SEED_OFFSET,
        })
    return cfg


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_id = make_run_id(args.condition, args.variant, args.seed)
    args.output_dir = resolve_output_dir(args, run_id)
    for sub in ("checkpoints", "metrics"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # Reproducibility: seed every global RNG too (data augmentation uses torch/numpy/python),
    # though all experiment-critical stochastic ops thread explicit generators. These are the
    # exact streams captured into each checkpoint's rng_state and restored on resume.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("PhenoVision preadapt trainer (pretraining-as-preadaptation)")
    print("=" * 60)
    print(f"  run_id           : {run_id}")
    print(f"  condition        : {args.condition}")
    print(f"  variant          : {args.variant}")
    print(f"  shared-tokenizer : {args.shared_tokenizer}")
    print(f"  seed             : {args.seed}")
    print(f"  output_dir       : {args.output_dir}")
    print(f"  epochs           : {args.num_epochs}")
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data (reuses the validated loaders; writes subset-index CSVs into output_dir).
    train_dl, val_dl, repr_dl = create_data_loaders(args)

    # Model with the requested shared-tokenizer policy.
    model = build_model(args, device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Snapshot the frozen input stage E(0) for the §6.6 assertion (taken before any training;
    # the input stage is value-identical regardless of phase since it is never updated).
    input_state0 = pm.input_stage_state(model)

    # --- resolve --resume (bit-exact, epoch-boundary) BEFORE training ----------------------
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    resume_path = resolve_resume_checkpoint(args.resume, ckpt_dir)
    resume_ckpt: Optional[Dict[str, Any]] = None
    resume_phase: Optional[str] = None
    if resume_path is not None:
        print(f"[preadapt_train] loading resume checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        resume_phase = resume_ckpt.get("phase")
        # The model weights are common to every checkpoint type; load them now.
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print(f"[preadapt_train] resume: phase={resume_phase} "
              f"epoch={resume_ckpt.get('epoch')} "
              f"global_step={resume_ckpt.get('global_step', resume_ckpt.get('step'))}")

    # Per-epoch deterministic shuffle scheme, recorded in the manifest.
    epoch_shuffle_scheme = (
        f"RandomSampler(generator=torch.Generator(manual_seed="
        f"seed*{EPOCH_SHUFFLE_MULT}+epoch)) reseeded at each Phase-2 epoch start; "
        f"data order is a pure function of (seed, epoch) -> resume at an epoch boundary is bit-exact."
    )

    # Manifest base (architecture + identity; finalized in Phase 2 once optimizer/projection exist).
    manifest_base = {
        "condition": args.condition,
        "variant": args.variant,
        "seed": args.seed,
        "run_id": run_id,
        "shared_tokenizer": args.shared_tokenizer,
        "arch": {"variant": "vit_large_patch16_224", "patch_size": 16, "resolution": 224,
                 "D": 1024, "n_heads": 16, "head_dim": 64, "n_blocks": 24, "num_classes": 2},
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "max_train_samples": args.max_train_samples,
        # --- resume / RNG provenance (briefing §7) ---
        "resume_supported": True,
        "rng_saved": True,
        "rng_version": RNG_VERSION,
        "rng_streams": ["torch", "cuda", "numpy", "python", "demographic_generator"],
        "epoch_shuffle_scheme": epoch_shuffle_scheme,
        "epoch_shuffle_mult": EPOCH_SHUFFLE_MULT,
        "resumed_from": resume_path,
    }

    scalar = ScalarStore(os.path.join(args.output_dir, "metrics", "scalars.parquet"))
    t_start = time.time()
    p2_meta: Dict[str, Any] = {}
    try:
        if resume_phase == "phase2":
            # Resume Phase 2 from the checkpoint's epoch boundary (skip Phase 1 entirely). The
            # per-epoch deterministic shuffle means re-entering at epoch start reproduces the data
            # order; a mid-epoch checkpoint resumes from that epoch's START (small re-do).
            gstep = int(resume_ckpt.get("global_step", resume_ckpt.get("step", 0)))
            p1_epoch = int(resume_ckpt.get("epoch", 0))
            print(f"[preadapt_train] RESUME Phase 2 from epoch {p1_epoch} (start of epoch "
                  f"{p1_epoch + 1}), global_step={gstep}.")
            gstep, p2_meta = run_phase2(
                model, train_dl, val_dl, criterion, device, args, scalar, run_id,
                gstep, p1_epoch, input_state0, manifest_base, t_start,
                resume_ckpt=resume_ckpt, repr_dl=repr_dl)
        elif resume_phase == "phase1_final":
            # phase1_final: skip Phase 1, start Phase 2 FRESH (mirrors the old --resume_phase2).
            gstep = int(resume_ckpt.get("global_step", resume_ckpt.get("step", 0)))
            p1_epoch = int(resume_ckpt.get("epoch", 0))
            # Restore the RNG saved at the phase transition so Phase 2 begins from the same stream.
            rng = resume_ckpt.get("rng_state")
            if rng is not None:
                restore_rng_state(rng)
                print("[preadapt_train] RESUME from phase1_final: RNG restored; "
                      "skipping Phase 1, starting Phase 2 fresh.")
            else:
                print("[preadapt_train] RESUME from phase1_final (legacy, no rng_state): "
                      "skipping Phase 1, starting Phase 2 fresh.")
            gstep, p2_meta = run_phase2(
                model, train_dl, val_dl, criterion, device, args, scalar, run_id,
                gstep, p1_epoch, input_state0, manifest_base, t_start, repr_dl=repr_dl)
        else:
            if resume_ckpt is not None:
                print(f"[preadapt_train] WARNING: resume checkpoint phase={resume_phase!r} is not "
                      "a resumable Phase-2 / phase1_final checkpoint; running fresh from Phase 1.")
            gstep, p1_epoch, _eco = run_phase1(
                model, train_dl, val_dl, criterion, device, args, scalar, run_id, t_start)
            gstep, p2_meta = run_phase2(
                model, train_dl, val_dl, criterion, device, args, scalar, run_id,
                gstep, p1_epoch, input_state0, manifest_base, t_start, repr_dl=repr_dl)
    finally:
        scalar.close()

    # Drop a completion sentinel so the collector knows the run finished (after the last ckpt).
    _es = p2_meta if isinstance(p2_meta, dict) else {}
    with open(os.path.join(args.output_dir, "RUN_COMPLETE"), "w") as f:
        f.write(f"run_id={run_id}\nfinal_step={gstep}\n"
                f"early_stop={_es.get('early_stop', False)}\n"
                f"early_stop_reason={_es.get('reason', '')}\n")
    print(f"\nTRAINING COMPLETE: run_id={run_id} final_step={gstep} "
          f"elapsed={(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
