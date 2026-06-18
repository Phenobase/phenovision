#!/usr/bin/env python3
"""
extractor/extract.py — the extraction DRIVER (§6 block orchestration; CONTRACT item #4).

This is the integration backbone that turns a single checkpoint on disk into a complete
set of §6.1–§6.8 records. It owns the per-run shared, run-invariant RESOURCES (the fixed
seeded probe-train / probe-eval split loaders, the one fixed Hessian/Fisher batch, the
seeded :class:`SparseRandomProjection`, the ScalarStore / ArrayStore, and the FIXED INIT
reference E(0) loaded ONCE from disk). ``build_ctx`` (in ``_ctx.py``) is the only place that
knows how a ``.pt`` becomes an :class:`ExtractCtx`; this module reuses it per checkpoint and
then runs every block in order.

STATELESS per-checkpoint extraction (CONTRACT (B)/(C), REDESIGN 2026-06-16). The single most
important property of this driver: :meth:`Extractor.extract_checkpoint` holds **NO
cross-checkpoint in-memory state**. There is no ``self.prev_model_sd`` / ``self.prev_opt`` /
``self.Q0`` carry-over and no in-memory processed counter. Every per-checkpoint block reads
ONLY (a) the checkpoint itself, (b) the fixed read-only shared resources, and (c) the fixed
INIT reference loaded from disk. As a consequence two collector processes can run
``extract_checkpoint`` on DIFFERENT checkpoints CONCURRENTLY and OUT OF ORDER; the atomic
``claim_for_processing`` rename guarantees exactly one collector processes each checkpoint, and
nothing here races on shared mutable state. The inherently cross-checkpoint diagnostics — §6.1
QL/QR rotation (vs prev / vs init) and the whole of §6.3 trajectory geometry (MSD / straightness
/ velocity-autocorr / motion flat-steep) — have MOVED OUT of the live path to a POST-HOC pass
(CONTRACT (D)) that reads the stored top-k QL/QR eigvecs + projected coords from the per-run
store plus the trainer's inline velocity/path-length log.

Design (mirrors the briefing §2 / plan C4):

* **One :class:`Extractor` per run_id (resource cache only).** The collector keeps one
  Extractor per run_id so the probe set, Hessian batch, projection, out store, and the fixed
  INIT reference are built/loaded once per trajectory and reused read-only (briefing §2.3:
  records and the retention ladder are per-run). The shared comparability invariants (probe set,
  Hessian batch, projection — briefing §4) are identical across runs *by construction* because
  they are built from the same seeds and the manifest-recorded projection seed/dim/method, so two
  Extractors for two runs produce a byte-identical coordinate system. NOTE: because the Extractor
  now holds no per-checkpoint mutable state, two Extractor instances for the SAME run (one per
  collector process) are also safe — they just each build their own copy of the read-only
  resources.

* **Block isolation.** Each §6 block's ``extract(ctx) -> dict`` is wrapped in try/except: a
  failing block logs a full traceback and the loop continues, recording per-block ok/err in
  the returned summary. One bad checkpoint or one numerically unhappy block never takes down
  the collector (briefing §2.2 / §2.3 robustness).

* **Guarded block imports (CONTRACT item #4).** The eight block modules are imported lazily
  via ``importlib`` inside a try/except, so this driver imports cleanly even before every
  sibling block module exists on disk (e.g. before ``block_weights`` / ``block_trajectory`` /
  ``block_probes`` land). A missing block is skipped with a logged warning and recorded in the
  summary as ``{"skipped": "module not found"}``.

* **Generator threading (CONVENTIONS.md §5).** No global RNG. The driver derives a per-run
  base :class:`torch.Generator` from the run seed and, per checkpoint, attaches deterministic
  per-block generators to the ctx (``gen_curvature`` / ``gen_fitness`` / …) seeded from
  ``(seed, step, block_tag)`` so the blocks that need probe noise are reproducible per
  checkpoint without ever touching the global RNG. Blocks that already self-resolve a
  generator from ``(ctx.seed, ctx.step)`` are unaffected; this just gives them a
  driver-supplied one when present.

ARCH: ViT-L/16, D=1024, 24 blocks, 16 heads, head_dim 64, patch16/224, num_classes=2
(reproductive: fruit, flower). The frozen input stage (``patch_embed.*`` / ``pos_embed``) is
never included in any displacement / projection / rank statistic — every block defers to
``trainable_named_parameters`` for that, and ``build_ctx`` loads it but the blocks skip it.
"""

from __future__ import annotations

import gc
import importlib
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- project paths: mirror _ctx.py so PlantCLEF2022 + xAI/py + two_noise import. ----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.dirname(_HERE)                                  # .../xAI/py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PY_DIR))         # .../phenovision
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
_TWO_NOISE = os.path.join(_PROJECT_ROOT, "xAI", "two_noise")
for _p in (_PY_DIR, _PROJECT_ROOT, _PLANTCLEF, _TWO_NOISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preadapt_common import (  # noqa: E402
    ScalarStore,
    ArrayStore,
    SparseRandomProjection,
    trainable_named_parameters,
    trainable_param_numel,
    read_manifest,
)
from extractor._ctx import (  # noqa: E402
    ExtractCtx,
    build_ctx,
    resolve_init_reference_path,
    load_init_model_sd,
)

# Reproductive head: 2 sigmoid outputs (fruit, flower). BCE-with-logits is the training loss.
NUM_CLASSES: int = 2

# Manifest-default projection (must match preadapt_train.PROJECTION_SEED / PROJECTION_DIM so
# the extractor's §6.2 coordinates coincide with the trainer's). Used only as a fallback when
# the run manifest is absent; the manifest is always authoritative when present.
DEFAULT_PROJECTION_SEED: int = 1234
DEFAULT_PROJECTION_DIM: int = 4096
DEFAULT_PROJECTION_METHOD: str = "achlioptas"

#: §6 block run order. weights/trajectory/probes are the three modules a sibling agent may not
#: have written yet — guarded imports below let this driver run with whatever is present.
#: REDESIGN 2026-06-16 (CONTRACT (C)): block_optim no longer computes the live QL/QR rotation
#: (it STORES the top-k eigvecs for the post-hoc pass), and block_trajectory's cross-checkpoint
#: diagnostics (§6.3 MSD / straightness / velocity-autocorr / motion-split) have ALL moved to the
#: POST-HOC pass — block_trajectory stays in the order (kept importable) but writes only a small
#: per-checkpoint-independent summary on the live path. Everything else is per-checkpoint
#: independent (vs the fixed INIT only).
BLOCK_ORDER: Tuple[str, ...] = (
    "block_optim",        # §6.1 optimizer geometric state (headline; eigvecs stored for post-hoc)
    "block_curvature",    # §6.4 curvature / loss-landscape (Hessian spectra)
    "block_weights",      # §6.2 dimension-reduced weights / displacement (vs fixed INIT)
    "block_trajectory",   # §6.3 trajectory geometry (cross-ckpt parts are POST-HOC; near no-op live)
    "block_circuits",     # §6.5 QK/OV circuits (weight-only)
    "block_interp",       # §6.6 ViT interpretability / CKA (vs fixed INIT)
    "block_probes",       # §6.7 per-patch + per-layer linear probes
    "block_fitness",      # §6.8 held-out fitness
)

#: The HEAVY interpretability blocks — the costly probe-set / many-head passes (§6.5/§6.6/§6.7).
#: These are TIERED to every-N checkpoints (``Extractor.heavy_every``) rather than run on every
#: checkpoint, because they dominate T_ex. The remaining (CHEAP) blocks — block_optim,
#: block_curvature, block_weights, block_trajectory, block_fitness — run on EVERY checkpoint.
#: Downstream analysis treats the heavy quantities as present only on heavy checkpoints by design.
HEAVY_BLOCK_TAGS: frozenset = frozenset({"block_circuits", "block_interp", "block_probes"})

#: v2 C4 CPU/GPU split. GPU-set blocks run a model forward/backward (HVP for curvature; forward
#: passes for fitness/interp/probes) -> need a GPU worker. CPU-set blocks are pure linear algebra
#: on the SAVED optimizer-state / weight tensors (no model eval) -> run on many CPU-only workers.
#: With ``block_set="cpu"`` the Extractor skips the probe-loader / Hessian-batch build entirely
#: (no CSV, no dataloaders, lower host RAM). ``block_set="all"`` (default) = legacy: every block.
GPU_BLOCK_TAGS: frozenset = frozenset({"block_curvature", "block_fitness", "block_interp", "block_probes"})
CPU_BLOCK_TAGS: frozenset = frozenset({"block_optim", "block_weights", "block_trajectory", "block_circuits"})

#: Per-block generator attribute the driver attaches to ctx (CONVENTIONS.md §5). A block that
#: looks for ``ctx.gen_<tag>`` finds a driver-supplied, deterministically-seeded generator.
_GEN_ATTRS: Dict[str, str] = {
    "block_curvature": "gen_curvature",
    "block_fitness": "gen_fitness",
    "block_interp": "gen_interp",
    "block_probes": "gen_probes",
    "block_trajectory": "gen_trajectory",
    "block_weights": "gen_weights",
}


def _load_block(name: str):
    """Import one ``extractor.<name>`` block module, returning the module or ``None`` if it is
    absent / fails to import. GUARDS the driver against missing sibling block modules
    (CONTRACT item #4): a missing block is skipped, not fatal."""
    try:
        return importlib.import_module(f"extractor.{name}")
    except Exception as e:  # ModuleNotFoundError or a top-level import error in the block
        print(f"[extract] block {name!r} not importable; skipping. ({e!r})",
              file=sys.stderr, flush=True)
        return None


# =============================================================================
# probe-set / Hessian-batch construction (built ONCE per Extractor; recorded in manifest)
# =============================================================================

def _read_label_csv(csv_path: str) -> Tuple[List[str], np.ndarray]:
    """Read a probe/val CSV in the project's schema (``file_name`` + ``fruiting`` +
    ``flowering``) into ``(file_names, labels[N,2])``. Mirrors ``xai_train.load_csv_data``
    so the probe labels match the training head order exactly (col 0 = fruiting, col 1 =
    flowering -> the model's [fruit, flower] outputs)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    file_names = df["file_name"].tolist()
    labels = df[["fruiting", "flowering"]].values.astype(np.float32)
    return file_names, labels


#: The four joint phenology cells (multilabel: col 0 = fruiting, col 1 = flowering). A row is
#: assigned to exactly one cell by its (fruiting, flowering) bits, so the four cells partition
#: the probe CSV. Cell order is fixed for reproducible per-cell counts in the manifest.
PROBE_CELLS: Tuple[str, ...] = ("neither", "flower_only", "fruit_only", "both")


def _assign_cells(labels: np.ndarray) -> np.ndarray:
    """Map each row's (fruiting, flowering) label pair to one of the four :data:`PROBE_CELLS`
    indices. ``labels`` is ``[N, 2]`` with col 0 = fruiting, col 1 = flowering (the head order).
    Returns an int array of cell indices in ``[0, 4)`` aligned to ``PROBE_CELLS``."""
    fruit = labels[:, 0] > 0.5
    flower = labels[:, 1] > 0.5
    cell = np.empty(labels.shape[0], dtype=np.int64)
    cell[(~fruit) & (~flower)] = 0  # neither
    cell[(~fruit) & (flower)] = 1   # flower_only
    cell[(fruit) & (~flower)] = 2   # fruit_only
    cell[(fruit) & (flower)] = 3    # both
    return cell


def _build_probe_loaders(
    probe_csv: str,
    n_probe: int,
    split_seed: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Build the FIXED probe-train / probe-eval loaders ONCE (briefing §4 invariant 1),
    STRATIFIED over the four joint phenology cells (CONTRACT: balanced probe set).

    A single held-out probe-image set, deterministically and STRATIFIED-subsampled to
    ``n_probe`` images — ``n_probe // 4`` from each of the four joint cells
    {neither, flower_only, fruit_only, both} — and split 50/50 into a probe-train (for §6.7
    linear probes and §6.6 gradient-covariance) and probe-eval (for §6.8 fitness) partition.
    The split is *also* stratified per cell, so probe-train and probe-eval are each balanced
    across the four cells. Everything is seeded by ``split_seed`` and sorted to file order, so
    the same images in the same order feed every checkpoint of every condition.

    If a cell has fewer than ``n_probe // 4`` rows in the CSV, all of its rows are taken and
    the shortfall is recorded in the manifest (``cell_shortfall``); the per-cell split still
    halves whatever was drawn.

    Returns ``(train_loader, eval_loader, ids)`` where ``ids`` records — for the manifest — the
    per-cell counts, the chosen probe-image file names, the split indices, and the
    ``balanced=True`` / ``stratify=joint4`` markers.

    No augmentation on either split (val transform): probe metrics must be comparable, not
    stochastic. The 50/50 fraction is large enough for a stable ridge probe and a stable AUC.
    """
    import timm
    from py.custom_dataset import PhenoDataset

    files, labels = _read_label_csv(probe_csv)
    n_total = len(files)
    if n_total == 0:
        raise ValueError(f"probe CSV {probe_csv!r} has no rows")

    rng = np.random.RandomState(int(split_seed))
    cells = _assign_cells(labels)

    target_per_cell = int(n_probe) // len(PROBE_CELLS)

    # --- STRATIFIED draw + STRATIFIED 50/50 split, per cell -----------------------------------
    train_local: List[int] = []   # selected CSV row indices for probe-train (global indices)
    eval_local: List[int] = []    # selected CSV row indices for probe-eval (global indices)
    per_cell_selected: Dict[str, int] = {}
    per_cell_train: Dict[str, int] = {}
    per_cell_eval: Dict[str, int] = {}
    per_cell_available: Dict[str, int] = {}
    cell_shortfall: Dict[str, int] = {}

    for ci, cname in enumerate(PROBE_CELLS):
        cell_rows = np.where(cells == ci)[0]
        n_avail = int(cell_rows.shape[0])
        per_cell_available[cname] = n_avail
        n_take = min(target_per_cell, n_avail)
        if n_take < target_per_cell:
            cell_shortfall[cname] = int(target_per_cell - n_take)
        if n_take == 0:
            per_cell_selected[cname] = 0
            per_cell_train[cname] = 0
            per_cell_eval[cname] = 0
            continue
        # Seeded subsample within this cell, then a seeded 50/50 split within the cell so both
        # halves are balanced. choice(replace=False) on the cell's own rows.
        chosen = rng.choice(cell_rows, n_take, replace=False)
        cperm = rng.permutation(n_take)
        n_ctrain = n_take // 2
        c_train = chosen[cperm[:n_ctrain]]
        c_eval = chosen[cperm[n_ctrain:]]
        train_local.extend(int(i) for i in c_train)
        eval_local.extend(int(i) for i in c_eval)
        per_cell_selected[cname] = int(n_take)
        per_cell_train[cname] = int(c_train.shape[0])
        per_cell_eval[cname] = int(c_eval.shape[0])

    # Sort each split to file order (deterministic, file-order-stable fixed probe set).
    train_idx = np.sort(np.asarray(train_local, dtype=np.int64))
    eval_idx = np.sort(np.asarray(eval_local, dtype=np.int64))
    n_sel = int(train_idx.shape[0] + eval_idx.shape[0])

    transform_val = timm.data.create_transform(input_size=224, is_training=False)

    def _loader(idx: np.ndarray):
        f = [files[i] for i in idx]
        y = labels[idx]
        ds = PhenoDataset(f, y)
        ds.transform = transform_val
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    train_loader = _loader(train_idx)
    eval_loader = _loader(eval_idx)

    ids = {
        "probe_csv": os.path.abspath(probe_csv),
        "n_total_in_csv": int(n_total),
        "n_probe_selected": int(n_sel),
        "split_seed": int(split_seed),
        "balanced": True,
        "stratify": "joint4",
        "cells": list(PROBE_CELLS),
        "target_per_cell": int(target_per_cell),
        "cell_counts_available": per_cell_available,
        "cell_counts_selected": per_cell_selected,
        "cell_counts_train": per_cell_train,
        "cell_counts_eval": per_cell_eval,
        "cell_shortfall": cell_shortfall,
        "probe_train_files": [files[i] for i in train_idx],
        "probe_eval_files": [files[i] for i in eval_idx],
        "probe_train_indices": [int(i) for i in train_idx],
        "probe_eval_indices": [int(i) for i in eval_idx],
    }
    return train_loader, eval_loader, ids


def _build_hessian_batch(
    eval_loader: Any,
    hessian_batch_size: int,
    device: torch.device,
    *,
    hessian_seed: int = 0,
) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    """Materialize the ONE fixed Hessian/Fisher batch (briefing §4 invariant 3) on ``device``,
    STRATIFIED over the four joint phenology cells (CONTRACT: balanced Hessian batch).

    Drawn deterministically from the probe-eval loader (which is unshuffled and fixed). We
    materialize the whole eval split once (it is small — ~512 images at the balanced default),
    assign each image to its joint cell from its targets, and seed-sample
    ``hessian_batch_size // 4`` per cell (16/cell = 64 at the default). If a cell is short, we
    take all of it (proportional fallback) and note the shortfall. The selection is fixed and
    seeded so sharpness numbers are comparable over time and across conditions.

    Returns ``((images, targets), ids)`` or ``(None, ids)`` if the loader is empty. ``ids``
    records the realized batch size, per-cell counts, and the stratification markers for the
    manifest. We keep the batch on the GPU for the run's lifetime — a single ViT-L Hessian
    batch is small relative to a model forward's activations.
    """
    images_chunks: List[torch.Tensor] = []
    targets_chunks: List[torch.Tensor] = []
    for batch in eval_loader:
        images_chunks.append(batch[0])
        targets_chunks.append(batch[-1])
    if not images_chunks:
        return None, {"hessian_batch_size": 0}

    all_images = torch.cat(images_chunks, dim=0)
    all_targets = torch.cat(targets_chunks, dim=0)
    cells = _assign_cells(all_targets.detach().cpu().numpy().astype(np.float32))

    rng = np.random.RandomState(int(hessian_seed))
    per_cell = max(1, int(hessian_batch_size) // len(PROBE_CELLS))
    sel_idx: List[int] = []
    per_cell_counts: Dict[str, int] = {}
    cell_shortfall: Dict[str, int] = {}
    for ci, cname in enumerate(PROBE_CELLS):
        rows = np.where(cells == ci)[0]
        n_take = min(per_cell, int(rows.shape[0]))
        if n_take < per_cell:
            cell_shortfall[cname] = int(per_cell - n_take)
        per_cell_counts[cname] = int(n_take)
        if n_take > 0:
            sel_idx.extend(int(i) for i in rng.choice(rows, n_take, replace=False))

    sel = np.sort(np.asarray(sel_idx, dtype=np.int64))
    if sel.shape[0] == 0:
        # Fallback: cells empty (e.g. degenerate eval split) — take a head slice.
        sel = np.arange(min(int(hessian_batch_size), all_images.shape[0]), dtype=np.int64)

    images = all_images[sel].to(device)
    targets = all_targets[sel].to(device)
    ids = {
        "hessian_batch_size": int(images.shape[0]),
        "hessian_seed": int(hessian_seed),
        "balanced": True,
        "stratify": "joint4",
        "per_cell_target": int(per_cell),
        "cell_counts": per_cell_counts,
        "cell_shortfall": cell_shortfall,
    }
    return (images, targets), ids


# =============================================================================
# Extractor
# =============================================================================

class Extractor:
    """Per-run extraction driver. Owns the shared comparability resources + per-run carried
    state, and runs the §6 blocks in order on each checkpoint.

    One Extractor per ``run_id`` (the collector keeps a dict keyed by run_id), because
    ``Q0`` / ``prev_model_sd`` / ``prev_opt`` are only meaningful within a single trajectory
    (briefing §2.3). The probe set, Hessian batch, and random projection are built ONCE here
    (briefing §4 invariants) and reused for every checkpoint of this run; the projection seed
    is taken from the run manifest so its coordinates coincide with the trainer's §6.2 /
    d90 coordinate system.
    """

    def __init__(
        self,
        run_dir: str,
        out_store_dir: str,
        device,
        *,
        probe_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        n_probe: int = 1024,
        hessian_batch_size: int = 64,
        probe_batch_size: int = 64,
        num_workers: int = 4,
        projection_seed: Optional[int] = None,
        projection_dim: Optional[int] = None,
        projection_method: Optional[str] = None,
        split_seed: int = 20240601,
        run_id: Optional[str] = None,
        scalar_store: Optional[ScalarStore] = None,
        array_store: Optional[ArrayStore] = None,
        heavy_every: int = 3,
        block_set: str = "all",
        blocks: Optional[Sequence[str]] = None,
    ):
        """Build the per-run shared resources once.

        Args:
            run_dir: the run's checkpoint directory (``<output_dir>/checkpoints``) OR the run
                output directory; the manifest is searched in ``run_dir`` and its parent.
            out_store_dir: directory for this run's ScalarStore + ArrayStore (defaults to a
                ``run_id``-named subdir if ``run_id`` is known). If ``scalar_store`` /
                ``array_store`` are passed in (the collector may share a single out store
                across runs since rows are keyed by run_id), they are used instead of
                creating per-run stores.
            device: torch device for the model + probe passes.
            probe_csv / val_csv: probe-image CSV. ``probe_csv`` is preferred; ``val_csv`` is
                the fallback. At least one must be readable, else forward/backward blocks
                degrade to "skipped" (the optimizer/weight/circuit blocks still run).
            n_probe: number of probe images to subsample, STRATIFIED into n_probe//4 per joint
                phenology cell (default 1024 -> 256/cell), fixed across runs/conditions.
            hessian_batch_size: size of the one fixed Hessian batch, STRATIFIED into
                hessian_batch_size//4 per cell (default 64 -> 16/cell).
            probe_batch_size: dataloader batch size for the probe passes.
            num_workers: dataloader workers.
            projection_seed/dim/method: override the manifest projection spec (normally left
                None so the manifest — hence the trainer's seed — is authoritative).
            split_seed: seed for the deterministic probe subsample + train/eval split.
            run_id: the run id; if None it is read from the manifest.
            scalar_store / array_store: optional shared stores (collector may pass a single
                pair so all runs write into one out store, keyed by run_id).
            heavy_every: TIER the HEAVY interpretability blocks (:data:`HEAVY_BLOCK_TAGS` —
                §6.5/§6.6/§6.7) to run only on every Nth checkpoint of this run. The cheap blocks
                run on EVERY checkpoint. REDESIGN 2026-06-16 (CONTRACT (B)): the heavy-due decision
                is now DETERMINISTIC IN THE CHECKPOINT'S OWN EMIT-ORDINAL (or step), NOT an
                in-memory counter, so two collectors processing out of order AGREE on which
                checkpoints are heavy. The checkpoint's emit-ordinal (``emit_ordinal`` /
                ``ckpt_index`` in the checkpoint dict, written by the trainer) modulo
                ``heavy_every`` == 0 => heavy; the init/Phase-2-start reference (ordinal 0 / step
                0) is therefore always a full heavy reference. If no ordinal is recorded we fall
                back to a deterministic rule on the step value (see :meth:`_emit_ordinal_for` /
                :meth:`_heavy_due_for_ckpt`). ``<= 1`` runs the heavy blocks every checkpoint.
                Default 3.
        """
        self.run_dir = os.path.abspath(run_dir)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # --- v2 C4: resolve the ACTIVE block set (which blocks this worker runs) ---
        if blocks:
            chosen = {b for b in blocks if b in BLOCK_ORDER}
        elif block_set == "gpu":
            chosen = set(GPU_BLOCK_TAGS)
        elif block_set == "cpu":
            chosen = set(CPU_BLOCK_TAGS)
        else:  # "all" (legacy)
            chosen = set(BLOCK_ORDER)
        self._active_blocks: Tuple[str, ...] = tuple(b for b in BLOCK_ORDER if b in chosen)
        self.block_set = block_set
        # Does this worker need the probe loaders / Hessian batch at all? Only the GPU/forward
        # blocks do; a CPU-only worker (optim/weights/trajectory/circuits) skips that build.
        self._needs_probe_resources = bool(set(self._active_blocks) & GPU_BLOCK_TAGS)

        # --- locate + read the run manifest (projection spec, identity, arch) ---
        self.manifest = self._load_run_manifest(self.run_dir)
        self.run_id = run_id or self.manifest.get("run_id")

        # --- criterion: multi-label BCE-with-logits (fruit, flower) ---
        self.criterion = nn.BCEWithLogitsLoss()

        # --- probe loaders + Hessian batch (built ONCE; briefing §4 invariants) ---
        chosen_csv = probe_csv or val_csv
        self.probe_train_loader = None
        self.probe_eval_loader = None
        self.hessian_batch = None
        probe_ids: Dict[str, Any] = {}
        hess_ids: Dict[str, Any] = {}
        if not self._needs_probe_resources:
            print(f"[Extractor] block_set={block_set!r}: CPU-only block set "
                  f"{list(self._active_blocks)} needs no model forward — skipping probe/Hessian "
                  f"build (no CSV/dataloaders, lower host RAM).", flush=True)
        elif chosen_csv and os.path.exists(chosen_csv):
            try:
                (self.probe_train_loader, self.probe_eval_loader,
                 probe_ids) = _build_probe_loaders(
                    chosen_csv, n_probe=n_probe, split_seed=split_seed,
                    batch_size=probe_batch_size, num_workers=num_workers)
                self.hessian_batch, hess_ids = _build_hessian_batch(
                    self.probe_eval_loader, hessian_batch_size, self.device,
                    hessian_seed=(int(split_seed) ^ 0x48455353))  # 'HESS' tag; fixed+seeded
            except Exception as e:
                print(f"[Extractor] probe/Hessian construction failed ({e!r}); "
                      f"forward/backward blocks will be skipped.", file=sys.stderr, flush=True)
                traceback.print_exc()
        else:
            print(f"[Extractor] no probe/val CSV available (probe_csv={probe_csv!r}, "
                  f"val_csv={val_csv!r}); forward/backward blocks will be skipped.",
                  file=sys.stderr, flush=True)

        # --- shared random projection (seed from the manifest so §6.2 coords match) ---
        pj = self.manifest.get("projection", {}) or {}
        self.projection_seed = int(
            projection_seed if projection_seed is not None
            else self.manifest.get("projection_seed", pj.get("seed", DEFAULT_PROJECTION_SEED)))
        self.projection_dim = int(
            projection_dim if projection_dim is not None
            else pj.get("d", DEFAULT_PROJECTION_DIM))
        self.projection_method = (
            projection_method if projection_method is not None
            else pj.get("method", DEFAULT_PROJECTION_METHOD))
        self._manifest_D_trainable = self.manifest.get("D_trainable") or pj.get("D_trainable")
        # The projection is built lazily on the first checkpoint, once D_trainable is known
        # from the model (or the manifest). It is then reused for every checkpoint of the run.
        self.projection: Optional[SparseRandomProjection] = None

        # --- out stores (per-run unless the collector shares a single pair) ---
        self._owns_stores = scalar_store is None and array_store is None
        if scalar_store is not None or array_store is not None:
            self.scalar = scalar_store
            self.array = array_store
        else:
            sub = self.run_id or os.path.basename(self.run_dir.rstrip("/"))
            store_root = os.path.join(out_store_dir, sub)
            os.makedirs(store_root, exist_ok=True)
            self.scalar = ScalarStore(os.path.join(store_root, "scalars.parquet"))
            self.array = ArrayStore(os.path.join(store_root, "arrays.zarr"))

        # --- FIXED INIT (E(0)) reference: loaded ONCE from disk; READ-ONLY (CONTRACT (C)) -----
        # This replaces the old "capture init from the first processed checkpoint" carry-over,
        # which depended on processing order. The reference is the run's on-disk
        # ``<run>/kept/init_model.pt`` (or ``init_model.pt`` / ``phase1_final.pt``); §6.2 "vs
        # init" and §6.6 CKA-vs-init read it via ``ctx.init_model_sd`` identically for EVERY
        # checkpoint. None if no init reference is on disk yet (those pieces then skip cleanly).
        self.init_ref_path = resolve_init_reference_path(self.run_dir)
        self.init_model_sd: Optional[Dict[str, Any]] = load_init_model_sd(self.init_ref_path)
        if self.init_model_sd is None:
            print(f"[Extractor] no INIT reference resolved under {self.run_dir!r} "
                  f"(looked for init_model.pt / phase1_final.pt in kept/ and the run dir); "
                  f"§6.2 'vs init' / §6.6 CKA-vs-init will skip.", file=sys.stderr, flush=True)

        # --- HEAVY-block tiering: cadence (DETERMINISTIC per-checkpoint; CONTRACT (B)) --------
        # NO in-memory counter. The heavy-due decision is a pure function of the checkpoint's own
        # recorded emit-ordinal (or, as a documented fallback, its step) so two collectors
        # processing checkpoints out of order AGREE which are heavy. See _heavy_due_for_ckpt.
        self.heavy_every = max(1, int(heavy_every))

        # --- per-run base generator (CONVENTIONS.md §5; no global RNG) ---
        seed = int(self.manifest.get("seed", 0) or 0)
        self._base_gen = torch.Generator()
        self._base_gen.manual_seed((seed * 2_654_435_761 + 0x9E3779B9) & 0x7FFFFFFFFFFFFFFF)
        self._seed = seed

        # --- record the comparability invariants into the manifest we thread into ctx ---
        # (These do not rewrite the run's manifest.json on disk; they are recorded so any
        # block that reads ctx.manifest sees the resolved probe / Hessian / projection IDs.)
        self.manifest = dict(self.manifest)
        self.manifest.setdefault("extractor", {})
        self.manifest["extractor"] = {
            "probe_ids": probe_ids,
            "hessian_ids": hess_ids,
            "projection_seed": self.projection_seed,
            "projection_dim": self.projection_dim,
            "projection_method": self.projection_method,
        }
        # The trainer's manifest does not record output_dir, but §6.3 (block_trajectory)
        # locates the run's INLINE §5 ScalarStore via manifest['output_dir'] ->
        # '{output_dir}/metrics/scalars.parquet'. run_dir is '<output_dir>/checkpoints', so
        # derive and surface output_dir here when it is not already present (so the trajectory
        # block's straightness / velocity-autocorr can use the trainer's inline history).
        if "output_dir" not in self.manifest:
            parent = os.path.dirname(self.run_dir.rstrip("/"))
            if os.path.basename(self.run_dir.rstrip("/")) == "checkpoints":
                self.manifest["output_dir"] = parent

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _load_run_manifest(run_dir: str) -> Dict[str, Any]:
        """Find and read ``manifest.json`` in ``run_dir`` or its parent (the trainer writes it
        at ``<output_dir>/manifest.json`` while checkpoints live in
        ``<output_dir>/checkpoints``). Returns an empty dict if none is found (the extractor
        still runs with defaults, but the projection coords may not match the trainer — log)."""
        for cand in (os.path.join(run_dir, "manifest.json"),
                     os.path.join(os.path.dirname(run_dir), "manifest.json")):
            if os.path.exists(cand):
                try:
                    return read_manifest(cand)
                except Exception as e:
                    print(f"[Extractor] manifest at {cand!r} unreadable ({e!r}); using defaults.",
                          file=sys.stderr, flush=True)
                    return {}
        print(f"[Extractor] no manifest.json near {run_dir!r}; using default projection seed "
              f"{DEFAULT_PROJECTION_SEED} (coords may not match the trainer).",
              file=sys.stderr, flush=True)
        return {}

    def _ensure_projection(self, model: nn.Module) -> None:
        """Build the shared projection on first use, sizing ``D_trainable`` from the model's
        trainable parameter count (frozen input stage excluded — ``trainable_param_numel``).
        Reused for every subsequent checkpoint of the run. If the manifest recorded a
        ``D_trainable`` that disagrees with the model's, we trust the model and log, since the
        projection must match the live coordinate vector the §6.2 block flattens."""
        if self.projection is not None:
            return
        d_train = trainable_param_numel(model, include_requires_grad_false=True)
        if self._manifest_D_trainable is not None and int(self._manifest_D_trainable) != int(d_train):
            print(f"[Extractor] manifest D_trainable={self._manifest_D_trainable} != model "
                  f"trainable numel={d_train}; using the model's (projection must match the "
                  f"live flattened-weight vector).", file=sys.stderr, flush=True)
        self.projection = SparseRandomProjection(
            D_trainable=int(d_train), d=int(self.projection_dim),
            seed=int(self.projection_seed), method=str(self.projection_method),
        )

    def _attach_generators(self, ctx: ExtractCtx) -> None:
        """Attach deterministic per-block generators to ``ctx`` (CONVENTIONS.md §5). Each is
        seeded from ``(run seed, step, block tag)`` so probe noise is reproducible per
        checkpoint and never touches global RNG. Blocks that look for ``ctx.gen_<tag>`` find
        one; blocks that self-resolve from ``(ctx.seed, ctx.step)`` are unaffected."""
        for block_name, attr in _GEN_ATTRS.items():
            tag_hash = abs(hash(block_name)) & 0xFFFF
            seed = (int(self._seed) * 1_000_003 + int(ctx.step) * 131 + tag_hash) & 0x7FFFFFFF
            g = torch.Generator()  # CPU generator; blocks re-derive a device-matched one if needed
            g.manual_seed(seed)
            setattr(ctx, attr, g)

    @staticmethod
    def _heavy_due(counter: int, heavy_every: int) -> bool:
        """Gating predicate for the HEAVY interpretability blocks given an ORDINAL ``counter``.
        Returns True iff ``counter % heavy_every == 0`` — so ordinals 0, heavy_every,
        2*heavy_every, ... get the heavy blocks (ordinal 0 = the init/first checkpoint = a full
        reference). ``heavy_every <= 1`` makes every checkpoint heavy. Pure function so the
        cadence is unit-testable without a ViT-L model. REDESIGN 2026-06-16: ``counter`` is now
        the checkpoint's OWN deterministic emit-ordinal (see :meth:`_emit_ordinal_for`), not an
        in-memory processed counter, so the decision is order-independent across collectors."""
        he = max(1, int(heavy_every))
        return (int(counter) % he) == 0

    @staticmethod
    def _emit_ordinal_for(ckpt_meta: Dict[str, Any]) -> Optional[int]:
        """Return the checkpoint's deterministic EMIT-ORDINAL (0-based index in this run's emitted
        sequence) from its own recorded metadata, or None if no ordinal field is present.

        The trainer records the emit ordinal under one of these keys when it writes a checkpoint
        (init/Phase-2-start = ordinal 0). Reading it from the checkpoint — rather than counting
        processed checkpoints in memory — is what makes the heavy-tier decision agree across two
        collectors that process out of order (CONTRACT (B))."""
        for key in ("emit_ordinal", "ckpt_index", "emit_index", "sample_index"):
            v = ckpt_meta.get(key)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    continue
        return None

    @classmethod
    def _heavy_due_for_ckpt(cls, ckpt_meta: Dict[str, Any], heavy_every: int) -> Tuple[bool, int, str]:
        """Deterministic heavy-due decision for ONE checkpoint from its own metadata (CONTRACT
        (B)). Returns ``(heavy_due, ordinal_used, basis)``.

        Resolution order (all order-independent — a pure function of the checkpoint's own dict):
          1. **emit-ordinal** (preferred): ``ordinal % heavy_every == 0``. The trainer's recorded
             0-based emit index; init/Phase-2-start = 0 so it is always heavy.
          2. **step==0** (init / Phase-2-start / phase1_final): always heavy (the full reference).
          3. **step fallback (DOCUMENTED)**: when no ordinal is recorded, derive a stable ordinal
             from the step value as ``step // max(1, step_quantum)`` and gate that — i.e. a
             coarse, deterministic stand-in. ``step_quantum`` defaults to the sampler floor
             (``sampler_min_step``, 5) so consecutive sampled steps map to distinct ordinals; the
             decision is still a pure function of the step and therefore identical across
             collectors. Documented because it is an approximation of the true emit ordinal when
             the trainer did not stamp one.
        """
        he = max(1, int(heavy_every))
        ordinal = cls._emit_ordinal_for(ckpt_meta)
        if ordinal is not None:
            return ((ordinal % he) == 0), int(ordinal), "emit_ordinal"
        try:
            step = int(ckpt_meta.get("step", ckpt_meta.get("global_step", 0)) or 0)
        except (TypeError, ValueError):
            step = 0
        if step <= 0:
            return True, 0, "step0_reference"
        # Documented step fallback: stable ordinal = step // step_quantum.
        step_quantum = int(ckpt_meta.get("sampler_min_step", 5) or 5)
        derived = step // max(1, step_quantum)
        return ((derived % he) == 0), int(derived), "step_fallback"

    def _ctx_resources(self) -> Dict[str, Any]:
        """Assemble the ``ctx_resources`` mapping ``build_ctx`` consumes. All entries are
        read-only shared resources (CONTRACT (B)); ``init_model_sd`` is the FIXED INIT reference
        loaded once from disk and threaded into every ctx unchanged."""
        return {
            "device": self.device,
            "criterion": self.criterion,
            "projection": self.projection,
            "scalar": self.scalar,
            "array": self.array,
            "manifest": self.manifest,
            "probe_eval_loader": self.probe_eval_loader,
            "probe_train_loader": self.probe_train_loader,
            "hessian_batch": self.hessian_batch,
            "init_model_sd": self.init_model_sd,
        }

    # ------------------------------------------------------------------ main entry

    def extract_checkpoint(self, ckpt_path: str) -> Dict[str, Any]:
        """STATELESS per-checkpoint extraction (CONTRACT (B)): build the ctx for ``ckpt_path``
        from the checkpoint + the FIXED shared resources + the FIXED INIT reference, run every §6
        block in order, flush. Holds NO cross-checkpoint in-memory state — two collectors can run
        this on different checkpoints concurrently and out of order.

        Sequence:
          1. ``build_ctx`` loads the model + remaps the optimizer state by ``param_names`` and
             threads in the FIXED INIT reference (E(0), from disk) and the shared resources.
          2. Build / reuse the shared projection (sized to the model's trainable params).
          3. Decide whether this is a HEAVY checkpoint DETERMINISTICALLY from the checkpoint's own
             emit-ordinal (or step), NOT an in-memory counter — so two collectors agree.
          4. Run each block's ``extract(ctx)`` wrapped in try/except (one failure does not stop
             the others); CHEAP blocks every checkpoint, HEAVY blocks only when due (skipped
             heavy blocks log a ``block_skipped`` scalar and write no arrays/scalars this ckpt);
             TIME each block (``block_seconds`` scalar). Record per-block ok/err in the summary.
          5. Flush the ScalarStore (the collector fsyncs/finalizes after this returns).
          6. Drop the per-checkpoint tensors. NO carried state is advanced (init is fixed from
             disk; prev/Q0 rotation + §6.3 trajectory are POST-HOC — CONTRACT (C)/(D)).

        Returns a summary dict ``{run_id, step, condition, variant, heavy, heavy_ordinal,
        heavy_basis, blocks:{name:summary|err}, ok, n_blocks_ok, n_blocks_err, t_seconds}`` for
        the collector's log.
        """
        t0 = time.time()

        # build_ctx loads the checkpoint ONCE and threads in the FIXED INIT reference + shared,
        # read-only resources. prev_model_sd / prev_opt / Q0 are intentionally NOT passed (they are
        # always None now — the diagnostics that used them are POST-HOC).
        ctx = build_ctx(
            ckpt_path,
            self._ctx_resources(),
            num_classes=NUM_CLASSES,
        )

        # HEAVY-block tiering decision for THIS checkpoint — DETERMINISTIC in the checkpoint's own
        # metadata (emit-ordinal, else step), so two collectors processing out of order agree
        # (CONTRACT (B)). ctx.cache carries the per-checkpoint emit_ordinal / sampler_min_step that
        # build_ctx surfaced from the checkpoint dict; ctx.step is the fallback.
        ckpt_meta = dict(getattr(ctx, "cache", None) or {})
        ckpt_meta.setdefault("step", ctx.step)
        heavy_due, heavy_ordinal, heavy_basis = self._heavy_due_for_ckpt(
            ckpt_meta, self.heavy_every)

        # Build / reuse the shared projection now that the model is loaded.
        try:
            self._ensure_projection(ctx.model)
            ctx.projection = self.projection
        except Exception as e:
            print(f"[Extractor] projection build failed ({e!r}); §6.2 will degrade.",
                  file=sys.stderr, flush=True)

        # Per-block generators (deterministic, no global RNG).
        self._attach_generators(ctx)

        summary: Dict[str, Any] = {
            "run_id": ctx.run_id,
            "step": ctx.step,
            "condition": ctx.condition,
            "variant": ctx.variant,
            "M_present": bool((ctx.opt or {}).get("M_present", False)),
            "heavy": bool(heavy_due),
            "heavy_ordinal": int(heavy_ordinal),
            "heavy_basis": heavy_basis,
            "blocks": {},
        }
        n_ok = 0
        n_err = 0
        ctx_step = ctx.step  # captured here; ctx is deleted before the profile line is printed
        block_times: Dict[str, float] = {}  # per-block extract() wall-time for the profile line

        for block_name in self._active_blocks:
            # TIER: skip the HEAVY interpretability blocks on non-heavy checkpoints. A skipped
            # heavy block writes a ``block_skipped`` marker scalar (value=1, layer=block_tag) and
            # NO other arrays/scalars this ckpt — downstream analysis treats heavy quantities as
            # present only on heavy checkpoints by design.
            if block_name in HEAVY_BLOCK_TAGS and not heavy_due:
                if ctx.scalar is not None:
                    try:
                        ctx.add("block_skipped", 1.0, layer=block_name)
                    except Exception:
                        pass
                summary["blocks"][block_name] = {"skipped": "heavy-tier (not due)"}
                print(f"[extract] step={ctx.step} SKIP heavy block {block_name} "
                      f"(ordinal={heavy_ordinal} basis={heavy_basis} "
                      f"heavy_every={self.heavy_every})", flush=True)
                continue

            mod = _load_block(block_name)
            if mod is None or not hasattr(mod, "extract"):
                summary["blocks"][block_name] = {"skipped": "module not found"}
                continue
            tb0 = time.time()
            try:
                block_summary = mod.extract(ctx)
                summary["blocks"][block_name] = block_summary if isinstance(block_summary, dict) else {"ok": True}
                n_ok += 1
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Extractor] block {block_name!r} FAILED at run_id={ctx.run_id} "
                      f"step={ctx.step}:\n{tb}", file=sys.stderr, flush=True)
                summary["blocks"][block_name] = {"error": repr(e)}
                n_err += 1
            finally:
                # PER-BLOCK TIMING: record the block's wall-time as a scalar so the next smoke /
                # pilot is a per-block profile, and collect it for the one-line stdout summary.
                dt = time.time() - tb0
                block_times[block_name] = dt
                if ctx.scalar is not None:
                    try:
                        ctx.add("block_seconds", float(dt), layer=block_name)
                    except Exception:
                        pass

        # Flush scalars to a durable part file. The collector fsyncs the store dir and then
        # finalizes (delete-or-keep) the checkpoint AFTER this returns (briefing §2.1 ordering).
        try:
            if self.scalar is not None:
                self.scalar.flush()
        except Exception as e:
            print(f"[Extractor] scalar flush failed at step={ctx.step}: {e!r}",
                  file=sys.stderr, flush=True)

        # REDESIGN 2026-06-16 (CONTRACT (B)/(C)): NO carried cross-checkpoint state is advanced.
        # There is no prev_model_sd / prev_opt / Q0 to snapshot — the §6.1 rotation-vs-prev/init
        # and the §6.3 trajectory diagnostics are computed POST-HOC from the stored top-k QL/QR
        # eigvecs + projected coords. The init reference is fixed-from-disk in self.init_model_sd
        # and never changes. This is exactly what makes extract_checkpoint order-independent.

        # --- per-checkpoint host/GPU memory cleanup (briefing §2.2: bounded footprint) -------
        # Explicitly drop the big per-checkpoint tensors so the next checkpoint's torch.load does
        # not transiently hold two full optimizer states. ctx.opt aliases the loaded checkpoint's
        # optimizer storage (build_ctx references, never copies it); nothing else references it
        # (no prev_opt/Q0 clone any more), so deleting it frees the loaded blob.
        try:
            del ctx.model
        except Exception:
            pass
        try:
            ctx.opt = None
        except Exception:
            pass
        try:
            del ctx
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # NO per-run counter to advance (CONTRACT (B)): the heavy-due decision was a pure function
        # of this checkpoint's own metadata, so nothing carries over to the next call.

        total_s = time.time() - t0
        summary["ok"] = n_err == 0
        summary["n_blocks_ok"] = n_ok
        summary["n_blocks_err"] = n_err
        summary["t_seconds"] = round(total_s, 3)
        summary["block_seconds"] = {k: round(v, 3) for k, v in block_times.items()}

        # PER-BLOCK profile line: one stdout summary per checkpoint so the pilot can read the
        # per-block cost breakdown directly from the collector log.
        prof = " ".join(f"{k}={block_times[k]:.2f}" for k in BLOCK_ORDER if k in block_times)
        print(f"[extract] step={ctx_step} heavy={'TRUE' if heavy_due else 'FALSE'} "
              f"total={total_s:.2f}s | {prof}", flush=True)
        return summary

    def close(self) -> None:
        """Flush any owned stores (idempotent). The collector calls this on shutdown."""
        if self._owns_stores and self.scalar is not None:
            try:
                self.scalar.close()
            except Exception:
                pass


def extract_checkpoint(
    ckpt_path: str,
    run_dir: str,
    out_store_dir: str,
    device="cuda",
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-shot convenience: build a throwaway :class:`Extractor` for ``run_dir`` and extract a
    single checkpoint. REDESIGN 2026-06-16: extraction is now STATELESS per checkpoint (CONTRACT
    (B)), so this one-shot is FULLY EQUIVALENT to the collector's per-checkpoint call — the only
    reason to reuse a persistent :class:`Extractor` is to amortize the one-time resource build
    (probe loaders, Hessian batch, projection, INIT reference) across many checkpoints of a run.
    The init reference is read from disk (``<run>/kept/init_model.pt`` etc.), not from a prior
    checkpoint, so 'vs init' diagnostics work even on a single-checkpoint call."""
    ex = Extractor(run_dir=run_dir, out_store_dir=out_store_dir, device=device, **kwargs)
    try:
        return ex.extract_checkpoint(ckpt_path)
    finally:
        ex.close()
