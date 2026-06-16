#!/usr/bin/env python3
"""
preadapt_common.py — shared foundation for the PhenoVision
"pretraining-as-preadaptation" experiment and its checkpoint-extractor pipeline.

This module is imported by the trainer (``preadapt_train.py``), the checkpoint
extractor (``extractor/``), and the init-d90 sweep (``d90/run_subspace.py``).
Its API is therefore meant to be clean and stable; downstream modules depend on
the storage schema and projection semantics defined here.

It implements, per the briefing (``xAI/notes/phenovision_full_briefing.md``,
Part II §2/§3/§7/§8) and the approved plan (components C3/C4):

  1. Atomic, idempotent checkpoint IO (write-temp -> rename -> ``.done`` sentinel,
     claim-via-``.processing``).
  2. The logarithmic retention ladder.
  3. Per-run JSON manifest read/write.
  4. A tidy-long Parquet scalar store (pyarrow).
  5. A Zarr array store (with a documented per-(group,step) ``.npy`` fallback when
     zarr is unavailable).
  6. A seeded, matrix-free **sparse / hashing** random projection
     (``matvec`` / ``rmatvec``) that NEVER materializes a dense ``[d x n_params]``
     matrix — reused by extractor §6.2 and the d90 sweep §9.1.
  7. §8 optimizer-variant detection (``detect_evolving_M``,
     ``optimizer_geometry_fields``).
  8. ``trainable_named_parameters`` — the single definition of "trainable" used
     everywhere displacement / projection / rank statistics are computed (EXCLUDES
     the frozen input-encoding stage: ``patch_embed.*`` and ``pos_embed``).

Conventions (see ``xAI/two_noise/CONVENTIONS.md`` §5): every stochastic op threads
an explicit ``torch.Generator``; we never rely on global RNG. The random projection
is regenerated deterministically from its integer seed.

Architecture facts used here: ViT-L/16, D=1024, 24 blocks, 16 heads, head_dim 64.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- optional heavy deps; degrade gracefully -------------------------------------
try:  # pyarrow is required for the Parquet scalar store
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except Exception:  # pragma: no cover - environment guard
    _HAVE_PYARROW = False

try:
    import zarr
    _HAVE_ZARR = True
except Exception:  # pragma: no cover - environment guard
    _HAVE_ZARR = False


# =============================================================================
# 0. Frozen input-encoding stage / trainable parameter selection
# =============================================================================

# The frozen input-encoding stage (briefing Part I §3,§4): the patch-embedding
# tokenizer and the positional embeddings. These are frozen AND excluded from the
# optimizer; they contribute zero to every displacement / projection / rank stat and
# MUST never be included in one (briefing "Do not"). The CLS token, blocks, norm, and
# head are trainable.
FROZEN_PREFIXES: Tuple[str, ...] = ("patch_embed.", "pos_embed")


def is_frozen_input_stage(name: str) -> bool:
    """True iff parameter ``name`` belongs to the frozen input-encoding stage
    (patch_embed.* or pos_embed). Used to exclude it from optimizer groups and from
    every per-layer displacement/projection/rank statistic."""
    return any(name == p.rstrip(".") or name.startswith(p) for p in FROZEN_PREFIXES)


def trainable_named_parameters(
    model: nn.Module, include_requires_grad_false: bool = False
) -> List[Tuple[str, torch.nn.Parameter]]:
    """Return ``(name, param)`` pairs for the *trainable* parameters only, EXCLUDING
    the frozen input-encoding stage (``patch_embed.*`` and ``pos_embed``).

    This is the single source of truth for "which parameters count" anywhere a
    displacement, random projection, or rank statistic is computed (briefing Part I §7,
    Part II §6.2). The frozen embedder contributes exactly zero and would dilute every
    per-layer signal, so it is dropped here rather than at each call site.

    By default a parameter is included iff it is not in the frozen input stage AND it
    has ``requires_grad=True`` (so the result matches the optimizer's param groups).
    Set ``include_requires_grad_false=True`` to ignore ``requires_grad`` and select on
    name alone (useful when reconstructing the trainable set from a bare state_dict-
    loaded model whose grad flags have not been set yet).
    """
    out: List[Tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        if is_frozen_input_stage(name):
            continue
        if include_requires_grad_false or p.requires_grad:
            out.append((name, p))
    return out


def trainable_param_numel(model: nn.Module, **kwargs: Any) -> int:
    """Total element count over :func:`trainable_named_parameters` — i.e. the
    ``D_trainable`` used to size the random projection."""
    return int(sum(p.numel() for _, p in trainable_named_parameters(model, **kwargs)))


# =============================================================================
# 1. Atomic / idempotent checkpoint IO (briefing Part II §2.1)
# =============================================================================

DONE_SUFFIX = ".done"
TMP_SUFFIX = ".tmp"
PROCESSING_SUFFIX = ".processing"


def build_train_ckpt(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **meta: Any,
) -> Dict[str, Any]:
    """Assemble the training-checkpoint ``state`` dict for :func:`save_checkpoint_atomic`.

    CRITICAL (briefing Part II §6.1 / escalation): the checkpoint MUST carry the
    optimizer state, not just the model weights — the optimizer's slow geometric state
    is the headline metric-evolution signal and cannot be recovered if dropped.

    Returns a dict with ``model_state_dict`` and (when ``optimizer`` is given)
    ``optimizer_state_dict``, plus any metadata keys passed through ``**meta``
    (typically ``step``, ``epoch``, ``phase``, ``condition``, ``variant``, ``seed``,
    ``run_id``). Metadata keys never clobber the two reserved state-dict keys.
    """
    state: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    for k, v in meta.items():
        if k in ("model_state_dict", "optimizer_state_dict"):
            raise ValueError(f"meta key {k!r} collides with a reserved checkpoint key")
        state[k] = v
    return state


def save_checkpoint_atomic(state: Dict[str, Any], path: str) -> str:
    """Atomically write ``state`` to ``path`` and drop a zero-byte ``path + '.done'``
    sentinel only after the file is fully flushed and renamed (briefing §2.1).

    Sequence: torch.save -> flush+fsync the temp file -> ``os.replace`` (atomic on the
    same filesystem) -> create the ``.done`` sentinel. The consumer only ever touches a
    checkpoint whose ``.done`` exists, so it can never read a half-written file.

    Returns ``path``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + TMP_SUFFIX
    # torch.save to the temp name, then force the bytes durable before the rename.
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic rename on the same filesystem
    done = path + DONE_SUFFIX
    with open(done, "wb") as f:
        f.flush()
        os.fsync(f.fileno())
    return path


def is_ready(path: str) -> bool:
    """True iff ``path`` is a checkpoint ready for the extractor: the file exists, its
    ``.done`` sentinel exists, and it is not already claimed (no ``.processing``)."""
    return (
        os.path.exists(path)
        and os.path.exists(path + DONE_SUFFIX)
        and not os.path.exists(path + PROCESSING_SUFFIX)
    )


def count_pending(watch_dir: str) -> int:
    """Count PENDING checkpoints in ``watch_dir`` — the trainer's backpressure signal
    (REDESIGN contract A).

    A checkpoint is PENDING iff it is fully written and not yet claimed by a collector:
    a ``"<base>.pt"`` file exists, its ``"<base>.pt.done"`` sentinel exists, and it is
    NOT currently being processed (no ``"<base>.pt.processing"``). This is exactly the
    set :func:`is_ready` would accept — i.e. the work queue depth.

    The watch dir is a PURE WORK QUEUE: kept ladder checkpoints live in ``<run>/kept/``,
    not here, so every ``*.pt`` present is a to-be-processed item. We count by scanning
    the ``*.pt.done`` sentinels (one per emitted checkpoint) and requiring the base
    ``*.pt`` to still exist and to be unclaimed.

    Robust to a missing / not-yet-created ``watch_dir`` (returns 0). Pure: no side
    effects, no claims, safe to call from the trainer between training steps.
    """
    if not watch_dir or not os.path.isdir(watch_dir):
        return 0
    n = 0
    for entry in os.listdir(watch_dir):
        if not entry.endswith(DONE_SUFFIX):
            continue
        base = entry[: -len(DONE_SUFFIX)]  # strip ".done" -> "<...>.pt"
        if not base.endswith(".pt"):
            continue
        base_path = os.path.join(watch_dir, base)
        if not os.path.exists(base_path):
            continue  # base .pt gone (consumed) — not pending
        if os.path.exists(base_path + PROCESSING_SUFFIX):
            continue  # claimed by a collector — not pending
        n += 1
    return n


def claim_for_processing(path: str) -> Optional[str]:
    """Atomically claim ``path`` for extraction by renaming it to ``path + '.processing'``.

    Returns the processing path on success, or ``None`` if the checkpoint is not ready,
    is already claimed, or the rename loses a race with another worker. ``os.rename`` is
    the atomic lock primitive (briefing §2.1): exactly one caller wins the rename.
    """
    if not is_ready(path):
        return None
    proc = path + PROCESSING_SUFFIX
    try:
        os.rename(path, proc)
    except OSError:
        return None  # lost the race or vanished
    return proc


def finalize_processed(processing_path: str, keep: bool, original_path: Optional[str] = None) -> None:
    """After a record is written, fsync'd, and verified durable, dispose of a claimed
    (``.processing``) checkpoint (briefing §2.1: write-record -> fsync -> verify -> THEN
    delete; never delete first).

    If ``keep`` is True (the checkpoint is on the retention ladder), rename it back to
    its original name and re-drop the ``.done`` sentinel so it stays a valid,
    never-deleted artifact. If ``keep`` is False, delete the ``.processing`` file and its
    ``.done`` sentinel.

    ``original_path`` defaults to ``processing_path`` with the ``.processing`` suffix
    stripped.
    """
    if original_path is None:
        if not processing_path.endswith(PROCESSING_SUFFIX):
            raise ValueError(f"{processing_path!r} is not a .processing path")
        original_path = processing_path[: -len(PROCESSING_SUFFIX)]
    if keep:
        os.replace(processing_path, original_path)
        done = original_path + DONE_SUFFIX
        if not os.path.exists(done):
            with open(done, "wb") as f:
                f.flush()
                os.fsync(f.fileno())
    else:
        if os.path.exists(processing_path):
            os.remove(processing_path)
        done = original_path + DONE_SUFFIX
        if os.path.exists(done):
            os.remove(done)


# =============================================================================
# 2. Retention ladder (briefing Part II §2)
# =============================================================================

def on_ladder(step: int, recent_steps: Optional[Iterable[int]] = None,
              latest_keep: int = 2) -> bool:
    """True iff ``step`` is on the never-delete retention ladder.

    The ladder is the union of:
      * **log-spaced steps** — powers of two ``1, 2, 4, 8, 16, ...`` (and step 0), which
        are O(log T) and trivially cheap to keep; plus
      * **the most recent ``latest_keep`` steps**, which the caller supplies via
        ``recent_steps`` (the extractor knows the live set of emitted steps; this
        function does not track them statefully). Pass the set/iterable of recently
        emitted steps and we keep the ``latest_keep`` largest of them.

    Everything off the ladder is extract-then-delete. For stateful tracking across a run
    use :class:`RetentionLadder`.
    """
    if step <= 0:
        return True
    if _is_power_of_two(step):
        return True
    if recent_steps is not None:
        rs = sorted(set(int(s) for s in recent_steps), reverse=True)
        if step in rs[:max(0, latest_keep)]:
            return True
    return False


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class RetentionLadder:
    """Stateful retention ladder that tracks the emitted steps of one run.

    Call :meth:`note` for every emitted step (the extractor does this as checkpoints
    arrive), then :meth:`should_keep` decides whether a given step is never-delete.
    Keeps log-spaced (power-of-two) steps forever plus the most recent ``latest_keep``
    emitted steps.
    """

    def __init__(self, latest_keep: int = 2):
        self.latest_keep = latest_keep
        self._emitted: List[int] = []

    def note(self, step: int) -> None:
        """Record that ``step`` was emitted (idempotent)."""
        if step not in self._emitted:
            self._emitted.append(step)

    def should_keep(self, step: int) -> bool:
        """True iff ``step`` is on the ladder given the steps seen so far."""
        self.note(step)
        return on_ladder(step, recent_steps=self._emitted, latest_keep=self.latest_keep)

    @property
    def emitted(self) -> List[int]:
        return sorted(self._emitted)


# =============================================================================
# 3. Manifest (briefing Part II §7)
# =============================================================================

def write_manifest(path: str, **fields: Any) -> str:
    """Write a per-run JSON manifest (briefing §7). Pass manifest fields as keyword
    arguments: ``condition``, ``run_id``, ``variant``, ``seed``, the optimizer config,
    ``M_present`` (the detected boolean), the random-projection seed/output_dim/method,
    probe-image IDs + probe-train/eval split, Hessian-batch IDs, architecture
    (variant/patch/resolution/D/n_heads/head_dim), retention-ladder policy, and the
    sampler params delta/k/K. Writes atomically (temp + replace). Returns ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + TMP_SUFFIX
    with open(tmp, "w") as f:
        json.dump(fields, f, indent=2, default=_json_default, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def read_manifest(path: str) -> Dict[str, Any]:
    """Read a JSON manifest written by :func:`write_manifest`."""
    with open(path, "r") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for numpy / torch scalars and tensors that show up in
    optimizer configs and manifests."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# =============================================================================
# 4. Tidy-long Parquet scalar store (briefing Part II §7)
# =============================================================================

# One row per (condition, run_id, step, wall_time, layer, head, quantity, value).
# layer/head are null for global quantities.
SCALAR_COLUMNS: Tuple[str, ...] = (
    "condition", "run_id", "step", "wall_time", "layer", "head", "quantity", "value",
)


class ScalarStore:
    """Append-only tidy-long Parquet store for scalars and small vectors (briefing §7).

    Schema: one row per ``(condition, run_id, step, wall_time, layer, head, quantity,
    value)``; ``layer`` and ``head`` are null for global quantities. Buffer rows with
    :meth:`add`, then :meth:`flush` to disk.

    **Append strategy (documented):** rather than rewrite a single growing Parquet file
    (which costs O(N) per flush), this store writes a *dataset directory* of immutable
    part files (``part-00000.parquet``, ``part-00001.parquet``, ...) — one per flush.
    pyarrow's ``ParquetDataset`` / :func:`read_scalars` reads the whole directory back as
    one table. This is crash-safe (each part is written via a temp file + atomic rename)
    and concurrency-safe for a single writer process. If ``path`` ends in ``.parquet``
    it is treated as the directory stem and the parts go in a sibling directory
    ``<stem>_parts/``; otherwise ``path`` itself is the dataset directory.
    """

    def __init__(self, path: str):
        if not _HAVE_PYARROW:
            raise RuntimeError(
                "pyarrow is required for ScalarStore. Install with "
                "`mamba install -n reticulate-gpu2 -c conda-forge pyarrow`."
            )
        self.dataset_dir = _scalar_dataset_dir(path)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self._buf: List[Dict[str, Any]] = []
        self._schema = pa.schema([
            ("condition", pa.string()),
            ("run_id", pa.string()),
            ("step", pa.int64()),
            ("wall_time", pa.float64()),
            ("layer", pa.string()),
            ("head", pa.int64()),
            ("quantity", pa.string()),
            ("value", pa.float64()),
        ])

    def add(self, condition: str, run_id: str, step: int, wall_time: float,
            quantity: str, value: float,
            layer: Optional[str] = None, head: Optional[int] = None) -> None:
        """Buffer one tidy-long row. ``layer``/``head`` default to null (global)."""
        self._buf.append({
            "condition": condition,
            "run_id": run_id,
            "step": int(step),
            "wall_time": float(wall_time),
            "layer": None if layer is None else str(layer),
            "head": None if head is None else int(head),
            "quantity": str(quantity),
            "value": float(value),
        })

    def add_vector(self, condition: str, run_id: str, step: int, wall_time: float,
                   quantity: str, values: Iterable[float],
                   layer: Optional[str] = None) -> None:
        """Convenience: buffer a small vector as one row per element, encoding the index
        in ``head`` (e.g. principal-angle summaries, top-k eigenvalue summaries). For
        larger arrays use :class:`ArrayStore` instead."""
        for i, v in enumerate(values):
            self.add(condition, run_id, step, wall_time, quantity, float(v),
                     layer=layer, head=i)

    def __len__(self) -> int:
        return len(self._buf)

    def flush(self) -> Optional[str]:
        """Write buffered rows as a new immutable part file (atomic temp + rename) and
        clear the buffer. Returns the part path, or ``None`` if the buffer was empty."""
        if not self._buf:
            return None
        table = pa.Table.from_pylist(self._buf, schema=self._schema)
        idx = _next_part_index(self.dataset_dir)
        part = os.path.join(self.dataset_dir, f"part-{idx:05d}.parquet")
        tmp = part + TMP_SUFFIX
        pq.write_table(table, tmp)
        os.replace(tmp, part)
        self._buf.clear()
        return part

    def close(self) -> Optional[str]:
        """Flush any remaining buffered rows."""
        return self.flush()

    def __enter__(self) -> "ScalarStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _scalar_dataset_dir(path: str) -> str:
    """Resolve the parts directory for a ScalarStore path."""
    if path.endswith(".parquet"):
        return path[: -len(".parquet")] + "_parts"
    return path


def _next_part_index(dataset_dir: str) -> int:
    existing = [f for f in os.listdir(dataset_dir)
                if f.startswith("part-") and f.endswith(".parquet")]
    return len(existing)


def read_scalars(path: str):
    """Read a :class:`ScalarStore` dataset directory back as a pandas ``DataFrame``
    (tidy long). Accepts either the dataset directory or the original ``.parquet`` stem
    passed to the store."""
    if not _HAVE_PYARROW:
        raise RuntimeError("pyarrow is required to read scalar stores.")
    import pandas as pd
    dataset_dir = _scalar_dataset_dir(path) if path.endswith(".parquet") else path
    if not os.path.isdir(dataset_dir):
        # Allow reading a single plain parquet file too.
        if os.path.isfile(path):
            return pq.read_table(path).to_pandas()
        raise FileNotFoundError(dataset_dir)
    parts = sorted(
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.startswith("part-") and f.endswith(".parquet")
    )
    if not parts:
        return pd.DataFrame(columns=list(SCALAR_COLUMNS))
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    return table.to_pandas()


# =============================================================================
# 5. Zarr array store (briefing Part II §7)
# =============================================================================

class ArrayStore:
    """Array store for the heavy per-checkpoint outputs (eigen/singular spectra, MSD(τ)
    curves, projection coordinates, CKA matrices, circuit spectra, sparse patch maps).

    Layout: one Zarr store per run; arrays are addressed by a logical ``group`` name
    keyed on ``(step, layer[, head])`` and **chunked along the step axis** so time-series
    reads are efficient (briefing §7). Internally each ``(group, layer, head)`` triple
    maps to one Zarr array of shape ``(n_steps, *array_shape)`` that grows by appending a
    step-slice; the per-slice ``step`` indices are recorded in the array's ``.attrs`` so
    a reader can map row -> step.

    **Fallback (documented):** if ``zarr`` is not importable, the store transparently
    falls back to writing one ``.npy`` file per ``(group, layer, head, step)`` under
    ``<path>/_npy_fallback/`` plus a JSON index, and the reader reconstructs the stacked
    array from those. The on-disk format then differs but the Python API is identical.
    """

    def __init__(self, path: str):
        self.path = path
        self.backend = "zarr" if _HAVE_ZARR else "npy"
        if self.backend == "zarr":
            self._root = zarr.open_group(path, mode="a")
        else:
            self._fallback_dir = os.path.join(path, "_npy_fallback")
            os.makedirs(self._fallback_dir, exist_ok=True)
            self._index_path = os.path.join(self._fallback_dir, "index.json")
            if os.path.exists(self._index_path):
                with open(self._index_path, "r") as f:
                    self._index = json.load(f)
            else:
                self._index = {}

    @staticmethod
    def _array_key(group: str, layer: Optional[str], head: Optional[int]) -> str:
        parts = [group]
        if layer is not None:
            parts.append(f"layer={layer}")
        if head is not None:
            parts.append(f"head={head}")
        return "/".join(parts)

    def put(self, group: str, step: int, array: Any,
            layer: Optional[str] = None, head: Optional[int] = None) -> None:
        """Append ``array`` as the step-slice for ``step`` under the logical
        ``(group, layer, head)`` array. ``array`` may be a numpy array or a torch tensor
        (it is moved to CPU/numpy). The step axis is chunked for time-series reads."""
        arr = _to_numpy(array)
        key = self._array_key(group, layer, head)
        if self.backend == "zarr":
            self._put_zarr(key, step, arr)
        else:
            self._put_npy(key, step, arr)

    def _put_zarr(self, key: str, step: int, arr: np.ndarray) -> None:
        slice_shape = arr.shape
        full_shape = (0,) + slice_shape
        # Chunk along the step axis (size 1 row of the full slice) for time-series reads.
        chunks = (1,) + slice_shape if slice_shape else (1,)
        if key in self._root:
            za = self._root[key]
        else:
            za = self._root.create_array(
                key, shape=full_shape, chunks=chunks, dtype=arr.dtype,
            )
            za.attrs["steps"] = []
        za.append(arr[np.newaxis, ...])
        steps = list(za.attrs["steps"])
        steps.append(int(step))
        za.attrs["steps"] = steps

    def _put_npy(self, key: str, step: int, arr: np.ndarray) -> None:
        safe = key.replace("/", "__")
        fname = f"{safe}__step={int(step)}.npy"
        fpath = os.path.join(self._fallback_dir, fname)
        tmp = fpath + TMP_SUFFIX
        np.save(tmp, arr)
        os.replace(tmp, fpath)
        self._index.setdefault(key, [])
        if int(step) not in self._index[key]:
            self._index[key].append(int(step))
        self._flush_index()

    def _flush_index(self) -> None:
        tmp = self._index_path + TMP_SUFFIX
        with open(tmp, "w") as f:
            json.dump(self._index, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._index_path)

    def read(self, group: str, layer: Optional[str] = None,
             head: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """Read back the stacked array for ``(group, layer, head)`` as
        ``(stacked_array, steps)`` where ``stacked_array[i]`` is the slice for
        ``steps[i]`` (steps in the order they were appended)."""
        key = self._array_key(group, layer, head)
        if self.backend == "zarr":
            if key not in self._root:
                raise KeyError(key)
            za = self._root[key]
            return np.asarray(za[:]), list(za.attrs.get("steps", []))
        if key not in self._index:
            raise KeyError(key)
        steps = list(self._index[key])
        safe = key.replace("/", "__")
        slices = [
            np.load(os.path.join(self._fallback_dir, f"{safe}__step={s}.npy"))
            for s in steps
        ]
        return (np.stack(slices, axis=0) if slices else np.empty((0,))), steps

    def keys(self) -> List[str]:
        """List the logical array keys present in the store."""
        if self.backend == "zarr":
            return sorted(self._root.array_keys())
        return sorted(self._index.keys())


def _to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


# =============================================================================
# 6. Seeded matrix-free sparse / hashing random projection (briefing §4, §6.2, §9.1)
# =============================================================================
#
# CRITICAL: never materialize a dense [d x D_trainable] matrix. The ViT-L trainable
# parameter count is ~3e8, so a dense projection is infeasible. We use a seeded sparse
# (Achlioptas / very-sparse Li et al.) or signed hashing (count-sketch) projection,
# regenerated deterministically from the seed and applied blockwise so memory stays
# O(block). Both directions of the operator are matrix-free.
#
# Conventions:
#   D_trainable : size of the full trainable parameter vector theta (the high-dim space)
#   d           : the low (projected) dimension
#   matvec(v: R^d) -> R^D_trainable        ("P v"   — used to EXPAND subspace coords)
#   rmatvec(w: R^D_trainable) -> R^d       ("P^T w" — used to REDUCE / project weights down)
#
# The operator R = P^T (rmatvec, the dimension-REDUCING map R^D -> R^d) is the canonical
# Johnson–Lindenstrauss embedding here: it is the direction calibrated to approximately
# preserve L2 distances (E[||R x||^2] = ||x||^2). This is the form used by §6.2 to give
# the conditions a shared low-dim coordinate system. matvec = P (R^d -> R^D) is its exact
# transpose; it is NOT separately norm-preserving (a single linear operator cannot be
# isometric in both directions when D >> d). For the d90 reparameterization
# theta = theta0 + P v (§9.1) the absolute scale of P is irrelevant — it is absorbed into
# the learned v — so what matters there is that P is a fixed, seed-reproducible map with
# roughly isotropic columns, which the count-sketch construction provides.
#
# For the d90 sweep (§9.1) theta = theta0 + P v with v in R^d -> P = matvec.
# For dimension-reduced weights (§6.2) we project trainable params down -> P^T = rmatvec.
# Both share this one class so the seed/dim/method recorded in the manifest fixes the
# coordinate system identically across conditions and across the two consumers.

_PROJ_METHODS = ("achlioptas", "hashing")


class SparseRandomProjection:
    """Seeded, matrix-free random projection between ``R^d`` and ``R^D_trainable``.

    The projection approximately preserves L2 distances (Johnson–Lindenstrauss). It is
    regenerated deterministically from ``seed`` every call, so it is identical across
    processes and never stored as a dense matrix. Work is done blockwise (default 1<<22
    elements) so peak memory is O(block), and all tensors stay on the device of the
    input vector (GPU-friendly).

    The norm-preserving direction is the dimension-REDUCING map ``rmatvec``
    (``R^D_trainable -> R^d``): ``E[||rmatvec(x)||^2] = ||x||^2``. ``matvec`` is its exact
    transpose (used to expand subspace coords for the d90 reparameterization, where the
    absolute scale is absorbed by the learned ``v``).

    Methods
    -------
    method="achlioptas" (very-sparse, Li et al. density 1/sqrt(D)):
        Each of the ``D_trainable`` input coordinates is hashed to exactly one of the
        ``d`` output coordinates with a signed weight ``+/- sqrt(1/density)`` chosen with
        probability ``density`` (else it drops out). This is the count-sketch-style
        very-sparse JL projection; it is column-sparse so ``rmatvec`` is a scatter-add and
        ``matvec`` is a gather. The ``sqrt(1/density)`` magnitude makes the reducing map
        norm-preserving in expectation.
    method="hashing" (signed feature hashing / count-sketch):
        Every input coordinate maps (via a seeded hash) to exactly one output coordinate
        with a random sign and unit magnitude. Densest, cheapest; norm-preserving in
        expectation in the reducing direction.

    The default ``achlioptas`` is the briefing's recommended sparse JL.

    The operator is defined so that ``matvec`` (``P v``, ``R^d -> R^D_trainable``) and
    ``rmatvec`` (``P^T w``, ``R^D_trainable -> R^d``) are exact transposes of one another.
    """

    def __init__(self, D_trainable: int, d: int, seed: int,
                 method: str = "achlioptas", density: Optional[float] = None,
                 block: int = 1 << 22):
        if method not in _PROJ_METHODS:
            raise ValueError(f"method must be one of {_PROJ_METHODS}, got {method!r}")
        if d <= 0 or D_trainable <= 0:
            raise ValueError("d and D_trainable must be positive")
        self.D_trainable = int(D_trainable)
        self.d = int(d)
        self.seed = int(seed)
        self.method = method
        self.block = int(block)
        if method == "achlioptas":
            # Very-sparse JL density ~ 1/sqrt(D); default per Li et al.
            self.density = float(density) if density is not None else 1.0 / math.sqrt(self.D_trainable)
            self.density = min(max(self.density, 1.0 / self.D_trainable), 1.0)
            # Signed nonzero magnitude so the REDUCING map preserves norm in expectation:
            # each kept input coord contributes scale^2*density*x_i^2 to its output bin, so
            # E[||rmatvec(x)||^2] = scale^2 * density * ||x||^2 = ||x||^2 when scale^2 = 1/density.
            self._scale = math.sqrt(1.0 / self.density)
        else:  # hashing / count-sketch: every coord kept, unit signed magnitude.
            self.density = 1.0
            self._scale = 1.0

    # --- deterministic per-coordinate (target, sign, mask) generation ---------------
    #
    # The map for input coordinate ``i`` MUST be a pure function of (seed, i) alone, so
    # the projection is invariant to the caller's ``block`` choice (which only controls
    # peak memory). To achieve that we generate the random stream in FIXED-SIZE internal
    # tiles (``_GEN_TILE``), each seeded from its absolute tile index. ``_block_maps``
    # assembles any requested ``[start, start+n)`` range out of those fixed tiles and
    # slices it — so the same coordinate gets the same draw regardless of how the caller
    # tiles its matvec/rmatvec loop. (Earlier this seeded directly on the variable
    # ``start``, which made the values depend on ``block`` — a latent bug, since ``block``
    # is not recorded in ``to_meta()`` and consumers may pick different values.)
    _GEN_TILE: int = 1 << 20

    def _tile_maps(self, tile_idx: int):
        """Generate (idx, w) for the fixed internal tile ``tile_idx`` (absolute input
        coordinates ``[tile_idx*_GEN_TILE, (tile_idx+1)*_GEN_TILE)``), seeded purely from
        (seed, tile_idx). CPU tensors; caller moves to device. The last tile is the full
        tile length even if it overruns ``D_trainable`` — the overrun is sliced off."""
        n = self._GEN_TILE
        g = torch.Generator()
        g.manual_seed((self.seed * 1_000_003 + tile_idx) & 0x7FFFFFFFFFFFFFFF)
        idx = torch.randint(0, self.d, (n,), generator=g, dtype=torch.long)
        sign = torch.randint(0, 2, (n,), generator=g, dtype=torch.int8).to(torch.float32) * 2.0 - 1.0
        if self.method == "achlioptas":
            keep = (torch.rand(n, generator=g) < self.density).to(torch.float32)
            w = sign * keep * self._scale
        else:
            w = sign * self._scale
        return idx, w

    def _block_maps(self, start: int, n: int, device: torch.device):
        """Return (idx[n] long, w[n] float) for input coordinates [start, start+n),
        assembled from the fixed internal tiles so the result is a pure function of
        (seed, absolute coordinate) and therefore identical for any caller ``block``.

        ``w`` is 0 for coordinates that drop out (achlioptas); ``idx`` is the output
        coordinate each input maps to. All returned tensors are on ``device``."""
        T = self._GEN_TILE
        end = start + n
        idx_parts: List[torch.Tensor] = []
        w_parts: List[torch.Tensor] = []
        tile = start // T
        while tile * T < end:
            tstart = tile * T
            tend = tstart + T
            tidx, tw = self._tile_maps(tile)
            lo = max(start, tstart) - tstart
            hi = min(end, tend) - tstart
            idx_parts.append(tidx[lo:hi])
            w_parts.append(tw[lo:hi])
            tile += 1
        idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts)
        w = w_parts[0] if len(w_parts) == 1 else torch.cat(w_parts)
        return idx.to(device), w.to(device)

    @torch.no_grad()
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute ``P v``: map a low-dim vector ``v in R^d`` to ``R^D_trainable``.

        ``out[i] = w[i] * v[idx[i]]`` for input coordinate ``i`` (gather), which is the
        transpose of :meth:`rmatvec`'s scatter."""
        v = v.reshape(-1)
        if v.numel() != self.d:
            raise ValueError(f"matvec expects v of length d={self.d}, got {v.numel()}")
        out = torch.empty(self.D_trainable, device=v.device, dtype=v.dtype)
        for start in range(0, self.D_trainable, self.block):
            n = min(self.block, self.D_trainable - start)
            idx, w = self._block_maps(start, n, v.device)
            out[start:start + n] = w.to(v.dtype) * v[idx]
        return out

    @torch.no_grad()
    def rmatvec(self, w_in: torch.Tensor) -> torch.Tensor:
        """Compute ``P^T w``: project a high-dim vector ``w in R^D_trainable`` down to
        ``R^d``.

        ``out[idx[i]] += w[i] * w_in[i]`` (scatter-add), the exact transpose of
        :meth:`matvec`."""
        w_in = w_in.reshape(-1)
        if w_in.numel() != self.D_trainable:
            raise ValueError(
                f"rmatvec expects w of length D_trainable={self.D_trainable}, "
                f"got {w_in.numel()}")
        out = torch.zeros(self.d, device=w_in.device, dtype=w_in.dtype)
        for start in range(0, self.D_trainable, self.block):
            n = min(self.block, self.D_trainable - start)
            idx, w = self._block_maps(start, n, w_in.device)
            contrib = (w.to(w_in.dtype) * w_in[start:start + n])
            out.index_add_(0, idx, contrib)
        return out

    def to_meta(self) -> Dict[str, Any]:
        """Manifest record fully specifying the projection (briefing §7): everything
        needed to regenerate it deterministically."""
        return {
            "method": self.method,
            "seed": self.seed,
            "d": self.d,
            "D_trainable": self.D_trainable,
            "density": self.density,
        }


def flatten_trainable(model: nn.Module, **kwargs: Any) -> torch.Tensor:
    """Flatten the trainable parameters (frozen input stage excluded) into one 1-D
    tensor, in :func:`trainable_named_parameters` order. The concatenation order is the
    coordinate convention the projection's ``D_trainable`` axis indexes."""
    chunks = [p.reshape(-1) for _, p in trainable_named_parameters(model, **kwargs)]
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def trainable_layout(model: nn.Module, **kwargs: Any) -> List[Tuple[str, int, int]]:
    """Return ``(name, offset, numel)`` for each trainable parameter, giving the
    blockwise layout of the flattened trainable vector. Lets callers slice per-layer
    coordinates out of a ``D_trainable``-vector without re-flattening the model."""
    layout: List[Tuple[str, int, int]] = []
    off = 0
    for name, p in trainable_named_parameters(model, **kwargs):
        n = p.numel()
        layout.append((name, off, n))
        off += n
    return layout


# =============================================================================
# 7. §8 optimizer-variant detection (briefing Part II §8)
# =============================================================================

# Candidate per-parameter buffer names for an evolving-M meta-loop (the Riccati/source
# target covariance). StableEvolutionSOAP and AdamW carry NONE of these, so
# detect_evolving_M -> False (the constant-isotropic canalization-floor case).
_M_CANDIDATES: Tuple[str, ...] = ("M", "M_target", "source", "precond_target", "G_target")

# Per-parameter state keys that the extractor branches on. Verified against the real
# StableEvolutionSOAP class (optim/stable_evolution_optimizer.py):
#   StableEvolutionSOAP: step, exp_avg, exp_avg_sq, precond, use_precond, L, R, QL, QR,
#                        alpha_last  (QL/QR/L/R/precond only on preconditioned 2-D params;
#                        alpha_last appears after the first preconditioned step)
#   AdamW (torch):       step, exp_avg, exp_avg_sq  (diagonal only)
_GEOMETRY_STATE_KEYS: Tuple[str, ...] = (
    "QL", "QR", "L", "R", "exp_avg", "exp_avg_sq", "precond", "alpha_last", "use_precond",
)


def detect_evolving_M(state_for_one_param: Dict[str, Any], group: Dict[str, Any]) -> bool:
    """Return True iff the optimizer carries an explicit, matrix-valued, slowly-updated
    *target* covariance ``M`` per parameter (the Riccati / evolving-M meta-loop;
    briefing §8).

    For ``StableEvolutionSOAP`` and ``AdamW`` this is **False** by construction — neither
    keeps any of the candidate target buffers; their "target" is a constant isotropic
    floor set by ``damping`` / ``kappa`` / ``alpha_min`` / ``alpha_max`` (the canalization
    floor). The extractor then skips the matrix-valued M block and records those scalars
    instead. A candidate is only accepted if it names a matrix-valued (2-D, square-ish)
    buffer, so a stray vector named ``"source"`` cannot trip it.
    """
    for name in _M_CANDIDATES:
        if name in state_for_one_param:
            buf = state_for_one_param[name]
            if isinstance(buf, torch.Tensor) and buf.dim() >= 2:
                return True
            # Non-tensor / 1-D candidate under a matching name -> not an evolving-M
            # matrix; keep scanning the remaining candidate names.
    return False


def optimizer_geometry_fields(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Summarize which per-parameter state keys the optimizer exposes so the extractor
    can branch between the StableEvo path (Kronecker eigenbasis QL/QR, generated
    preconditioner ``precond``, dynamic exponent ``alpha_last``) and the AdamW path
    (diagonal ``exp_avg_sq`` only).

    Returns a dict with:
      * ``state_keys``         : sorted union of all per-parameter state keys seen
      * ``has``                : {key: bool} for the geometry-relevant keys
      * ``M_present``          : the :func:`detect_evolving_M` boolean (any param)
      * ``n_preconditioned``   : count of params with a Kronecker eigenbasis (``QL`` or
                                 ``QR`` present and non-None)
      * ``n_params_with_state``: number of params that have any optimizer state
      * ``variant``            : "stable_evo" if a Kronecker/generative basis is present,
                                 "adamw" if only the diagonal moments are, else "unknown"
      * ``group_keys``         : sorted union of param_group config keys (so the manifest
                                 can record alpha_min/alpha_max/kappa/damping/etc.)
    """
    state = optimizer.state
    all_keys: set = set()
    has = {k: False for k in _GEOMETRY_STATE_KEYS}
    n_preconditioned = 0
    n_with_state = 0
    m_present = False

    groups_by_param: Dict[int, Dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            groups_by_param[id(p)] = group

    for p, st in state.items():
        if not isinstance(st, dict) or not st:
            continue
        n_with_state += 1
        for k in st:
            all_keys.add(k)
            if k in has:
                has[k] = True
        if (st.get("QL") is not None) or (st.get("QR") is not None):
            n_preconditioned += 1
        grp = groups_by_param.get(id(p), {})
        if detect_evolving_M(st, grp):
            m_present = True

    if has.get("QL") or has.get("QR") or has.get("precond") or has.get("alpha_last"):
        variant = "stable_evo"
    elif has.get("exp_avg_sq"):
        variant = "adamw"
    else:
        variant = "unknown"

    group_keys: set = set()
    for group in optimizer.param_groups:
        group_keys.update(k for k in group if k != "params")

    return {
        "state_keys": sorted(all_keys),
        "has": has,
        "M_present": m_present,
        "n_preconditioned": n_preconditioned,
        "n_params_with_state": n_with_state,
        "variant": variant,
        "group_keys": sorted(group_keys),
    }


# =============================================================================
# Module self-test (run directly: `python preadapt_common.py`)
# =============================================================================

def _selftest() -> None:  # pragma: no cover - exercised manually / in smoke test
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="preadapt_common_test_")
    try:
        # --- atomicity ---
        ckpt = os.path.join(tmpdir, "ckpt_step10.pt")
        save_checkpoint_atomic({"model_state_dict": {"w": torch.zeros(3)}, "step": 10}, ckpt)
        assert is_ready(ckpt), "is_ready should be True after save"
        proc = claim_for_processing(ckpt)
        assert proc is not None and proc.endswith(PROCESSING_SUFFIX)
        assert claim_for_processing(ckpt) is None, "second claim must fail"
        finalize_processed(proc, keep=True)
        assert os.path.exists(ckpt) and is_ready(ckpt), "ladder-kept ckpt restored"
        proc2 = claim_for_processing(ckpt)
        finalize_processed(proc2, keep=False)
        assert not os.path.exists(ckpt) and not os.path.exists(ckpt + DONE_SUFFIX)

        # --- count_pending (backpressure work-queue depth) ---
        wq = os.path.join(tmpdir, "watch_queue")
        assert count_pending(wq) == 0, "missing dir -> 0"
        os.makedirs(wq, exist_ok=True)
        assert count_pending(wq) == 0, "empty dir -> 0"
        # two ready+unclaimed -> pending 2
        for s in (5, 6):
            save_checkpoint_atomic({"model_state_dict": {}, "step": s},
                                   os.path.join(wq, f"step{s:08d}.pt"))
        assert count_pending(wq) == 2, count_pending(wq)
        # claim one -> pending 1 (claimed not counted)
        claimed = claim_for_processing(os.path.join(wq, "step00000005.pt"))
        assert claimed is not None
        assert count_pending(wq) == 1, count_pending(wq)
        # a .pt with no .done sentinel is NOT pending (half-written)
        with open(os.path.join(wq, "step00000007.pt"), "wb") as _f:
            _f.write(b"partial")
        assert count_pending(wq) == 1, count_pending(wq)
        # finalize the claimed one as deleted -> pending stays 1 (only step6 ready)
        finalize_processed(claimed, keep=False)
        assert count_pending(wq) == 1, count_pending(wq)

        # --- ladder ---
        assert on_ladder(1) and on_ladder(2) and on_ladder(8)
        assert not on_ladder(7)
        assert on_ladder(7, recent_steps=[3, 5, 7], latest_keep=2)
        lad = RetentionLadder(latest_keep=1)
        for s in (1, 3, 5, 6):
            lad.should_keep(s)
        assert lad.should_keep(6) and not on_ladder(5, recent_steps=lad.emitted, latest_keep=1)

        # --- manifest ---
        man = os.path.join(tmpdir, "manifest.json")
        write_manifest(man, condition="plantclef", run_id="plantclef__stable_evo__s42",
                        M_present=False, projection={"seed": 7, "d": 1000})
        assert read_manifest(man)["condition"] == "plantclef"

        # --- scalar store ---
        ss = ScalarStore(os.path.join(tmpdir, "scalars.parquet"))
        ss.add("imagenet", "imagenet__adamw__s42", 0, 0.0, "loss", 1.23)
        ss.add("imagenet", "imagenet__adamw__s42", 1, 0.5, "disp", 0.1, layer="blocks.0.attn.qkv.weight")
        ss.flush()
        ss.add("imagenet", "imagenet__adamw__s42", 2, 1.0, "loss", 0.99)
        ss.close()
        df = read_scalars(os.path.join(tmpdir, "scalars.parquet"))
        assert len(df) == 3 and set(df["quantity"]) == {"loss", "disp"}

        # --- array store ---
        arr = ArrayStore(os.path.join(tmpdir, "arrays.zarr"))
        arr.put("eigspec", step=1, array=np.arange(5, dtype="f4"), layer="blocks.0")
        arr.put("eigspec", step=2, array=np.arange(5, dtype="f4") + 10, layer="blocks.0")
        stacked, steps = arr.read("eigspec", layer="blocks.0")
        assert stacked.shape == (2, 5) and steps == [1, 2], (stacked.shape, steps)
        print(f"  ArrayStore backend = {arr.backend}")

        # --- random projection (transpose identity + JL distance preservation) ---
        torch.manual_seed(0)
        D, d = 50000, 800
        for method in ("achlioptas", "hashing"):
            proj = SparseRandomProjection(D, d, seed=123, method=method, block=4096)
            v = torch.randn(d)
            w = torch.randn(D)
            # <P v, w> == <v, P^T w>  (transpose identity; rtol loose for float accumulation)
            lhs = torch.dot(proj.matvec(v), w)
            rhs = torch.dot(v, proj.rmatvec(w))
            assert torch.allclose(lhs, rhs, rtol=1e-2, atol=1e-2), (method, lhs, rhs)
            # determinism
            assert torch.equal(proj.matvec(v), proj.matvec(v))
            # approximate L2-distance preservation of rmatvec (project down)
            a, b = torch.randn(D), torch.randn(D)
            full = (a - b).norm()
            proj_d = (proj.rmatvec(a) - proj.rmatvec(b)).norm()
            ratio = float(proj_d / full)
            assert 0.6 < ratio < 1.6, (method, ratio)
            assert proj.to_meta()["D_trainable"] == D
        print("  SparseRandomProjection transpose + JL checks passed")

        # --- trainable selection excludes frozen input stage ---
        import torch.nn as _nn

        class _Toy(_nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = _nn.Module()
                self.patch_embed.proj = _nn.Linear(4, 8)
                self.pos_embed = _nn.Parameter(torch.zeros(1, 5, 8))
                self.cls_token = _nn.Parameter(torch.zeros(1, 1, 8))
                self.blocks = _nn.ModuleList([_nn.Linear(8, 8)])
                self.head = _nn.Linear(8, 2)

        toy = _Toy()
        names = [n for n, _ in trainable_named_parameters(toy, include_requires_grad_false=True)]
        assert not any(n.startswith("patch_embed.") for n in names)
        assert "pos_embed" not in names
        assert "cls_token" in names and "head.weight" in names
        assert any(n.startswith("blocks.0") for n in names)
        layout = trainable_layout(toy, include_requires_grad_false=True)
        assert sum(n for _, _, n in layout) == flatten_trainable(toy, include_requires_grad_false=True).numel()

        # --- optimizer detection: AdamW (diagonal) ---
        adam = torch.optim.AdamW([p for _, p in trainable_named_parameters(toy, include_requires_grad_false=True)], lr=1e-3)
        out = torch.cat([p.reshape(-1) for _, p in trainable_named_parameters(toy, include_requires_grad_false=True)]).sum()
        # build a tiny loss to populate state
        x = torch.randn(2, 4)
        toy.train()
        loss = toy.head(toy.blocks[0](toy.patch_embed.proj(x))).sum()
        loss.backward()
        adam.step()
        fields = optimizer_geometry_fields(adam)
        assert fields["variant"] == "adamw" and fields["M_present"] is False, fields
        st0 = next(iter(adam.state.values()))
        assert detect_evolving_M(st0, adam.param_groups[0]) is False

        print("All preadapt_common self-tests passed.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
