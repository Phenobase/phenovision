#!/usr/bin/env python3
"""
extractor/collector.py — the PARALLEL producer/consumer collector loop (REDESIGN BUILD C).

The collector is the **consumer** in the producer/consumer architecture: the trainer (producer)
writes checkpoints into ``<run>/checkpoints/`` (the WATCH DIR) plus a ``.done`` sentinel; one or
more collector PROCESSES watch those dirs, claim ready checkpoints atomically, run the STATELESS
per-checkpoint §6 extraction (:func:`extractor.extract.extract_checkpoint`), make the extracted
record durable, and then dispose of the checkpoint.

REDESIGN contract (this module conforms to A/B/C):

  (A) The watch dir ``<run>/checkpoints/`` is a PURE WORK QUEUE — it holds ONLY to-be-processed
      checkpoints (``stepNNN.pt`` + ``stepNNN.pt.done``, transiently ``*.pt.processing``). After
      a checkpoint is extracted, a KEPT ladder checkpoint is MOVED to ``<run>/kept/`` (model-only,
      except the latest ``--keep-full-latest`` which move FULL for resume) and a non-kept one is
      DELETED. Either way nothing processed is left in the watch dir. Backpressure (the trainer
      pausing when ``count_pending`` is high) therefore sees a true queue depth.

  (B) STATELESS per-checkpoint extraction. Each claimed checkpoint is handed to
      :func:`extractor.extract.extract_checkpoint(ckpt_path, resources)`, which builds its own
      :class:`ExtractCtx` from the checkpoint + the FIXED shared ``resources`` (probe loaders,
      Hessian batch, projection, scalar/array stores, manifest). There is NO cross-checkpoint
      in-memory state (no ``prev_model_sd`` / ``prev_opt`` / ``Q0``-from-prev): the cross-checkpoint
      diagnostics (§6.1 rotation, §6.3 trajectory) are computed by a separate POST-HOC pass. As a
      result two collector processes can call ``extract_checkpoint`` on DIFFERENT checkpoints
      CONCURRENTLY and OUT OF ORDER; the atomic claim
      (:func:`preadapt_common.claim_for_processing`) guarantees exactly one collector owns each
      checkpoint.

  (C) PARALLEL-SAFE STORES. The §6 ScalarStore part-file writer derives a part index from the
      directory contents, which two writers would collide on. Each collector therefore writes to
      its OWN per-run dataset dir ``<out-store>/<run_id>/scalars_w<worker_id>.parquet`` (parts in
      ``scalars_w<worker_id>_parts/``). A reader (the idempotency probe here, and the POST-HOC
      pass) must GLOB ALL ``scalars*_parts`` part dirs of a run to read the full table — the
      collector's :meth:`_processed_steps` does exactly that. The ArrayStore (per-run Zarr / npy
      fallback) is opened in append mode with a per-(group,step) key, so two workers writing
      DISTINCT steps do not collide on the same array slice.

Backpressure is the TRAINER's job (it pauses emission when ``count_pending`` is high); the
collector always DRAINS — extract-then-dispose is exactly what frees the queue and the disk. The
disk-PAUSE sentinel likewise pauses the trainer, never the collector, which only logs it.

Idempotency / crash-safety: a relaunched or second worker SKIPS any step already present in the
out store (read across all worker part dirs) and, failing that, the atomic claim stops a double
process. ``--once`` drains all currently-ready checkpoints and exits (the SLURM wrapper loops it);
``run_forever`` polls.

---------------------------------------------------------------------------------------------
Two-collector layout (REDESIGN: 1 trainer + 2 collectors per run). Launch two processes over the
SAME ``--watch-dirs``/``--out-store`` with DISTINCT ``--worker-id`` (default from ``SLURM_PROCID``
or a random tag). They coexist with no shared-file write races beyond the atomic claim and the
per-worker part dirs. Size the pair so their combined extraction throughput keeps the queue depth
below the trainer's backpressure HIGH-water mark during the dense early burst.
---------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

# --- project paths (mirror extract.py / _ctx.py) -----------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.dirname(_HERE)                                  # .../xAI/py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PY_DIR))         # .../phenovision
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
_TWO_NOISE = os.path.join(_PROJECT_ROOT, "xAI", "two_noise")
for _p in (_PY_DIR, _PROJECT_ROOT, _PLANTCLEF, _TWO_NOISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from preadapt_common import (  # noqa: E402
    DONE_SUFFIX,
    PROCESSING_SUFFIX,
    TMP_SUFFIX,
    is_ready,
    claim_for_processing,
    save_checkpoint_atomic,
    on_ladder,
    read_scalars,
    read_manifest,
    ScalarStore,
    ArrayStore,
    SparseRandomProjection,
    trainable_param_numel,
    # v2 C4 two-pass refcount primitives (only used when --require-passes has >1 pass)
    is_ready_for_pass,
    claim_pass,
    mark_pass_complete,
    all_passes_complete,
    acquire_finalize,
    cleanup_pass_sentinels,
)
import torch.nn as nn  # noqa: E402

# The STATELESS per-checkpoint extractor (BUILD X). We import the MODULE (not just the symbol) so
# the collector keeps importing cleanly even while extract.py is being rewritten in parallel; the
# ``extract_checkpoint(ckpt_path, resources)`` function is resolved at CALL time. The shared
# constants + the probe / Hessian builders are reused from it when present, with conservative
# fallbacks so this module imports even if BUILD X renames those internals.
from extractor import extract as _extract  # noqa: E402

NUM_CLASSES: int = int(getattr(_extract, "NUM_CLASSES", 2))
DEFAULT_PROJECTION_SEED: int = int(getattr(_extract, "DEFAULT_PROJECTION_SEED", 1234))
DEFAULT_PROJECTION_DIM: int = int(getattr(_extract, "DEFAULT_PROJECTION_DIM", 4096))
DEFAULT_PROJECTION_METHOD: str = str(getattr(_extract, "DEFAULT_PROJECTION_METHOD", "achlioptas"))
#: Probe/Hessian builders (briefing §4 invariants). Resolved from extract.py if it still exposes
#: them; otherwise None and the collector runs without forward/backward resources (the optimizer /
#: weight / circuit blocks still run). The collector NEVER hard-fails on their absence.
_build_probe_loaders = getattr(_extract, "_build_probe_loaders", None)
_build_hessian_batch = getattr(_extract, "_build_hessian_batch", None)
#: The STATELESS per-run Extractor class (BUILD X). Resolved from the module so collector.py still
#: imports cleanly if extract.py is mid-rewrite; the actual object is built lazily in
#: :meth:`Collector._extractor_for`. Its ``extract_checkpoint(ckpt_path)`` is the X<->C contract.
_Extractor = getattr(_extract, "Extractor", None)

from models_vit import vit_large_patch16  # noqa: E402


# =============================================================================
# Disk-pause sentinel + model-only retention helpers
# =============================================================================

#: Disk-pressure PAUSE sentinel. The disk-safety watcher drops this file when group ``/blue`` free
#: space crosses the low-water mark; it pauses the TRAINER's emission, but the collector must keep
#: DRAINING — extract-then-dispose is precisely what frees disk, so stopping here is
#: counterproductive. The collector only LOGS the state at the top of each cycle.
DISK_PAUSE_SENTINEL = os.path.join(
    _PROJECT_ROOT, "xAI", "output", "preadapt", "_DISK_PAUSE"
)

#: Metadata keys preserved when a kept ladder checkpoint is moved MODEL-ONLY into ``<run>/kept/``.
#: Everything else (notably ``optimizer_state_dict`` and ``rng_state``) is dropped, shrinking a
#: kept ViT-L checkpoint from ~6.6 GB to ~1.2 GB. ``proj_coords`` / ``proj_meta`` (the inline §6.2
#: global coords the trainer stamped at emission) are KEPT — they are tiny and the POST-HOC §6.3
#: trajectory pass reads them per kept step.
SLIM_META_KEYS: Tuple[str, ...] = (
    "step", "epoch", "phase", "condition", "variant", "seed", "run_id",
    "global_step", "rng_version", "wall_time", "reason", "param_names",
    "proj_coords", "proj_meta",
)


def _is_full_checkpoint(state: Dict[str, Any]) -> bool:
    """True iff ``state`` still carries resume-critical optimizer/RNG state (i.e. it is a FULL
    checkpoint, not an already-slimmed model-only one)."""
    return bool(state.get("optimizer_state_dict")) or ("rng_state" in state)


def _slim_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Strip a loaded checkpoint dict down to ``{model_state_dict + minimal metadata}``, dropping
    ``optimizer_state_dict`` and ``rng_state``. Returns a new dict (does not mutate ``state``)."""
    slim: Dict[str, Any] = {"model_state_dict": state["model_state_dict"]}
    for k in SLIM_META_KEYS:
        if k in state:
            slim[k] = state[k]
    return slim


# =============================================================================
# checkpoint discovery
# =============================================================================

def _parse_step_from_name(path: str) -> Optional[int]:
    """Parse the step index from a checkpoint file name.

    The trainer writes ``step{step:08d}.pt`` for sampled checkpoints, ``init_model.pt`` for the
    Phase-2 init, and ``phase1_final.pt`` for the equalization endpoint. init/phase1 map to step 0
    (the ancestral references; the actual step is read from the checkpoint dict by ``build_ctx``,
    but the queue ordering only needs a monotone key, and 0 sorts them first which is correct)."""
    base = os.path.basename(path)
    if base.startswith("step") and base.endswith(".pt"):
        digits = base[len("step"):-len(".pt")]
        try:
            return int(digits)
        except ValueError:
            return None
    if base in ("init_model.pt", "phase1_final.pt"):
        return 0
    return None


def _discover_ready(watch_dir: str) -> List[Tuple[int, str, float]]:
    """Return ``(step, ckpt_path, mtime)`` for every READY checkpoint in ``watch_dir``.

    A checkpoint is ready iff it exists, its ``.done`` sentinel exists, and it is not already
    claimed (``.processing``) — :func:`is_ready`. mtime is the wall-time order key for
    oldest-first scheduling; the parsed step is the per-run order key."""
    out: List[Tuple[int, str, float]] = []
    for done in glob.glob(os.path.join(watch_dir, "*.pt" + DONE_SUFFIX)):
        ckpt = done[: -len(DONE_SUFFIX)]
        if not is_ready(ckpt):
            continue
        step = _parse_step_from_name(ckpt)
        if step is None:
            continue
        try:
            mtime = os.path.getmtime(ckpt)
        except OSError:
            continue
        out.append((step, ckpt, mtime))
    return out


def _run_id_for_dir(watch_dir: str, cache: Dict[str, str]) -> str:
    """Resolve the ``run_id`` for a watch directory from its run manifest (cached). Falls back to
    the parent directory name (the run_id under ``<run_id>/checkpoints``)."""
    if watch_dir in cache:
        return cache[watch_dir]
    rid = None
    for cand in (os.path.join(watch_dir, "manifest.json"),
                 os.path.join(os.path.dirname(watch_dir), "manifest.json")):
        if os.path.exists(cand):
            try:
                rid = read_manifest(cand).get("run_id")
            except Exception:
                rid = None
            if rid:
                break
    if not rid:
        parent = os.path.dirname(os.path.abspath(watch_dir.rstrip("/")))
        rid = os.path.basename(parent) or os.path.basename(watch_dir.rstrip("/"))
    cache[watch_dir] = rid
    return rid


def _kept_dir_for(watch_dir: str) -> str:
    """The ``<run>/kept/`` sibling of a ``<run>/checkpoints`` watch dir — where kept ladder
    checkpoints are MOVED after extraction (contract A). Derived from the watch dir's parent."""
    run_dir = os.path.dirname(os.path.abspath(watch_dir.rstrip("/")))
    return os.path.join(run_dir, "kept")


def _init_reference_path(watch_dir: str) -> Optional[str]:
    """Locate the fixed INIT reference checkpoint on disk for "vs init" blocks (§6.2/§6.6),
    which the STATELESS extractor loads from disk rather than from in-memory ``prev`` state.

    Looks, in order, for ``<run>/kept/init_model.pt``, ``<run>/init_model.pt``,
    ``<run>/checkpoints/init_model.pt``, then ``phase1_final.pt`` in the same places. Returns the
    first that exists, else None (the extractor degrades the vs-init blocks gracefully)."""
    run_dir = os.path.dirname(os.path.abspath(watch_dir.rstrip("/")))
    kept = os.path.join(run_dir, "kept")
    cks = os.path.abspath(watch_dir.rstrip("/"))
    for name in ("init_model.pt", "phase1_final.pt"):
        for base in (kept, run_dir, cks):
            cand = os.path.join(base, name)
            if os.path.exists(cand):
                return cand
    return None


# =============================================================================
# Collector
# =============================================================================

class Collector:
    """Parallel-safe collector over multiple producer checkpoint directories.

    Multiple Collector PROCESSES (one per ``--worker-id``) may share the same ``watch_dirs`` and
    ``out_store``. Each builds its OWN fixed shared resources per run (probe loaders, Hessian
    batch, projection — identical across workers because they are seeded from the same manifest /
    split seed) and its OWN per-worker ScalarStore dataset dir so part-file writes never collide.
    The atomic ``.processing`` claim makes each checkpoint owned by exactly one worker; extraction
    is STATELESS so order and concurrency are irrelevant.
    """

    def __init__(
        self,
        watch_dirs: List[str],
        out_store: str,
        device: str = "cuda",
        poll_interval: float = 10.0,
        retention_latest_keep: int = 2,
        keep_full_latest: int = 2,
        probe_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        n_probe: int = 1024,
        hessian_batch_size: int = 64,
        probe_batch_size: int = 64,
        num_workers: int = 4,
        heavy_every: int = 3,
        worker_id: Optional[str] = None,
        split_seed: int = 20240601,
        block_set: str = "all",
        blocks: Optional[List[str]] = None,
        require_passes: Optional[List[str]] = None,
    ):
        self.watch_dirs = [os.path.abspath(d) for d in watch_dirs]
        self.out_store = os.path.abspath(out_store)
        os.makedirs(self.out_store, exist_ok=True)
        self.device = device
        self.poll_interval = float(poll_interval)
        self.retention_latest_keep = int(retention_latest_keep)
        self.keep_full_latest = max(0, int(keep_full_latest))
        self.probe_csv = probe_csv
        self.val_csv = val_csv
        self.n_probe = int(n_probe)
        self.hessian_batch_size = int(hessian_batch_size)
        self.probe_batch_size = int(probe_batch_size)
        self.num_workers = int(num_workers)
        self.heavy_every = max(1, int(heavy_every))
        self.split_seed = int(split_seed)
        self.worker_id = _resolve_worker_id(worker_id)
        # --- v2 C4 CPU/GPU split + two-pass refcount ---
        self.block_set = block_set
        self.blocks = blocks
        # require_passes: the set of passes that must BOTH complete before a checkpoint is disposed.
        # Default ["all"] == LEGACY single-pass (claim_for_processing/finalize, byte-unchanged).
        self.require_passes: List[str] = list(require_passes) if require_passes else ["all"]
        self.legacy_single_pass = (self.require_passes == ["all"])
        # This worker's pass tag = its block_set ("gpu"/"cpu"), or "all" in legacy mode.
        self.pass_tag = "all" if self.legacy_single_pass else block_set

        # Per-run shared resources (built ONCE per run_id; STATELESS extraction reuses them).
        self._resources: Dict[str, Dict[str, Any]] = {}
        # Per-run STATELESS Extractor (BUILD X). Built ONCE per run_id; owns the shared, run-invariant
        # resources (probe loaders, Hessian batch, projection, INIT reference). Its
        # ``extract_checkpoint(ckpt_path)`` holds NO cross-checkpoint state, so this single object is
        # reused for every checkpoint of the run and is safe under the 2-worker split.
        self._extractors: Dict[str, Any] = {}
        # Per-run ScalarStore / ArrayStore handles owned by THIS worker (for flush/close).
        self._scalars: Dict[str, ScalarStore] = {}
        self._arrays: Dict[str, ArrayStore] = {}
        # Per-run processed-step cache (seeded from the out store across ALL worker part dirs).
        self._processed: Dict[str, Set[int]] = {}
        # Map watch_dir -> run_id.
        self._dir_run_id: Dict[str, str] = {}

    # ------------------------------------------------------------------ run id / store paths

    def _scalar_path(self, run_id: str) -> str:
        """This worker's per-run ScalarStore stem. Parts go in the sibling
        ``scalars_w<worker_id>_parts/`` dir so two workers never collide on a part index."""
        return os.path.join(self.out_store, run_id, f"scalars_w{self.worker_id}.parquet")

    @staticmethod
    def _all_scalar_part_dirs(run_root: str) -> List[str]:
        """Every worker's ScalarStore parts dir under ``<out-store>/<run_id>/`` — i.e. the
        per-worker ``scalars_w*_parts`` directories PLUS the legacy single-writer ``scalars_parts``
        if present. The idempotency probe and the POST-HOC pass GLOB all of these to read the full
        per-run table (contract C: read_scalars globs all parts)."""
        if not os.path.isdir(run_root):
            return []
        dirs: List[str] = []
        for entry in sorted(os.listdir(run_root)):
            full = os.path.join(run_root, entry)
            if not os.path.isdir(full):
                continue
            if entry == "scalars_parts" or (
                entry.startswith("scalars_w") and entry.endswith("_parts")
            ):
                dirs.append(full)
        return dirs

    # ------------------------------------------------------------------ per-run resources

    def _resources_for(self, run_id: str, watch_dir: str) -> Dict[str, Any]:
        """Build (once, cached) the FIXED shared ``resources`` mapping handed to the stateless
        :func:`extract.extract_checkpoint`. Mirrors the ``ctx_resources`` surface ``build_ctx``
        consumes (``device``/``criterion``/``projection``/``scalar``/``array``/``manifest`` +
        optional probe loaders / hessian batch), and adds the on-disk INIT reference path the
        vs-init blocks load (contract C). Identical across workers by construction (same manifest,
        same seeds), except each worker's ``scalar`` store writes to its own per-worker part dir."""
        res = self._resources.get(run_id)
        if res is not None:
            return res

        manifest = self._load_run_manifest(watch_dir)

        # --- per-worker ScalarStore + shared-key ArrayStore (contract C) -----------------------
        run_root = os.path.join(self.out_store, run_id)
        os.makedirs(run_root, exist_ok=True)
        scalar = ScalarStore(self._scalar_path(run_id))
        array = ArrayStore(os.path.join(run_root, "arrays.zarr"))
        self._scalars[run_id] = scalar
        self._arrays[run_id] = array

        # --- probe loaders + fixed Hessian batch (briefing §4 invariants; built ONCE) ----------
        chosen_csv = self.probe_csv or self.val_csv
        probe_train_loader = probe_eval_loader = hessian_batch = None
        device = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
        if chosen_csv and os.path.exists(chosen_csv) and _build_probe_loaders is not None \
                and _build_hessian_batch is not None:
            try:
                (probe_train_loader, probe_eval_loader, _ids) = _build_probe_loaders(
                    chosen_csv, n_probe=self.n_probe, split_seed=self.split_seed,
                    batch_size=self.probe_batch_size, num_workers=self.num_workers)
                hessian_batch, _hids = _build_hessian_batch(
                    probe_eval_loader, self.hessian_batch_size, device,
                    hessian_seed=(int(self.split_seed) ^ 0x48455353))
            except Exception as e:
                print(f"[collector w{self.worker_id}] probe/Hessian build failed ({e!r}); "
                      f"forward/backward blocks will be skipped for run_id={run_id}.",
                      file=sys.stderr, flush=True)
                traceback.print_exc()
        else:
            print(f"[collector w{self.worker_id}] no probe/val CSV (probe_csv={self.probe_csv!r}, "
                  f"val_csv={self.val_csv!r}); forward/backward blocks skipped for run_id={run_id}.",
                  file=sys.stderr, flush=True)

        # --- shared seeded projection (seed from the manifest so §6.2 coords match the trainer) --
        projection = self._build_projection(manifest)

        res = {
            "device": device,
            "criterion": nn.BCEWithLogitsLoss(),
            "projection": projection,
            "scalar": scalar,
            "array": array,
            "manifest": manifest,
            "probe_train_loader": probe_train_loader,
            "probe_eval_loader": probe_eval_loader,
            "hessian_batch": hessian_batch,
            "num_classes": NUM_CLASSES,
            "heavy_every": self.heavy_every,
            # On-disk INIT reference for the "vs init" blocks (contract C: NOT in-memory prev).
            "init_ref_path": _init_reference_path(watch_dir),
            "run_id": run_id,
            "worker_id": self.worker_id,
        }
        self._resources[run_id] = res
        return res

    def _extractor_for(self, run_id: str, watch_dir: str):
        """Build (once, cached) the STATELESS BUILD-X :class:`extract.Extractor` for this run.

        X's canonical stateless API is ``Extractor(run_dir, out_store_dir, ..., scalar_store=,
        array_store=).extract_checkpoint(ckpt_path)``: the Extractor builds the run-invariant shared
        resources (probe loaders, Hessian batch, manifest-seeded projection, on-disk INIT reference)
        ONCE in ``__init__`` and ``extract_checkpoint`` carries NO cross-checkpoint state (CONTRACT
        (B)). We pass THIS worker's per-run ScalarStore + the shared ArrayStore (created in
        :meth:`_resources_for`) so the part-index write race is avoided (per-worker scalar parts) and
        rows are keyed by run_id. The watch dir is ``<run>/checkpoints``; the Extractor finds the
        manifest in it or its parent and loads its own INIT reference from ``<run>/kept/``."""
        ex = self._extractors.get(run_id)
        if ex is not None:
            return ex
        # Ensures the per-worker scalar store + shared array store handles exist and are cached in
        # self._scalars / self._arrays (used by _fsync_store + shutdown flush).
        self._resources_for(run_id, watch_dir)
        scalar = self._scalars[run_id]
        array = self._arrays[run_id]
        ex = _Extractor(
            run_dir=watch_dir,
            out_store_dir=self.out_store,
            device=self.device,
            probe_csv=self.probe_csv,
            val_csv=self.val_csv,
            n_probe=self.n_probe,
            hessian_batch_size=self.hessian_batch_size,
            probe_batch_size=self.probe_batch_size,
            num_workers=self.num_workers,
            split_seed=self.split_seed,
            run_id=run_id,
            scalar_store=scalar,
            array_store=array,
            heavy_every=self.heavy_every,
            block_set=self.block_set,
            blocks=self.blocks,
        )
        self._extractors[run_id] = ex
        return ex

    def _build_projection(self, manifest: Dict[str, Any]) -> Optional[SparseRandomProjection]:
        """Build the seeded reducing projection (§6.2). ``D_trainable`` is taken from the manifest
        if recorded, else sized from a freshly-built ViT-L/16's trainable param count (the same
        count the trainer uses). Seed/dim/method come from the manifest so the coordinate system is
        byte-identical to the trainer's and across workers. Returns None on failure (the extractor
        degrades §6.2)."""
        pj = manifest.get("projection", {}) or {}
        seed = int(manifest.get("projection_seed", pj.get("seed", DEFAULT_PROJECTION_SEED)))
        dim = int(pj.get("d", DEFAULT_PROJECTION_DIM))
        method = pj.get("method", DEFAULT_PROJECTION_METHOD)
        d_train = manifest.get("D_trainable") or pj.get("D_trainable")
        try:
            if d_train is None:
                model = vit_large_patch16(num_classes=NUM_CLASSES)
                d_train = trainable_param_numel(model, include_requires_grad_false=True)
                del model
            return SparseRandomProjection(
                D_trainable=int(d_train), d=int(dim), seed=int(seed), method=str(method))
        except Exception as e:
            print(f"[collector w{self.worker_id}] projection build failed ({e!r}); §6.2 degrades.",
                  file=sys.stderr, flush=True)
            return None

    @staticmethod
    def _load_run_manifest(watch_dir: str) -> Dict[str, Any]:
        """Read ``manifest.json`` from the watch dir or its parent (the trainer writes it at
        ``<output_dir>/manifest.json`` while checkpoints live in ``<output_dir>/checkpoints``).
        Surfaces ``output_dir`` (derived from the watch dir) when absent so the POST-HOC §6.3 pass
        can find the trainer's inline metrics store. Returns an empty dict if none is found."""
        manifest: Dict[str, Any] = {}
        for cand in (os.path.join(watch_dir, "manifest.json"),
                     os.path.join(os.path.dirname(watch_dir), "manifest.json")):
            if os.path.exists(cand):
                try:
                    manifest = dict(read_manifest(cand))
                except Exception as e:
                    print(f"[collector] manifest at {cand!r} unreadable ({e!r}); using defaults.",
                          file=sys.stderr, flush=True)
                    manifest = {}
                break
        if "output_dir" not in manifest:
            wd = os.path.abspath(watch_dir.rstrip("/"))
            if os.path.basename(wd) == "checkpoints":
                manifest["output_dir"] = os.path.dirname(wd)
        return manifest

    def _should_keep(self, step: int, watch_dir: str) -> bool:
        """Decide whether ``step`` is on the never-delete retention ladder, using the GLOBAL set of
        emitted steps discovered ON DISK (NOT a per-worker in-memory history).

        Under the parallel 2-collector split a single worker only ever processes a SUBSET of a
        run's checkpoints, so a per-worker :class:`RetentionLadder` would compute the wrong
        "latest-N recent" set (a worker that happened to process only the early steps would wrongly
        keep them as "recent"). The ladder's "recent_steps" must therefore be the run's full
        emitted set, which we reconstruct from disk every decision: the union of steps still in the
        watch dir (queued / claimed) + steps already moved to ``<run>/kept/`` + this step. Powers
        of two and step<=0 are absolute and need no history. This makes the keep/delete verdict
        identical across workers and independent of processing order (contract C)."""
        emitted = self._emitted_steps_on_disk(watch_dir)
        emitted.add(int(step))
        return on_ladder(int(step), recent_steps=emitted,
                         latest_keep=self.retention_latest_keep)

    @staticmethod
    def _emitted_steps_on_disk(watch_dir: str) -> Set[int]:
        """The run's emitted steps reconstructed from disk: every ``stepNNN.pt`` the trainer wrote
        that is still in the watch dir (as ``.pt``, ``.pt.done``, or ``.pt.processing``) PLUS every
        ``stepNNN.pt`` already moved to ``<run>/kept/``. The union is the global emitted set the
        retention ladder's "recent" window is computed over."""
        emitted: Set[int] = set()
        kept_dir = _kept_dir_for(watch_dir)
        for d in (os.path.abspath(watch_dir.rstrip("/")), kept_dir):
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                base = entry
                for suf in (DONE_SUFFIX, PROCESSING_SUFFIX, TMP_SUFFIX):
                    if base.endswith(suf):
                        base = base[: -len(suf)]
                        break
                if not base.endswith(".pt"):
                    continue
                s = _parse_step_from_name(os.path.join(d, base))
                if s is not None and s > 0:
                    emitted.add(s)
        return emitted

    def _processed_steps(self, run_id: str) -> Set[int]:
        """Per-run processed-step set, seeded once from the out store by GLOBBING ALL worker part
        dirs (contract C). A step present in ANY worker's ScalarStore is treated as processed, so a
        relaunched or second worker never redoes it (idempotency)."""
        if run_id in self._processed:
            return self._processed[run_id]
        seen: Set[int] = set()
        run_root = os.path.join(self.out_store, run_id)
        for parts_dir in self._all_scalar_part_dirs(run_root):
            try:
                df = read_scalars(parts_dir)
                if df is not None and len(df) and "step" in df:
                    seen.update(int(s) for s in df["step"].dropna().unique())
            except Exception:
                continue
        self._processed[run_id] = seen
        return seen

    # ------------------------------------------------------------------ scheduling

    def _build_queue(self) -> List[Dict[str, Any]]:
        """Scan all watch dirs for ready, not-yet-processed checkpoints. Each entry:
        ``{run_id, step, path, mtime, watch_dir}``. Sorted oldest-first (mtime, step, run_id).

        Note: two workers build INDEPENDENT queues; the atomic claim resolves who processes each
        entry, so no cross-worker queue coordination is needed. We do not assume a single worker or
        a particular order."""
        entries: List[Dict[str, Any]] = []
        for wd in self.watch_dirs:
            if not os.path.isdir(wd):
                continue
            run_id = _run_id_for_dir(wd, self._dir_run_id)
            # LEGACY single-pass: idempotency is per-STEP (any worker's scalars => done).
            # TWO-PASS: idempotency is per-PASS (the .<pass>.complete sentinel via
            # is_ready_for_pass) — a step the OTHER pass already wrote scalars for must NOT be
            # skipped for MY pass, so we do NOT consult _processed_steps in two-pass mode.
            processed = self._processed_steps(run_id) if self.legacy_single_pass else None
            for step, path, mtime in _discover_ready(wd):
                if self.legacy_single_pass:
                    if step in processed:
                        continue
                else:
                    if not is_ready_for_pass(path, self.pass_tag):
                        continue
                entries.append({"run_id": run_id, "step": step, "path": path,
                                "mtime": mtime, "watch_dir": wd})
        entries.sort(key=lambda e: (e["mtime"], e["step"], e["run_id"]))
        return entries

    # ------------------------------------------------------------------ processing one ckpt

    def _process_one(self, entry: Dict[str, Any]) -> bool:
        """Claim, extract (STATELESS), durably persist, then dispose of one checkpoint.

        Returns True iff processed (record durable + disposed). False if skipped (lost the claim
        race, already processed, or extraction failed before any record)."""
        run_id = entry["run_id"]
        step = entry["step"]
        path = entry["path"]
        watch_dir = entry["watch_dir"]

        # v2 C4: in two-pass mode this worker runs only ITS pass's blocks; disposal is deferred to
        # the finalize sweep (run_once) once BOTH passes' .complete sentinels exist.
        if not self.legacy_single_pass:
            return self._process_one_pass(entry)

        # ===== LEGACY single-pass path (byte-unchanged; require_passes == ["all"]) =====
        # Idempotency double-check just before claiming (cheap; the authoritative guard is the
        # atomic claim below).
        if step in self._processed_steps(run_id):
            return False

        # Atomic claim (the lock). Lose the race -> another worker owns it; skip.
        proc = claim_for_processing(path)
        if proc is None:
            return False

        extractor = self._extractor_for(run_id, watch_dir)
        keep = self._should_keep(step, watch_dir)

        try:
            # X's stateless API (CONTRACT (B)): the Extractor holds the run-invariant shared
            # resources + the fixed INIT reference; extract_checkpoint carries NO cross-checkpoint
            # state, so calling it per claimed checkpoint (out of order, across 2 workers) is safe.
            summary = extractor.extract_checkpoint(proc)
        except Exception:
            # Extraction blew up before a usable record. Do NOT discard the checkpoint
            # (briefing §2.1: never delete/move before the record is durable) — restore it to the
            # watch dir so a later pass / a human can retry, and log loudly.
            tb = traceback.format_exc()
            print(f"[collector w{self.worker_id}] extract FAILED run_id={run_id} step={step}:\n{tb}",
                  file=sys.stderr, flush=True)
            self._restore_to_queue(proc, path)
            return False

        # --- ensure the record is DURABLE before disposing of the checkpoint (§2.1) ---
        self._fsync_store(run_id)

        # Mark processed (after the record is durable), THEN dispose.
        self._processed_steps(run_id).add(step)
        if keep:
            disp = self._move_to_kept(run_id, step, proc, watch_dir)
        else:
            disp = self._delete_from_queue(proc)

        nb_ok = summary.get("n_blocks_ok") if isinstance(summary, dict) else None
        nb_err = summary.get("n_blocks_err") if isinstance(summary, dict) else None
        t = summary.get("t_seconds") if isinstance(summary, dict) else None
        heavy = summary.get("heavy") if isinstance(summary, dict) else None
        print(f"[collector w{self.worker_id}] processed run_id={run_id} step={step} "
              f"heavy={heavy} blocks_ok={nb_ok} blocks_err={nb_err} t={t}s -> {disp}",
              flush=True)
        return True

    def _process_one_pass(self, entry: Dict[str, Any]) -> bool:
        """v2 C4 two-pass: claim THIS worker's pass on the checkpoint, extract only this pass's
        block set on the ORIGINAL ``.pt`` (NOT renamed to .processing, so the other pass can read
        it concurrently), fsync the record, then drop the ``.<pass>.complete`` sentinel. Disposal
        (delete / kept-move) is deferred to :meth:`_finalize_sweep`, which fires once BOTH passes'
        ``.complete`` sentinels exist. Returns True iff this worker ran (and completed) its pass."""
        run_id, step, path, watch_dir = (entry["run_id"], entry["step"], entry["path"],
                                         entry["watch_dir"])
        if not claim_pass(path, self.pass_tag):
            return False  # lost the per-pass claim race, or pass already done
        extractor = self._extractor_for(run_id, watch_dir)
        try:
            # Extract on the .pt path itself (the file stays put for the other pass). Stateless.
            summary = extractor.extract_checkpoint(path)
        except Exception:
            tb = traceback.format_exc()
            print(f"[collector w{self.worker_id}] {self.pass_tag}-pass extract FAILED "
                  f"run_id={run_id} step={step}:\n{tb}", file=sys.stderr, flush=True)
            # Release our claim so it can be retried; never delete the checkpoint (§2.1).
            claim_marker = f"{path}.{self.pass_tag}.claim"
            if os.path.exists(claim_marker):
                try:
                    os.remove(claim_marker)
                except OSError:
                    pass
            return False
        # Record durable BEFORE marking the pass complete (§2.1 ordering).
        self._fsync_store(run_id)
        mark_pass_complete(path, self.pass_tag)
        nb_ok = summary.get("n_blocks_ok") if isinstance(summary, dict) else None
        t = summary.get("t_seconds") if isinstance(summary, dict) else None
        print(f"[collector w{self.worker_id}] {self.pass_tag}-pass DONE run_id={run_id} "
              f"step={step} blocks_ok={nb_ok} t={t}s "
              f"(all_passes_complete={all_passes_complete(path, self.require_passes)})", flush=True)
        return True

    def _finalize_sweep(self) -> int:
        """v2 C4: dispose of every checkpoint whose required passes are ALL complete. Each such
        checkpoint is claimed via the atomic ``.pt -> .processing`` rename (exactly one winner),
        then kept-moved (model-only) or deleted, then its per-pass sentinels are cleaned. This also
        recovers ORPHANS — both passes finished but the would-be finalizer died — since it runs
        every cycle over all dirs. No-op in legacy single-pass mode. Returns #finalized here."""
        if self.legacy_single_pass:
            return 0
        n = 0
        for wd in self.watch_dirs:
            if not os.path.isdir(wd):
                continue
            run_id = _run_id_for_dir(wd, self._dir_run_id)
            for entry_name in list(os.listdir(wd)):
                if not entry_name.endswith(".pt"):
                    continue
                path = os.path.join(wd, entry_name)
                if not all_passes_complete(path, self.require_passes):
                    continue
                step = _parse_step_from_name(path)
                if step is None:
                    continue
                proc = acquire_finalize(path, self.require_passes)  # atomic; one winner
                if proc is None:
                    continue  # lost the finalize race, or not all complete
                keep = self._should_keep(step, wd)
                try:
                    if keep:
                        disp = self._move_to_kept(run_id, step, proc, wd)
                    else:
                        disp = self._delete_from_queue(proc)
                finally:
                    cleanup_pass_sentinels(path, self.require_passes)
                n += 1
                print(f"[collector w{self.worker_id}] FINALIZED run_id={run_id} step={step} "
                      f"-> {disp}", flush=True)
        return n

    # ------------------------------------------------------------------ disposition

    def _restore_to_queue(self, proc: str, original_path: str) -> None:
        """On extraction failure, rename a claimed ``.processing`` checkpoint back to its watch-dir
        name (and re-drop ``.done``) so it returns to the work queue for a retry. Never deletes on
        failure (§2.1)."""
        try:
            if os.path.exists(proc):
                os.replace(proc, original_path)
            done = original_path + DONE_SUFFIX
            if not os.path.exists(done):
                with open(done, "wb") as f:
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:
            print(f"[collector w{self.worker_id}] could not restore {proc!r}: {e!r}",
                  file=sys.stderr, flush=True)

    def _delete_from_queue(self, proc: str) -> str:
        """Non-ladder checkpoint: delete it (and its ``.done``) entirely from the watch dir, so the
        watch dir is left with ZERO processed checkpoints (pure queue, contract A)."""
        base = proc[: -len(PROCESSING_SUFFIX)] if proc.endswith(PROCESSING_SUFFIX) else proc
        for p in (proc, base + DONE_SUFFIX):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                print(f"[collector w{self.worker_id}] delete {p!r} failed: {e!r}",
                      file=sys.stderr, flush=True)
        return "deleted"

    def _move_to_kept(self, run_id: str, step: int, proc: str, watch_dir: str) -> str:
        """KEPT ladder checkpoint: MOVE it out of the watch dir into ``<run>/kept/`` (contract A).

        Form decision (contract C, on-disk so it works across 2 workers): the latest
        ``keep_full_latest`` ladder steps stay FULL (resume points); every other kept ladder
        checkpoint is MODEL-ONLY (~6.6 GB -> ~1.2 GB). "Latest" is determined from the CURRENT
        contents of ``<run>/kept/`` (steps already kept on disk) UNIONED with this step — NOT from
        in-memory state — so two workers moving different steps converge on the same latest-N set.

        Mechanism: write the chosen form into ``<run>/kept/`` via
        :func:`save_checkpoint_atomic` (temp + fsync + os.replace + ``.done``), then remove the
        original claimed file + its watch-dir ``.done``/``.processing``. After this the watch dir
        holds nothing for this step. If a NEWLY-non-latest older full kept checkpoint is now bumped
        out of the latest-N window, re-slim it in ``<run>/kept/`` (model-only) so the run only ever
        keeps ``keep_full_latest`` full checkpoints on disk."""
        kept_dir = _kept_dir_for(watch_dir)
        os.makedirs(kept_dir, exist_ok=True)
        base_name = os.path.basename(proc[: -len(PROCESSING_SUFFIX)]
                                     if proc.endswith(PROCESSING_SUFFIX) else proc)
        dest = os.path.join(kept_dir, base_name)

        # Load the claimed checkpoint to decide + write its kept form.
        try:
            state = torch.load(proc, map_location="cpu", weights_only=False)
        except Exception as e:
            # Cannot load to move; leave the checkpoint in the queue for a retry (never lose it).
            print(f"[collector w{self.worker_id}] move-to-kept: load {proc!r} failed ({e!r}); "
                  f"restoring to queue.", file=sys.stderr, flush=True)
            self._restore_to_queue(proc, proc[: -len(PROCESSING_SUFFIX)]
                                   if proc.endswith(PROCESSING_SUFFIX) else proc)
            return "KEPT(restore-after-load-fail)"

        # Decide FULL vs model-only from the on-disk kept set unioned with this step. The latest-N
        # window is over POSITIVE steps only (the regular sampled ladder); step<=0 (init_model /
        # phase1_final / step00000000) are fixed ANCESTRAL ANCHORS that stay FULL forever (resume
        # roots) and are excluded from the rotation — so they are never bumped out / re-slimmed.
        is_anchor = int(step) <= 0
        kept_steps = set(self._discover_kept_ladder(kept_dir).keys())  # positive steps only
        if int(step) > 0:
            kept_steps.add(int(step))
        latest_full = set(sorted(kept_steps)[-self.keep_full_latest:]) \
            if self.keep_full_latest > 0 else set()

        is_full_src = _is_full_checkpoint(state)
        if is_anchor:
            # Ancestral anchor: keep FULL (resume root), outside the latest-N rotation.
            out_state = state
            form = "FULL anchor"
        elif int(step) in latest_full:
            # Keep FULL (resume point) — among the latest-N positive ladder steps.
            out_state = state
            form = f"FULL latest{self.keep_full_latest}"
        else:
            out_state = _slim_state(state) if is_full_src else state
            form = "model-only"
        try:
            save_checkpoint_atomic(out_state, dest)
        except Exception as e:
            print(f"[collector w{self.worker_id}] move-to-kept: write {dest!r} failed ({e!r}); "
                  f"restoring to queue.", file=sys.stderr, flush=True)
            del state
            self._restore_to_queue(proc, proc[: -len(PROCESSING_SUFFIX)]
                                   if proc.endswith(PROCESSING_SUFFIX) else proc)
            return "KEPT(restore-after-write-fail)"
        del state, out_state

        # The kept artifact is durable in kept/. Now remove the original from the watch dir.
        self._delete_from_queue(proc)

        # Re-slim any older full kept checkpoint that this newer one bumped out of latest-N, so the
        # run keeps only keep_full_latest full checkpoints on disk (self-healing across workers).
        self._reslim_bumped_full(run_id, kept_dir, latest_full)

        return f"KEPT(->kept/,{form})"

    @staticmethod
    def _discover_kept_ladder(kept_dir: str) -> Dict[int, str]:
        """Return ``{step: path}`` for every ``step*.pt`` checkpoint present in ``<run>/kept/``
        (with a ``.done`` sentinel). ``init_model.pt`` / ``phase1_final.pt`` are NOT included in the
        latest-N decision (they are fixed ancestral anchors)."""
        out: Dict[int, str] = {}
        if not os.path.isdir(kept_dir):
            return out
        for done in glob.glob(os.path.join(kept_dir, "step*.pt" + DONE_SUFFIX)):
            ckpt = done[: -len(DONE_SUFFIX)]
            if not os.path.exists(ckpt):
                continue
            s = _parse_step_from_name(ckpt)
            if s is None or s <= 0:
                continue
            out[s] = ckpt
        return out

    def _reslim_bumped_full(self, run_id: str, kept_dir: str, latest_full: Set[int]) -> None:
        """Re-slim (to model-only) any kept ``step*.pt`` that is FULL on disk but is NOT among the
        latest-N steps — i.e. an older resume point a newer kept checkpoint has bumped out of the
        window. On-disk driven, so it self-heals across 2 workers and across restarts."""
        for s, p in self._discover_kept_ladder(kept_dir).items():
            if s in latest_full:
                continue
            try:
                state = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not isinstance(state, dict) or "model_state_dict" not in state:
                del state
                continue
            if not _is_full_checkpoint(state):
                del state
                continue
            slim = _slim_state(state)
            del state
            try:
                save_checkpoint_atomic(slim, p)
                print(f"[collector w{self.worker_id}] re-slimmed kept run_id={run_id} step={s} "
                      f"MODEL-ONLY (bumped out of latest-{self.keep_full_latest}).", flush=True)
            except Exception as e:
                print(f"[collector w{self.worker_id}] re-slim {p!r} failed ({e!r}); left FULL.",
                      file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ durability

    def _fsync_store(self, run_id: str) -> None:
        """Flush + fsync this worker's out store for ``run_id`` so the extracted record is durable
        before the checkpoint is moved/deleted (briefing §2.1). The block code flushed the
        ScalarStore inside ``extract_checkpoint``; here we flush again (idempotent) and fsync the
        on-disk dataset/array dirs so the new part-file directory entries are durable on POSIX."""
        scalar = self._scalars.get(run_id)
        array = self._arrays.get(run_id)
        store_dirs: List[str] = []
        if scalar is not None:
            try:
                scalar.flush()
            except Exception:
                pass
            sd = getattr(scalar, "dataset_dir", None)
            if sd:
                store_dirs.append(sd)
        if array is not None:
            ap = getattr(array, "path", None)
            if ap:
                store_dirs.append(ap)
        store_dirs.append(os.path.join(self.out_store, run_id))
        for d in store_dirs:
            self._fsync_dir(d)

    @staticmethod
    def _fsync_dir(path: str) -> None:
        """fsync a directory's metadata (best-effort; not all FS / platforms support it)."""
        if not os.path.isdir(path):
            return
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except (OSError, PermissionError):
            pass

    # ------------------------------------------------------------------ main loops

    def run_once(self) -> int:
        """Drain all currently-ready checkpoints (oldest-first) and return the number processed by
        THIS worker. Used by ``--once`` and as the body of the polling loop. Two workers each run
        their own ``run_once``; the atomic claim divides the work between them."""
        if os.path.exists(DISK_PAUSE_SENTINEL):
            print(f"[collector w{self.worker_id}] DISK PAUSE active ({DISK_PAUSE_SENTINEL}); "
                  f"collector keeps DRAINING (extract+dispose frees disk). Trainer emission is "
                  f"the paused side.", flush=True)
        n_processed = 0
        # v2 C4: finalize any checkpoints whose passes all completed (incl. orphans) before draining.
        n_final = self._finalize_sweep()
        while True:
            queue = self._build_queue()
            if not queue:
                break
            entry = queue[0]
            ok = self._process_one(entry)
            if ok:
                n_processed += 1
            else:
                # Lost the claim race / failed: drop this entry and continue with the rest so a
                # contended entry (owned by the other worker) does not stall this worker. If it was
                # the only entry, the next _build_queue returns empty and we break.
                remaining = [e for e in queue
                             if not (e["run_id"] == entry["run_id"] and e["step"] == entry["step"])]
                if not remaining:
                    break
        # v2 C4: finalize again to dispose of checkpoints THIS worker's pass just completed (when it
        # was the second pass) without waiting for the next poll cycle.
        n_final += self._finalize_sweep()
        return n_processed + n_final

    def run_forever(self) -> None:
        """Poll the watch dirs forever, draining ready checkpoints each cycle and sleeping
        ``poll_interval`` seconds when the queue is empty."""
        print(f"[collector w{self.worker_id}] watching {len(self.watch_dirs)} dir(s); "
              f"out_store={self.out_store}; poll={self.poll_interval}s; "
              f"retention_latest_keep={self.retention_latest_keep}; "
              f"keep_full_latest={self.keep_full_latest}; heavy_every={self.heavy_every}",
              flush=True)
        try:
            while True:
                n = self.run_once()
                if n == 0:
                    time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print(f"[collector w{self.worker_id}] interrupted; flushing stores and exiting.",
                  flush=True)
        finally:
            self.close()

    def close(self) -> None:
        """Flush + close this worker's per-run stores on shutdown."""
        for scalar in self._scalars.values():
            try:
                scalar.close()
            except Exception:
                pass


def _resolve_worker_id(worker_id: Optional[str]) -> str:
    """Resolve the worker id used to namespace this collector's ScalarStore part dir. Explicit
    ``--worker-id`` wins; else ``SLURM_PROCID`` (set when SLURM launches >1 task); else a short
    random tag so two ad-hoc processes never share a part dir."""
    if worker_id:
        return str(worker_id)
    env = os.environ.get("SLURM_PROCID")
    if env:
        return str(env)
    return uuid.uuid4().hex[:8]


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PhenoVision PARALLEL checkpoint-extractor collector (REDESIGN BUILD C).")
    p.add_argument("--watch-dirs", nargs="+", required=True,
                   help="run checkpoint (watch) directories, e.g. "
                        "xAI/output/preadapt/<run_id>/checkpoints. Accepts multiple. Kept "
                        "checkpoints are moved to the sibling <run_id>/kept/.")
    p.add_argument("--out-store", required=True,
                   help="output store root; per-run stores go under <out-store>/<run_id>/. Each "
                        "worker writes its own scalars_w<worker_id>.parquet part dir.")
    p.add_argument("--poll-interval", type=float, default=10.0,
                   help="seconds to sleep when no ready checkpoints (default 10).")
    p.add_argument("--retention-latest-keep", type=int, default=2,
                   help="keep the most recent N emitted steps per run (plus log2-spaced steps) on "
                        "the never-delete retention ladder.")
    p.add_argument("--keep-full-latest", type=int, default=2,
                   help="keep the most recent N kept ladder checkpoints per run as FULL (optimizer "
                        "+ rng -> resumable); every other kept ladder checkpoint is MODEL-ONLY "
                        "(~6.6 GB -> ~1.2 GB). Kept checkpoints are moved to <run>/kept/.")
    p.add_argument("--once", action="store_true",
                   help="drain all currently-ready checkpoints once, then exit (no polling).")
    p.add_argument("--device", default="cuda",
                   help="torch device for the model + probe passes (default cuda).")
    p.add_argument("--worker-id", default=None,
                   help="id namespacing this worker's ScalarStore part dir so 2+ collectors over "
                        "one watch dir never collide on a part index. Default: SLURM_PROCID or a "
                        "random tag.")
    # Probe / Hessian comparability resources (briefing §4 invariants).
    p.add_argument("--probe-csv", default=None,
                   help="held-out probe-image CSV (file_name, fruiting, flowering). Preferred.")
    p.add_argument("--val-csv", default=None,
                   help="fallback CSV if --probe-csv is unset (same schema).")
    p.add_argument("--n-probe", type=int, default=1024,
                   help="number of probe images to subsample, STRATIFIED into n_probe//4 per joint "
                        "phenology cell (default 1024 -> 256/cell); fixed across runs/conditions.")
    p.add_argument("--hessian-batch-size", type=int, default=64,
                   help="size of the one fixed Hessian/Fisher batch (briefing §4 invariant 3).")
    p.add_argument("--probe-batch-size", type=int, default=64,
                   help="dataloader batch size for the probe passes.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="dataloader workers for the probe loaders.")
    p.add_argument("--heavy-every", type=int, default=3,
                   help="heavy interpretability blocks (§6.5/6.6/6.7) run every Nth checkpoint, "
                        "cheap blocks every checkpoint. The stateless extractor gates heavy blocks "
                        "on the checkpoint's own step (not an in-memory counter) so the cadence is "
                        "deterministic and worker-order-independent (default 3).")
    # --- v2 C4: CPU/GPU split + two-pass refcount ---
    p.add_argument("--block-set", choices=["all", "gpu", "cpu"], default="all",
                   help="which extraction blocks THIS worker runs. all (default, legacy) = every "
                        "block. gpu = curvature + fitness/interp/probes (model fwd/HVP). cpu = "
                        "optim/weights/trajectory/circuits (pure linalg on saved tensors, no model "
                        "forward; skips probe/Hessian build).")
    p.add_argument("--blocks", default=None,
                   help="comma-separated explicit block list (overrides --block-set).")
    p.add_argument("--require-passes", default="all",
                   help="comma-separated passes that must BOTH complete before a checkpoint is "
                        "disposed (delete/kept-move). 'all' (default) = LEGACY single-pass. For the "
                        "CPU/GPU split use 'gpu,cpu' on every worker (each worker's own pass = its "
                        "--block-set); a checkpoint is finalized only once both .complete sentinels "
                        "exist.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    blocks = [b.strip() for b in args.blocks.split(",")] if args.blocks else None
    require_passes = [p.strip() for p in args.require_passes.split(",") if p.strip()] or ["all"]
    collector = Collector(
        watch_dirs=args.watch_dirs,
        out_store=args.out_store,
        device=args.device,
        poll_interval=args.poll_interval,
        retention_latest_keep=args.retention_latest_keep,
        keep_full_latest=args.keep_full_latest,
        probe_csv=args.probe_csv,
        val_csv=args.val_csv,
        n_probe=args.n_probe,
        hessian_batch_size=args.hessian_batch_size,
        probe_batch_size=args.probe_batch_size,
        num_workers=args.num_workers,
        heavy_every=args.heavy_every,
        worker_id=args.worker_id,
        block_set=args.block_set,
        blocks=blocks,
        require_passes=require_passes,
    )
    if args.once:
        n = collector.run_once()
        collector.close()
        print(f"[collector w{collector.worker_id}] --once drained {n} checkpoint(s); exiting.",
              flush=True)
        return 0
    collector.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
