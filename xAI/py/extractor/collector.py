#!/usr/bin/env python3
"""
extractor/collector.py — the producer/consumer collector loop (briefing Part II §2, §2.1, §2.3).

The collector is the **consumer** in the producer/consumer architecture (briefing §2): the
training processes (producers) write checkpoints on the adaptive schedule of §3 plus a
``.done`` sentinel; this single collector watches all of their checkpoint directories,
maintains one shared work queue of ``(run_id, step, path)`` entries for ready checkpoints, and
processes them **strictly sequentially** (one at a time, so peak GPU memory is one model plus
its extraction activations — never three; briefing §2.3 Plan A). For each it claims the
checkpoint (atomic rename-to-``.processing``), runs the full §6 extraction via
:class:`extractor.extract.Extractor`, ensures the extracted record is durable (ScalarStore
flush + ArrayStore writes + an ``os.fsync`` of the store directory), and only **then** disposes
of the checkpoint (delete unless it is on the retention ladder). It never deletes before the
record is durable (briefing §2.1: write record -> fsync -> verify -> THEN delete).

Retention is **model-only** (locked /blue-frugality decision): a kept ladder checkpoint is
rewritten in place to ``{model_state_dict + minimal metadata}`` — dropping ``optimizer_state_dict``
and ``rng_state`` — shrinking it ~6.6 GB -> ~1.2 GB (:meth:`Collector._slim_in_place`). The
exception is the latest ``--keep-full-latest`` ladder checkpoints per run, which stay FULL so the
run is resumable from them (``preadapt_train --resume`` must target a FULL checkpoint; a model-only
one cannot restore Phase-2 optimizer state). Non-ladder checkpoints are deleted entirely.

Disk PAUSE: when the disk-safety watcher drops ``xAI/output/preadapt/_DISK_PAUSE`` it pauses the
TRAINER's emission, not the collector — the collector keeps DRAINING, because extract-then-delete
is exactly what frees disk. The collector only logs the pause state at the top of each cycle.

Per-run fairness (briefing §2.3 "Fairness"): schedule oldest-first by wall-time, but cap how
many consecutive checkpoints come from one ``run_id`` (``--max-consec-per-run``); and if any
run_id has more than two queued while another has had none processed this cycle, serve the
starved run first — so a fast-moving trainer cannot monopolize the collector and let another
condition's checkpoints pile past its retention ladder. Records and the retention ladder are
**per-run_id**: one :class:`Extractor` and one :class:`RetentionLadder` per run_id (so the
Q0 / prev-checkpoint carried state in §6.1 / §6.2 stays per-trajectory).

Idempotency / crash-safety (briefing §2.1): a restart skips steps already present in the out
store (a per-run processed-steps set, seeded from :func:`read_scalars`). Claiming is the atomic
lock (``os.rename`` of the checkpoint to ``.processing``); a second worker or a restart that
loses the rename simply moves on.

---------------------------------------------------------------------------------------------
Utilization pilot note (briefing §2.3 — RUN ON A PILOT BEFORE LAUNCHING THE FULL BLOCK).
Confirm one collector can keep up with the (here two, per the locked plan C5) concurrent
trainers:

    T_ex         = full extraction wall-time on one ViT-L checkpoint
                   (this optimizer, this probe set, this Lanczos budget)  -- printed per ckpt
                   as ``t_seconds`` in the collector log; average it over a pilot.
    sec_per_step = measured training step time.
    r_max        = 1 / (min_gap_steps * sec_per_step)   # worst-case emissions/sec per trainer
    utilization  = N_trainers * r_max * T_ex             # N_trainers concurrent, one collector

Keep ``utilization`` below ~0.7 for burst headroom, and set the trainer's ``min_gap_steps``
(the §3 sampler floor) from this inequality rather than from convenience. The stress window is
early training, when all trainers cross the displacement threshold quickly and emit together;
the adaptive sampler front-loads emission, so the whole-run average sits well below the early
peak — size for the peak. If ``utilization`` exceeds ~0.7: raise ``min_gap_steps``, cut Lanczos
iterations (``block_curvature.LANCZOS_K`` / ``LANCZOS_N_ITER``) or the probe-set size
(``--n-probe`` / ``--hessian-batch-size``), and re-measure. The retention ladder absorbs
transient early bursts; sustained over-subscription is the failure mode to avoid (escalate per
the briefing rather than letting disk fill).
---------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
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
    is_ready,
    claim_for_processing,
    finalize_processed,
    save_checkpoint_atomic,
    RetentionLadder,
    read_scalars,
    read_manifest,
)
from extractor.extract import Extractor  # noqa: E402


# =============================================================================
# Disk-pause sentinel + model-only retention ladder
# =============================================================================

#: Disk-pressure PAUSE sentinel. The disk-safety watcher (WARN + AUTO-PAUSE) drops this file
#: when group ``/blue`` free space crosses the low-water mark; it pauses the TRAINER's *emission*
#: but the collector must keep DRAINING — extract-then-delete is precisely what frees disk, so
#: stopping the collector here would be counterproductive. The collector therefore only LOGS the
#: pause state at the top of each cycle and otherwise behaves identically (the locked decision).
DISK_PAUSE_SENTINEL = os.path.join(
    _PROJECT_ROOT, "xAI", "output", "preadapt", "_DISK_PAUSE"
)

#: Metadata keys preserved when a ladder checkpoint is rewritten MODEL-ONLY. Everything else
#: (notably ``optimizer_state_dict`` and ``rng_state``) is dropped, shrinking a kept ViT-L ladder
#: checkpoint from ~6.6 GB to ~1.2 GB. These are the minimal provenance fields the §6 extractor /
#: downstream analysis read from a checkpoint dict; ``global_step`` mirrors ``step`` for resume-
#: style readers, and ``rng_version`` is a tiny tag worth keeping if present.
SLIM_META_KEYS: Tuple[str, ...] = (
    "step", "epoch", "phase", "condition", "variant", "seed", "run_id",
    "global_step", "rng_version", "wall_time", "reason", "param_names",
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
    Phase-2 init (step = the Phase-2 start step), and ``phase1_final.pt`` for the equalization
    endpoint. We map init/phase1 to step 0 (they are the ancestral references; the actual step
    is read from the checkpoint dict by ``build_ctx``, but the queue ordering only needs a
    monotone key, and 0 sorts them first which is correct — init/phase1 precede sampled steps).
    """
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
    claimed (``.processing``) — :func:`is_ready`. We use the file mtime as the wall-time order
    key for cross-run scheduling (oldest-first), and the parsed step as the per-run order key.
    """
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
    """Resolve the ``run_id`` for a watch directory from its run manifest (cached). Falls back
    to the parent directory name (which is the run_id under the trainer's
    ``xAI/output/preadapt/<run_id>/checkpoints`` layout)."""
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
        # <output_dir>/checkpoints -> run_id is the output dir's basename.
        parent = os.path.dirname(os.path.abspath(watch_dir.rstrip("/")))
        rid = os.path.basename(parent) or os.path.basename(watch_dir.rstrip("/"))
    cache[watch_dir] = rid
    return rid


# =============================================================================
# Collector
# =============================================================================

class Collector:
    """Single-consumer collector over multiple producer checkpoint directories."""

    def __init__(
        self,
        watch_dirs: List[str],
        out_store: str,
        device: str = "cuda",
        poll_interval: float = 10.0,
        max_consec_per_run: int = 3,
        retention_latest_keep: int = 2,
        keep_full_latest: int = 2,
        probe_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        n_probe: int = 1024,
        hessian_batch_size: int = 64,
        probe_batch_size: int = 64,
        num_workers: int = 4,
        heavy_every: int = 3,
    ):
        self.watch_dirs = [os.path.abspath(d) for d in watch_dirs]
        self.out_store = os.path.abspath(out_store)
        os.makedirs(self.out_store, exist_ok=True)
        self.device = device
        self.poll_interval = float(poll_interval)
        self.max_consec_per_run = int(max_consec_per_run)
        self.retention_latest_keep = int(retention_latest_keep)
        self.keep_full_latest = max(0, int(keep_full_latest))
        self.probe_csv = probe_csv
        self.val_csv = val_csv
        self.n_probe = int(n_probe)
        self.hessian_batch_size = int(hessian_batch_size)
        self.probe_batch_size = int(probe_batch_size)
        self.num_workers = int(num_workers)
        self.heavy_every = max(1, int(heavy_every))

        # Per-run_id state: Extractor (carries Q0/prev), RetentionLadder, processed-step set.
        self._extractors: Dict[str, Extractor] = {}
        self._ladders: Dict[str, RetentionLadder] = {}
        self._processed: Dict[str, Set[int]] = {}
        # Per-run map step -> on-disk path of ladder checkpoints currently held FULL (with
        # optimizer + rng state). Used to (a) decide which ladder ckpts stay full as the
        # latest-N resume points, and (b) re-slim an OLDER full ladder ckpt once a newer one
        # has bumped it out of the latest-N window (it is no longer needed for resume).
        self._full_ladder_paths: Dict[str, Dict[int, str]] = {}
        # Map watch_dir -> run_id (and run_dir for the Extractor's manifest lookup).
        self._dir_run_id: Dict[str, str] = {}
        # Track which run_id was served in the previous cycle (for the starvation rule).
        self._last_consec_run: Optional[str] = None
        self._last_consec_count: int = 0

    # ------------------------------------------------------------------ per-run resources

    def _extractor_for(self, run_id: str, watch_dir: str) -> Extractor:
        """Lazily build (and cache) the per-run Extractor. ``watch_dir`` is the run's
        checkpoint dir; the Extractor reads the run manifest from it / its parent. All runs
        share ONE out store (rows are keyed by run_id), so we pass a per-run store rooted at
        ``<out_store>/<run_id>/`` for a clean per-run on-disk layout that is still queryable
        together via read_scalars over each subdir."""
        ex = self._extractors.get(run_id)
        if ex is None:
            ex = Extractor(
                run_dir=watch_dir,
                out_store_dir=self.out_store,
                device=self.device,
                probe_csv=self.probe_csv,
                val_csv=self.val_csv,
                n_probe=self.n_probe,
                hessian_batch_size=self.hessian_batch_size,
                probe_batch_size=self.probe_batch_size,
                num_workers=self.num_workers,
                run_id=run_id,
                heavy_every=self.heavy_every,
            )
            self._extractors[run_id] = ex
        return ex

    def _ladder_for(self, run_id: str) -> RetentionLadder:
        lad = self._ladders.get(run_id)
        if lad is None:
            lad = RetentionLadder(latest_keep=self.retention_latest_keep)
            self._ladders[run_id] = lad
        return lad

    def _processed_for(self, run_id: str) -> Set[int]:
        """Per-run processed-step set, seeded once from the out store (idempotency on restart):
        a step already present in the run's scalar store is skipped (briefing §2.1)."""
        if run_id in self._processed:
            return self._processed[run_id]
        seen: Set[int] = set()
        store_dir = os.path.join(self.out_store, run_id, "scalars.parquet")
        try:
            df = read_scalars(store_dir)
            # The store directory is ALREADY per-run (<out_store>/<run_id>/), so every row in
            # it belongs to this run by construction. Do NOT additionally filter on
            # ``df["run_id"] == run_id``: the rows carry the *checkpoint's embedded* run_id
            # (from build_ctx), while ``run_id`` here is the collector's directory/manifest-
            # derived id; if those diverge (e.g. a run with no manifest, where the id falls back
            # to the dir name) the equality filter would return an empty set and silently break
            # restart idempotency, reprocessing already-extracted checkpoints.
            if df is not None and len(df) and "step" in df:
                seen = set(int(s) for s in df["step"].dropna().unique())
        except Exception:
            seen = set()
        self._processed[run_id] = seen
        return seen

    # ------------------------------------------------------------------ scheduling

    def _build_queue(self) -> List[Dict[str, Any]]:
        """Scan all watch dirs and build the shared work queue of ready, not-yet-processed
        checkpoints. Each entry: ``{run_id, step, path, mtime, watch_dir}``. Sorted oldest-first
        by wall-time (mtime), then by step, then run_id for a stable order."""
        entries: List[Dict[str, Any]] = []
        for wd in self.watch_dirs:
            if not os.path.isdir(wd):
                continue
            run_id = _run_id_for_dir(wd, self._dir_run_id)
            processed = self._processed_for(run_id)
            for step, path, mtime in _discover_ready(wd):
                if step in processed:
                    continue
                entries.append({"run_id": run_id, "step": step, "path": path,
                                "mtime": mtime, "watch_dir": wd})
        entries.sort(key=lambda e: (e["mtime"], e["step"], e["run_id"]))
        return entries

    def _pick_next(self, queue: List[Dict[str, Any]],
                   served_this_cycle: Set[str]) -> Optional[Dict[str, Any]]:
        """Pick the next checkpoint to process from ``queue`` with per-run fairness
        (briefing §2.3).

        Rule (in priority order):
          1. **Anti-starvation:** if some run_id has > 2 queued while another run_id that has
             ready work has had NONE served this cycle, serve the starved run's oldest first.
          2. **Consecutive cap:** otherwise serve the oldest, unless that would exceed
             ``max_consec_per_run`` consecutive checkpoints from the same run_id and another
             run_id has ready work — in which case serve the oldest of a different run.
          3. **Default:** the globally oldest ready checkpoint.
        """
        if not queue:
            return None

        # Per-run counts in the current queue.
        counts: Dict[str, int] = {}
        for e in queue:
            counts[e["run_id"]] = counts.get(e["run_id"], 0) + 1
        run_ids_with_work = set(counts.keys())

        # (1) Anti-starvation: a run with >2 queued while a different ready run had none served.
        starved = [rid for rid in run_ids_with_work
                   if rid not in served_this_cycle]
        heavy = [rid for rid in run_ids_with_work if counts[rid] > 2]
        if starved and heavy and any(h not in starved for h in heavy):
            # Serve the oldest checkpoint of the starved run that has the *least* service.
            starved_set = set(starved)
            for e in queue:  # queue is oldest-first
                if e["run_id"] in starved_set:
                    return e

        # (2) Consecutive cap: avoid monopoly if another run has ready work.
        oldest = queue[0]
        if (self._last_consec_run == oldest["run_id"]
                and self._last_consec_count >= self.max_consec_per_run
                and len(run_ids_with_work) > 1):
            for e in queue:
                if e["run_id"] != oldest["run_id"]:
                    return e

        # (3) Default: globally oldest.
        return oldest

    # ------------------------------------------------------------------ processing one ckpt

    def _process_one(self, entry: Dict[str, Any]) -> bool:
        """Claim, extract, durably persist, then finalize (delete-or-keep) one checkpoint.

        Returns True iff the checkpoint was processed (record durable) — so the caller can
        update fairness/processed bookkeeping. False if it was skipped (lost the claim race,
        already processed, or extraction failed before any record).
        """
        run_id = entry["run_id"]
        step = entry["step"]
        path = entry["path"]
        watch_dir = entry["watch_dir"]

        # Idempotency double-check just before claiming.
        if step in self._processed_for(run_id):
            return False

        # Atomic claim (the lock). Lose the race -> skip.
        proc = claim_for_processing(path)
        if proc is None:
            return False

        ex = self._extractor_for(run_id, watch_dir)
        ladder = self._ladder_for(run_id)
        keep = ladder.should_keep(step)

        try:
            summary = ex.extract_checkpoint(proc)
        except Exception as e:
            # Extraction blew up before producing a usable record. Do NOT delete the checkpoint
            # (briefing §2.1: never delete before the record is durable) — restore it so a later
            # pass / a human can retry, and log loudly.
            tb = traceback.format_exc()
            print(f"[collector] extract_checkpoint FAILED for run_id={run_id} step={step}:\n{tb}",
                  file=sys.stderr, flush=True)
            try:
                finalize_processed(proc, keep=True, original_path=path)  # keep => restore it
            except Exception as e2:
                print(f"[collector] could not restore {proc!r} after failure: {e2!r}",
                      file=sys.stderr, flush=True)
            return False

        # --- ensure the record is DURABLE before disposing of the checkpoint (§2.1) ---
        self._fsync_store(run_id)

        # Mark processed, THEN finalize (delete unless on the retention ladder).
        self._processed_for(run_id).add(step)
        try:
            finalize_processed(proc, keep=keep, original_path=path)
        except Exception as e:
            print(f"[collector] finalize_processed failed for run_id={run_id} step={step}: "
                  f"{e!r}", file=sys.stderr, flush=True)

        # --- model-only retention ladder (locked decision) ---------------------------------
        # finalize_processed restored the kept ckpt to ``path`` (FULL, ~6.6 GB). Now decide its
        # on-disk form: the latest --keep-full-latest ladder steps stay FULL (resumable from the
        # optimizer + rng state); every other kept ladder ckpt is rewritten MODEL-ONLY (~1.2 GB).
        # Re-slimming a NEWLY-non-latest older full ladder ckpt also happens here, so the run only
        # ever holds keep_full_latest full checkpoints on disk.
        if keep:
            disp = self._apply_retention_form(run_id, step, path, watch_dir)
        else:
            disp = "deleted"

        nb_ok = summary.get("n_blocks_ok")
        nb_err = summary.get("n_blocks_err")
        t = summary.get("t_seconds")
        heavy = summary.get("heavy")
        print(f"[collector] processed run_id={run_id} step={step} "
              f"heavy={heavy} blocks_ok={nb_ok} blocks_err={nb_err} t={t}s -> {disp}",
              flush=True)
        return True

    # ------------------------------------------------------------------ retention form

    def _apply_retention_form(self, run_id: str, step: int, path: str, watch_dir: str) -> str:
        """For a checkpoint that should be KEPT on the ladder, choose its on-disk form and apply
        it: keep the latest ``keep_full_latest`` ladder steps FULL (resume points), rewrite every
        other kept ladder checkpoint MODEL-ONLY (~1.2 GB). Also re-slims any OLDER full ladder
        checkpoint that this newer one bumped out of the latest-N window.

        The candidate ladder set is discovered from the on-disk ``step*.pt`` files in
        ``watch_dir``, restricted to steps this collector has ALREADY PROCESSED (kept), and folded
        into the in-memory tracker — so the policy is **self-healing across collector restarts**:
        a restart that lost ``_full_ladder_paths`` still re-slims any pre-restart full ladder
        checkpoint that is no longer among the latest-N. The processed-set restriction is essential:
        an on-disk checkpoint that is still READY-but-unprocessed has not had its keep decision yet
        and is owned by its own :meth:`_process_one` call, so it must NOT be folded in here (else a
        newer in-queue checkpoint could wrongly bump a kept one out of the latest-N before its turn).
        The latest-N by step stay FULL; the rest become model-only.

        Returns a short disposition string for the log line.
        """
        full_paths = self._full_ladder_paths.setdefault(run_id, {})
        # This step's ckpt is currently FULL on disk (finalize restored it). Register it, and fold
        # in any other ALREADY-PROCESSED on-ladder step*.pt present in the dir (covers a restart's
        # pre-existing kept set). Unprocessed ready checkpoints are deliberately excluded.
        full_paths[step] = path
        processed = self._processed_for(run_id)
        for cand_step, cand_path in self._discover_ladder_ckpts(watch_dir).items():
            if cand_step in processed:
                full_paths.setdefault(cand_step, cand_path)

        # The latest-N ladder steps (by step) stay FULL; everything else becomes model-only.
        latest_full = set(sorted(full_paths.keys())[-self.keep_full_latest:]) \
            if self.keep_full_latest > 0 else set()

        for s in sorted(full_paths.keys()):
            if s in latest_full:
                continue
            p = full_paths.get(s)
            if p and self._slim_in_place(run_id, s, p):
                # Slimmed (or already model-only / gone): drop from the full tracker so we do not
                # reload it every pass. It stays on the ladder on disk, just model-only now.
                full_paths.pop(s, None)

        if step in latest_full:
            return f"KEPT(ladder,FULL latest{self.keep_full_latest})"
        # This step itself is not in the latest-N (keep_full_latest small / ladder dense early) —
        # its slim happened in the loop above.
        return "KEPT(ladder,model-only)"

    @staticmethod
    def _discover_ladder_ckpts(watch_dir: str) -> Dict[int, str]:
        """Return ``{step: path}`` for every on-LADDER ``step*.pt`` checkpoint present (with a
        ``.done`` sentinel) in ``watch_dir`` — i.e. surviving sampled checkpoints, which after the
        collector has run are by construction ladder members (power-of-two steps or the recent-N).
        Used to make the model-only retention policy self-healing across restarts. ``init_model.pt``
        / ``phase1_final.pt`` (the FULL ancestral resume anchors) are NOT touched here."""
        out: Dict[int, str] = {}
        for done in glob.glob(os.path.join(watch_dir, "step*.pt" + DONE_SUFFIX)):
            ckpt = done[: -len(DONE_SUFFIX)]
            if not os.path.exists(ckpt):
                continue
            s = _parse_step_from_name(ckpt)
            if s is None or s <= 0:
                continue
            out[s] = ckpt
        return out

    def _slim_in_place(self, run_id: str, step: int, path: str) -> bool:
        """Load the FULL checkpoint at ``path``, strip it to ``{model_state_dict + minimal
        metadata}`` (dropping optimizer + rng state), and atomically rewrite it in place via
        :func:`save_checkpoint_atomic` (temp + os.replace + refreshed ``.done``). No-op (returns
        True) if it is already model-only. Returns True on success (or already-slim), False on
        error (the FULL checkpoint is left intact and stays tracked for a retry next pass)."""
        if not os.path.exists(path):
            # Already gone (e.g. a manual cleanup); nothing to slim, drop from tracking.
            return True
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[collector] slim: could not load {path!r} for run_id={run_id} step={step}: "
                  f"{e!r}; leaving FULL.", file=sys.stderr, flush=True)
            return False
        if not isinstance(state, dict) or "model_state_dict" not in state:
            print(f"[collector] slim: {path!r} has no model_state_dict; leaving as-is.",
                  file=sys.stderr, flush=True)
            return True
        if not _is_full_checkpoint(state):
            return True  # already model-only; idempotent
        slim = _slim_state(state)
        # Free the heavy tensors before the rewrite so peak memory is one model, not two.
        del state
        try:
            save_checkpoint_atomic(slim, path)  # temp + fsync + os.replace + refresh .done
        except Exception as e:
            print(f"[collector] slim: rewrite FAILED for {path!r} (run_id={run_id} step={step}): "
                  f"{e!r}; the FULL checkpoint is intact.", file=sys.stderr, flush=True)
            return False
        print(f"[collector] slim: run_id={run_id} step={step} rewritten MODEL-ONLY "
              f"(dropped optimizer + rng) -> {os.path.basename(path)}", flush=True)
        return True

    def _fsync_store(self, run_id: str) -> None:
        """Flush + fsync the run's out store so the extracted record is durable before the
        checkpoint is deleted (briefing §2.1). The ScalarStore was flushed inside
        ``extract_checkpoint``; here we fsync the store directory file descriptors so the
        Parquet part files and Zarr chunks are on stable storage. fsync of a directory makes
        the new directory entries durable on POSIX filesystems."""
        ex = self._extractors.get(run_id)
        store_dirs: List[str] = []
        if ex is not None and ex.scalar is not None:
            try:
                ex.scalar.flush()
            except Exception:
                pass
            # Use the store objects' authoritative on-disk dirs rather than guessing the
            # names: ScalarStore writes part files into ``<stem>_parts`` (e.g.
            # ``scalars_parts``), NOT ``scalars`` — fsync the real dataset dir so the new
            # part file's directory entry is durable before the ckpt is deleted (§2.1).
            sd = getattr(ex.scalar, "dataset_dir", None)
            if sd:
                store_dirs.append(sd)
        if ex is not None and ex.array is not None:
            ap = getattr(ex.array, "path", None)
            if ap:
                store_dirs.append(ap)
        store_root = os.path.join(self.out_store, run_id)
        store_dirs.append(store_root)
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
            pass  # e.g. some network filesystems; the prior file fsyncs already covered data

    # ------------------------------------------------------------------ main loops

    def run_once(self) -> int:
        """Drain all currently-ready checkpoints (oldest-first, with fairness) and return the
        number processed. Used by ``--once`` and as the body of the polling loop."""
        # Disk PAUSE state: the disk-safety watcher pauses the TRAINER's emission, not the
        # collector — draining (extract-then-delete) is what FREES disk, so the collector must
        # keep consuming under pause. Log the state; do not change draining behavior.
        if os.path.exists(DISK_PAUSE_SENTINEL):
            print(f"[collector] DISK PAUSE active ({DISK_PAUSE_SENTINEL}); collector keeps "
                  f"DRAINING (extract+delete frees disk). Trainer emission is the paused side.",
                  flush=True)
        n_processed = 0
        served_this_cycle: Set[str] = set()
        while True:
            queue = self._build_queue()
            entry = self._pick_next(queue, served_this_cycle)
            if entry is None:
                break
            run_id = entry["run_id"]
            ok = self._process_one(entry)
            if ok:
                n_processed += 1
                served_this_cycle.add(run_id)
                # Update consecutive-run tracking for the monopoly cap.
                if self._last_consec_run == run_id:
                    self._last_consec_count += 1
                else:
                    self._last_consec_run = run_id
                    self._last_consec_count = 1
            else:
                # Could not process this entry (race / failure). Avoid a tight spin on it:
                # if nothing else is ready either, break so the caller can poll/sleep.
                remaining = [e for e in queue
                             if not (e["run_id"] == entry["run_id"] and e["step"] == entry["step"])]
                if not remaining:
                    break
        return n_processed

    def run_forever(self) -> None:
        """Poll the watch dirs forever, draining ready checkpoints each cycle and sleeping
        ``poll_interval`` seconds when the queue is empty. Sized for the §2.3 utilization
        budget; the sleep only happens when there is no ready work."""
        print(f"[collector] watching {len(self.watch_dirs)} dir(s); out_store={self.out_store}; "
              f"poll={self.poll_interval}s; max_consec_per_run={self.max_consec_per_run}; "
              f"retention_latest_keep={self.retention_latest_keep}; "
              f"keep_full_latest={self.keep_full_latest}; heavy_every={self.heavy_every}",
              flush=True)
        try:
            while True:
                n = self.run_once()
                if n == 0:
                    time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("[collector] interrupted; flushing stores and exiting.", flush=True)
        finally:
            self.close()

    def close(self) -> None:
        """Flush + close all per-run Extractor stores on shutdown."""
        for ex in self._extractors.values():
            try:
                ex.close()
            except Exception:
                pass


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PhenoVision checkpoint-extractor collector (briefing Part II §2).")
    p.add_argument("--watch-dirs", nargs="+", required=True,
                   help="run checkpoint directories to watch (one per condition/run), e.g. "
                        "xAI/output/preadapt/<run_id>/checkpoints. Accepts multiple.")
    p.add_argument("--out-store", required=True,
                   help="output store root; per-run ScalarStore + ArrayStore go under "
                        "<out-store>/<run_id>/.")
    p.add_argument("--poll-interval", type=float, default=10.0,
                   help="seconds to sleep when no ready checkpoints (default 10).")
    p.add_argument("--max-consec-per-run", type=int, default=3,
                   help="cap on consecutive checkpoints from one run_id before yielding to "
                        "another run with ready work (fairness; briefing §2.3).")
    p.add_argument("--retention-latest-keep", type=int, default=2,
                   help="keep the most recent N emitted steps per run (plus log2-spaced steps) "
                        "on the never-delete retention ladder (briefing §2).")
    p.add_argument("--keep-full-latest", type=int, default=2,
                   help="keep the most recent N ladder checkpoints per run as FULL (optimizer + "
                        "rng state) so the run is resumable from them; every other kept ladder "
                        "checkpoint is rewritten MODEL-ONLY (~6.6 GB -> ~1.2 GB). RESUME NOTE: "
                        "`preadapt_train --resume auto` picks the latest step*.pt, and a "
                        "model-only checkpoint cannot restore Phase-2 optimizer state, so a "
                        "resume must target a FULL one — the latest-N kept here are full "
                        "(default 2).")
    p.add_argument("--once", action="store_true",
                   help="drain all currently-ready checkpoints once, then exit (no polling).")
    p.add_argument("--device", default="cuda",
                   help="torch device for the model + probe passes (default cuda).")
    # Probe / Hessian comparability resources (briefing §4 invariants).
    p.add_argument("--probe-csv", default=None,
                   help="held-out probe-image CSV (file_name, fruiting, flowering). Preferred.")
    p.add_argument("--val-csv", default=None,
                   help="fallback CSV if --probe-csv is unset (same schema).")
    p.add_argument("--n-probe", type=int, default=1024,
                   help="number of probe images to subsample, STRATIFIED into n_probe//4 per "
                        "joint phenology cell (default 1024 -> 256/cell); fixed across "
                        "runs/conditions.")
    p.add_argument("--hessian-batch-size", type=int, default=64,
                   help="size of the one fixed Hessian/Fisher batch (briefing §4 invariant 3).")
    p.add_argument("--probe-batch-size", type=int, default=64,
                   help="dataloader batch size for the probe passes.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="dataloader workers for the probe loaders.")
    p.add_argument("--heavy-every", type=int, default=3,
                   help="heavy interpretability blocks (§6.5/6.6/6.7) run every Nth checkpoint, "
                        "cheap blocks every checkpoint. Per-run counter 0,N,2N,... get the heavy "
                        "blocks, so the first/init checkpoint is always a full reference; <=1 "
                        "runs heavy blocks every checkpoint (default 3).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    collector = Collector(
        watch_dirs=args.watch_dirs,
        out_store=args.out_store,
        device=args.device,
        poll_interval=args.poll_interval,
        max_consec_per_run=args.max_consec_per_run,
        retention_latest_keep=args.retention_latest_keep,
        keep_full_latest=args.keep_full_latest,
        probe_csv=args.probe_csv,
        val_csv=args.val_csv,
        n_probe=args.n_probe,
        hessian_batch_size=args.hessian_batch_size,
        probe_batch_size=args.probe_batch_size,
        num_workers=args.num_workers,
        heavy_every=args.heavy_every,
    )
    if args.once:
        n = collector.run_once()
        collector.close()
        print(f"[collector] --once drained {n} checkpoint(s); exiting.", flush=True)
        return 0
    collector.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
