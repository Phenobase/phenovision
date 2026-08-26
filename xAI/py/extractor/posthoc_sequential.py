#!/usr/bin/env python3
"""
extractor/posthoc_sequential.py — POST-HOC SEQUENTIAL cross-checkpoint diagnostics
(REDESIGN CONTRACT (D)).

Background (the redesign)
-------------------------
In the new collector-bottlenecked architecture the per-checkpoint extraction is made FULLY
STATELESS (CONTRACT (B)): ``extract.extract_checkpoint`` builds its context from the checkpoint
plus fixed shared resources and never carries cross-checkpoint in-memory state, so two collector
processes can extract DIFFERENT checkpoints concurrently and out of order. The diagnostics that
are inherently CROSS-CHECKPOINT therefore CANNOT run on the live path. They are deferred to this
ONE-SHOT post-hoc pass (CONTRACT (C)/(D)), which runs ONCE per run after the trainer + both
collectors have finished, reads the per-run store + the trainer's inline log, and writes the
cross-checkpoint diagnostics back into the SAME store.

What this pass computes
-----------------------
* **§6.1 rotation (was the live ``block_optim`` principal_angles).** Per (layer, step) the live
  ``block_optim`` now stores ONLY the top-k QL/QR EIGENVECTORS to the array store (each
  per-checkpoint independent). Here we read those stored bases back, and for each consecutive
  step pair (sorted by step) compute the principal angles between ``QL_t`` and ``QL_{t-1}`` (and
  QR), plus vs the EARLIEST available step (the init / ancestral reference). The mean-cosine
  alignment is written as scalars ``ql_rotation_vs_prev`` / ``qr_rotation_vs_prev`` /
  ``ql_rotation_vs_init`` / ``qr_rotation_vs_init`` (per layer per step) and the full angle
  vectors to the array store (``ql_principal_angles_vs_prev`` etc.). Reuses
  ``two_noise/curvature/lanczos.principal_angles`` — the same routine the old live block used.

* **§6.3 trajectory (was the live ``block_trajectory`` MSD / straightness / velocity-autocorr).**
  Read the GLOBAL projected-coordinate trajectory ``proj_coords`` per step (from the array store
  group ``proj_coords`` and/or the trainer's inline log) and compute, per run (global; per-layer
  optional from ``proj_coords_layer``):
    - **straightness** (net displacement / path length): preferred uses the trainer's inline
      ``path_length`` (true walked distance) with net = ``||x_last - x_first||`` in proj space;
      falls back to a proj-space net/path proxy when the inline path length is unavailable.
    - **MSD(tau)** = ``mean_t ||x_{t+tau} - x_t||^2`` vs lag, its log-log slope (~2 ballistic /
      ~1 diffusive / ~0 confined) and a crude plateau flag; the full MSD(tau) curve is written to
      the array store. Lags are TRUE step gaps (handles non-uniform sampling): we resample the
      irregular (step, coord) series onto the recorded step grid and form lags in step units.
    - **velocity autocorrelation** for small lags, read straight from the trainer's inline
      ``velocity_autocorr`` (lag in the ``head`` column) and re-emitted as the run-level
      ``velocity_autocorr_final`` per lag (the converged value over the whole run).

Robustness / idempotency
-------------------------
Every piece is independently guarded: a missing array group, an unreadable inline log, or a
degenerate (single-step) trajectory is SKIPPED with a logged warning, never fatal. The pass is
idempotent — re-running it appends a fresh ScalarStore part with the same rows (the tidy-long
store is append-only; downstream readers de-dup on (run_id, step, quantity, layer, head) by taking
the latest, and the array store overwrites a (group, step) slice idempotently is NOT guaranteed,
so array writes are guarded to append only steps not already present for that group/layer).

This module touches NO other file. It imports the store classes + projection-free helpers from
``preadapt_common`` and ``principal_angles`` from the two_noise curvature package; it does not
import the per-checkpoint blocks (no model load, no GPU needed — CPU only).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- project paths: mirror the extractor modules so preadapt_common + two_noise import. -----
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
    read_scalars,
    read_manifest,
)

# Floors / guards for log-space and division stability (mirror the old live blocks).
_EPS: float = 1e-12

# Minimum number of accumulated trajectory points needed to estimate an MSD slope
# (mirrors block_trajectory._MSD_MIN_POINTS so the diagnostic is comparable to the old live one).
_MSD_MIN_POINTS: int = 4

# Inline-logged quantity names the trainer writes (preadapt_train.py §5 / VELOCITY_LAGS).
_PATH_LENGTH_Q = "path_length"
_VELOCITY_NORM_Q = "velocity_norm"
_VELOCITY_AUTOCORR_Q = "velocity_autocorr"  # lag is stored in the `head` column

#: Candidate ARRAY-STORE group names the live ``block_optim`` may use for the stored top-k
#: QL / QR eigenvectors (CONTRACT (C): "LIVE = store top-k QL/QR eigvectors (array store)"). The
#: exact group name is owned by the (separately-built) live block; we accept any of these so this
#: post-hoc pass conforms regardless of which the live block lands on. First match per side wins.
_QL_EIGVEC_GROUPS: Tuple[str, ...] = ("ql_eigvecs", "ql_basis", "ql_topk_eigvecs", "QL", "ql")
_QR_EIGVEC_GROUPS: Tuple[str, ...] = ("qr_eigvecs", "qr_basis", "qr_topk_eigvecs", "QR", "qr")

#: GLOBAL projected-coordinate trajectory group (block_weights writes ``proj_coords`` global;
#: per-layer coords live in the distinct ``proj_coords_layer`` group — see block_weights).
_PROJ_COORDS_GROUP = "proj_coords"
_PROJ_COORDS_LAYER_GROUP = "proj_coords_layer"


def _log(msg: str) -> None:
    print(f"[posthoc_sequential] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[posthoc_sequential] WARNING: {msg}", file=sys.stderr, flush=True)


# =============================================================================
# Array-store key discovery (per-layer keys are NESTED under a group on the Zarr backend)
# =============================================================================

def _all_logical_keys(array: ArrayStore) -> List[str]:
    """Enumerate EVERY logical array key in the store, INCLUDING per-layer nested keys
    (``<group>/layer=<name>``).

    ``ArrayStore.keys()`` is NOT sufficient: on the Zarr (v3) backend it returns only the
    ROOT-level arrays via ``array_keys()`` — a per-layer array written under ``<group>/layer=...``
    lives in a CHILD GROUP and is invisible to a single ``array_keys()`` call (it shows up under
    ``group_keys()`` instead). The live ``block_trajectory`` sidesteps this by enumerating layer
    names from the LIVE MODEL; the post-hoc pass has no model, so it must walk the group tree
    itself. For the ``npy`` fallback backend ``keys()`` already lists the flat logical keys (the
    JSON index), so we use it directly there. Never raises (returns ``[]`` on any failure)."""
    backend = getattr(array, "backend", None)
    if backend == "zarr":
        root = getattr(array, "_root", None)
        if root is None:
            return []
        out: List[str] = []

        def _walk(group, prefix: str) -> None:
            try:
                for k in group.array_keys():
                    out.append(prefix + k)
            except Exception:
                pass
            try:
                child_keys = list(group.group_keys())
            except Exception:
                child_keys = []
            for ck in child_keys:
                try:
                    _walk(group[ck], prefix + ck + "/")
                except Exception:
                    continue

        try:
            _walk(root, "")
        except Exception as e:
            _warn(f"zarr tree walk failed: {e!r}")
        return out
    # npy fallback: keys() already returns the flat logical keys from the JSON index.
    try:
        return list(array.keys())
    except Exception as e:
        _warn(f"array.keys() failed: {e!r}")
        return []


def _group_layer_keys(array: ArrayStore, group: str) -> Dict[Optional[str], None]:
    """Discover the ``layer`` values present under a logical array ``group``.

    ``ArrayStore.keys()`` lists logical keys as ``"<group>"`` (global, layer=None) and
    ``"<group>/layer=<name>"`` (per-layer). We parse those to enumerate which ``(group, layer)``
    slices exist so the caller can read each one back. Returns an ordered dict mapping
    ``layer -> None`` (``None`` key = the global/root array if present). Robust to either backend
    (zarr lists nested array_keys; the npy fallback lists the same logical keys). Never raises.
    """
    out: "Dict[Optional[str], None]" = {}
    all_keys = _all_logical_keys(array)
    prefix = group + "/layer="
    for k in all_keys:
        if k == group:
            out.setdefault(None, None)
        elif k.startswith(prefix):
            # Strip a trailing /head=... if present (we only need the layer here).
            rest = k[len(prefix):]
            layer = rest.split("/", 1)[0]
            out.setdefault(layer, None)
    return out


def _read_group_layer(array: ArrayStore, group: str, layer: Optional[str]
                      ) -> Tuple[Optional[np.ndarray], List[int]]:
    """Read back ``(stacked[n_steps, *shape], steps)`` for ``(group, layer)`` sorted by step.
    Returns ``(None, [])`` on any failure / absence. Never raises."""
    try:
        stacked, steps = array.read(group, layer=layer)
    except Exception:
        return None, []
    if stacked is None or np.asarray(stacked).size == 0 or not steps:
        return None, []
    stacked = np.asarray(stacked)
    steps_arr = np.asarray(steps)
    order = np.argsort(steps_arr)
    return stacked[order], [int(steps_arr[i]) for i in order]


def _first_present_group(array: ArrayStore, candidates: Tuple[str, ...]) -> Optional[str]:
    """Return the first group in ``candidates`` that has ANY (global or per-layer) array in the
    store, or None if none are present. Lets us accept whatever name the live block landed on."""
    all_keys = _all_logical_keys(array)
    keyset = set(all_keys)
    for g in candidates:
        if g in keyset or any(k.startswith(g + "/layer=") for k in all_keys):
            return g
    return None


# =============================================================================
# §6.1 ROTATION — principal angles between stored QL/QR bases over steps
# =============================================================================

def _principal_angle_summary(U: Optional[np.ndarray], V: Optional[np.ndarray]
                             ) -> Tuple[float, Optional[np.ndarray]]:
    """Mean cosine of the principal angles between the column spaces of ``U`` and ``V`` (1 =
    perfectly aligned / unrotated, 0 = orthogonal) plus the full angle vector (radians).

    Reuses ``curvature.lanczos.principal_angles`` (QR-orthonormalizes internally). ``U`` / ``V``
    are 2-D ``[m, k]`` eigvec matrices (columns = leading eigenvectors). Returns ``(nan, None)``
    when a basis is missing or the row dimensions are incompatible. Never raises."""
    if U is None or V is None:
        return float("nan"), None
    Ua = np.asarray(U)
    Va = np.asarray(V)
    if Ua.ndim != 2 or Va.ndim != 2:
        return float("nan"), None
    if Ua.shape[0] != Va.shape[0]:
        return float("nan"), None
    if Ua.shape[1] == 0 or Va.shape[1] == 0:
        return float("nan"), None
    try:
        import torch
        from curvature.lanczos import principal_angles
    except Exception as e:
        _warn(f"principal_angles unavailable ({e!r}); rotation skipped.")
        return float("nan"), None
    try:
        Ut = torch.as_tensor(Ua, dtype=torch.float32)
        Vt = torch.as_tensor(Va, dtype=torch.float32)
        ang = principal_angles(Ut, Vt)  # ascending radians, length min(k1,k2)
    except Exception:
        return float("nan"), None
    ang_np = ang.detach().to("cpu").numpy().astype(np.float64)
    if ang_np.size == 0:
        return float("nan"), None
    return float(np.mean(np.cos(ang_np))), ang_np


def _slice_basis(stacked: np.ndarray, i: int) -> Optional[np.ndarray]:
    """Extract the ``i``-th step-slice of a stacked eigvec array and coerce to a 2-D ``[m, k]``
    matrix. A stored basis slice is itself 2-D (``[m, k]``); ``stacked`` is ``[n_steps, m, k]``.
    A 1-D slice (a single eigenvector) is promoted to a column. Returns None on a bad shape."""
    try:
        sl = np.asarray(stacked[i])
    except Exception:
        return None
    if sl.ndim == 1:
        return sl.reshape(-1, 1)
    if sl.ndim == 2:
        return sl
    return None


def _rotation_for_side(scalar: ScalarStore, array: ArrayStore, *,
                       condition: str, run_id: str, group: str,
                       tag: str, write_arrays: bool,
                       existing_array_steps: Dict[Tuple[str, Optional[str]], set]
                       ) -> Dict[str, int]:
    """Compute the §6.1 rotation diagnostics for ONE side (``tag`` in {"ql","qr"}) from the
    stored eigvec ``group`` (per layer, over steps). Writes scalars
    ``<tag>_rotation_vs_prev`` / ``<tag>_rotation_vs_init`` (per layer per step) and the full
    angle vectors to the array store (``<tag>_principal_angles_vs_prev`` /
    ``..._vs_init``). Returns a small count summary. Never raises."""
    counts = {"layers": 0, "pairs": 0, "vs_init": 0}
    layers = _group_layer_keys(array, group)
    if not layers:
        return counts
    for layer in layers:
        stacked, steps = _read_group_layer(array, group, layer)
        if stacked is None or len(steps) == 0:
            continue
        counts["layers"] += 1
        # The earliest step is the init / ancestral reference for "vs_init".
        init_basis = _slice_basis(stacked, 0)
        init_step = steps[0]
        prev_basis = None
        prev_step = None
        for i, step in enumerate(steps):
            cur = _slice_basis(stacked, i)
            if cur is None:
                prev_basis, prev_step = None, None
                continue
            # vs previous step (consecutive pair)
            if prev_basis is not None and prev_step is not None:
                mc, ang = _principal_angle_summary(cur, prev_basis)
                if np.isfinite(mc):
                    scalar.add(condition, run_id, int(step), 0.0,
                               quantity=f"{tag}_rotation_vs_prev", value=mc, layer=layer)
                    counts["pairs"] += 1
                if write_arrays and ang is not None:
                    _put_array_once(array, f"{tag}_principal_angles_vs_prev", int(step),
                                    ang.astype(np.float32), layer, existing_array_steps)
            # vs init (earliest available step); skip the trivial self-comparison at init_step
            if init_basis is not None and step != init_step:
                mc0, ang0 = _principal_angle_summary(cur, init_basis)
                if np.isfinite(mc0):
                    scalar.add(condition, run_id, int(step), 0.0,
                               quantity=f"{tag}_rotation_vs_init", value=mc0, layer=layer)
                    counts["vs_init"] += 1
                if write_arrays and ang0 is not None:
                    _put_array_once(array, f"{tag}_principal_angles_vs_init", int(step),
                                    ang0.astype(np.float32), layer, existing_array_steps)
            prev_basis, prev_step = cur, step
    return counts


def compute_rotation(scalar: ScalarStore, array: ArrayStore, *,
                     condition: str, run_id: str, write_arrays: bool = True
                     ) -> Dict[str, Any]:
    """§6.1: principal-angle rotation of the stored QL/QR eigvec bases vs previous-step and vs
    init, per layer per step. Returns a summary dict. Never raises."""
    summary: Dict[str, Any] = {"section": "6.1_rotation"}
    ql_group = _first_present_group(array, _QL_EIGVEC_GROUPS)
    qr_group = _first_present_group(array, _QR_EIGVEC_GROUPS)
    summary["ql_group"] = ql_group
    summary["qr_group"] = qr_group
    if ql_group is None and qr_group is None:
        _warn("no stored QL/QR eigvec arrays found (groups tried: "
              f"{_QL_EIGVEC_GROUPS} / {_QR_EIGVEC_GROUPS}); §6.1 rotation skipped "
              "(expected for the adamw variant, which has no Kronecker basis).")
        summary["skipped"] = True
        return summary

    existing: Dict[Tuple[str, Optional[str]], set] = {}
    if ql_group is not None:
        try:
            c = _rotation_for_side(scalar, array, condition=condition, run_id=run_id,
                                   group=ql_group, tag="ql", write_arrays=write_arrays,
                                   existing_array_steps=existing)
            summary["ql"] = c
        except Exception as e:  # defensive: never let one side kill the pass
            _warn(f"§6.1 QL rotation failed: {e!r}")
            summary["ql_error"] = repr(e)
    if qr_group is not None:
        try:
            c = _rotation_for_side(scalar, array, condition=condition, run_id=run_id,
                                   group=qr_group, tag="qr", write_arrays=write_arrays,
                                   existing_array_steps=existing)
            summary["qr"] = c
        except Exception as e:
            _warn(f"§6.1 QR rotation failed: {e!r}")
            summary["qr_error"] = repr(e)
    return summary


# =============================================================================
# §6.3 TRAJECTORY — straightness / MSD(tau) / velocity autocorrelation
# =============================================================================

def _stack_proj_coords(stacked: np.ndarray) -> Optional[np.ndarray]:
    """Coerce a stacked proj-coords array ``[n_steps, ...]`` to a 2-D ``[n_points, d]`` float64
    trajectory (one row per step). Each per-step slice is a 1-D ``R^d`` vector; a stray extra
    axis is flattened. Returns None if it cannot be made 2-D."""
    arr = np.asarray(stacked, dtype=np.float64)
    if arr.ndim == 1:
        # single step, d coords -> [1, d]
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return None


def _read_proj_trajectory_from_arrays(array: ArrayStore, layer: Optional[str]
                                      ) -> Tuple[Optional[np.ndarray], List[int]]:
    """Read the accumulated projected-coordinate trajectory ``(traj[n_points, d], steps)`` for the
    GLOBAL (layer=None) run from the ``proj_coords`` group, or per-layer from
    ``proj_coords_layer``. Sorted by step. Returns ``(None, [])`` if absent. Never raises."""
    group = _PROJ_COORDS_GROUP if layer is None else _PROJ_COORDS_LAYER_GROUP
    stacked, steps = _read_group_layer(array, group, layer)
    if stacked is None:
        return None, []
    traj = _stack_proj_coords(stacked)
    if traj is None or traj.shape[0] == 0:
        return None, []
    return traj, steps


def _msd_curve_uneven(traj: np.ndarray, steps: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """``MSD(tau) = mean over pairs ||x_j - x_i||^2`` where ``tau = step_j - step_i`` is the TRUE
    step gap (handles non-uniform sampling). Returns ``(taus, msd)`` for the distinct positive
    gaps present in the trajectory, sorted by tau. Empty if < 2 points.

    Non-uniform spacing is handled by binning all O(n^2) ordered pairs by their exact step gap and
    averaging the squared displacement within each gap. This is the natural generalization of the
    uniform-grid ``mean_t ||x_{t+tau} - x_t||^2`` to the recorded (irregular) step indices."""
    n = traj.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    st = np.asarray(steps, dtype=np.int64)
    sums: Dict[int, float] = {}
    cnts: Dict[int, int] = {}
    for i in range(n):
        di = traj[i:] - traj[i]                       # [n-i, d], rows j>=i
        sq = np.sum(di * di, axis=1)                  # [n-i]
        gaps = st[i:] - st[i]                         # [n-i]
        for j in range(1, gaps.shape[0]):             # j=0 is the zero-gap self pair; skip
            g = int(gaps[j])
            if g <= 0:
                continue
            sums[g] = sums.get(g, 0.0) + float(sq[j])
            cnts[g] = cnts.get(g, 0) + 1
    if not sums:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    taus = np.array(sorted(sums.keys()), dtype=np.float64)
    msd = np.array([sums[int(t)] / cnts[int(t)] for t in taus], dtype=np.float64)
    return taus, msd


def _msd_slope_and_plateau(taus: np.ndarray, msd: np.ndarray) -> Tuple[float, bool]:
    """Log-log slope of ``MSD(tau)`` and a crude plateau flag (mirrors
    block_trajectory._msd_slope_and_plateau). Slope ~2 ballistic / ~1 diffusive / ~0 confined.
    Plateau when the upper-half lags barely grow relative to the max and slope < 0.5. Returns
    ``(nan, False)`` when too few valid points."""
    valid = (taus > 0) & np.isfinite(msd) & (msd > _EPS)
    t = taus[valid]
    m = msd[valid]
    if t.size < _MSD_MIN_POINTS:
        return float("nan"), False
    lt = np.log(t)
    lm = np.log(m)
    try:
        slope = float(np.polyfit(lt, lm, 1)[0])
    except Exception:
        slope = float("nan")
    half = max(1, t.size // 2)
    upper = m[-half:]
    mmax = float(np.max(m))
    plateau = bool(mmax > 0 and (float(np.mean(upper)) >= 0.95 * mmax) and slope < 0.5)
    return slope, plateau


def _straightness(traj: np.ndarray, path_length: float) -> Tuple[float, str]:
    """Straightness = net displacement / path length over the projected trajectory.

    ``net_displacement = ||x_last - x_first||`` in projected coordinates (the JL embedding
    approximately preserves this L2 distance). Preferred denominator is the trainer's inline
    ``path_length`` (the true sum of per-step velocity norms, walked in full param space — also
    norm-preserved by the projection in expectation). Falls back to the proj-space path length
    ``sum_t ||x_{t+1} - x_t||`` when the inline path length is unavailable. ~1 = directed
    (ballistic), ~0 = churning. Returns ``(value, basis)``."""
    if traj.shape[0] < 2:
        return float("nan"), "none"
    net = float(np.linalg.norm(traj[-1] - traj[0]))
    if np.isfinite(path_length) and path_length > _EPS:
        return float(net / path_length), "inline_path_length"
    # Proj-space path length fallback.
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    pl = float(np.sum(seg))
    if pl > _EPS:
        return float(net / pl), "proj_path_length"
    return float("nan"), "none"


def _put_array_once(array: ArrayStore, group: str, step: int, arr: np.ndarray,
                    layer: Optional[str],
                    existing: Dict[Tuple[str, Optional[str]], set]) -> None:
    """Append ``arr`` as the step-slice for ``(group, layer, step)`` ONLY if that step is not
    already stored (idempotency: a re-run does not double-append). ``existing`` caches the set of
    already-present steps per (group, layer) so the on-disk steps are read once. Never raises."""
    cache_key = (group, layer)
    if cache_key not in existing:
        _, steps = _read_group_layer(array, group, layer)
        existing[cache_key] = set(int(s) for s in steps)
    if int(step) in existing[cache_key]:
        return
    try:
        array.put(group=group, step=int(step), array=arr, layer=layer)
        existing[cache_key].add(int(step))
    except Exception as e:
        _warn(f"array.put({group!r}, step={step}, layer={layer!r}) failed: {e!r}")


def _inline_path_length(inline_df, step_cap: Optional[int] = None) -> float:
    """Latest inline ``path_length`` value (<= ``step_cap`` if given), or NaN. Never raises."""
    if inline_df is None:
        return float("nan")
    try:
        pl = inline_df[inline_df["quantity"] == _PATH_LENGTH_Q]
        if step_cap is not None:
            pl = pl[pl["step"] <= step_cap]
        if len(pl) == 0:
            return float("nan")
        return float(pl.sort_values("step")["value"].iloc[-1])
    except Exception:
        return float("nan")


def _velocity_autocorr_final(inline_df) -> Dict[int, float]:
    """Final (latest-step) inline ``velocity_autocorr`` per lag (lag = the ``head`` column).
    Returns ``{lag: value}``. Empty if unavailable. Never raises."""
    out: Dict[int, float] = {}
    if inline_df is None:
        return out
    try:
        ac = inline_df[inline_df["quantity"] == _VELOCITY_AUTOCORR_Q]
        if len(ac) == 0:
            return out
        ac = ac.sort_values("step")
        for lag, grp in ac.groupby("head"):
            try:
                lag_i = int(lag)
            except (TypeError, ValueError):
                continue
            val = float(grp["value"].iloc[-1])
            if np.isfinite(val):
                out[lag_i] = val
    except Exception as e:
        _warn(f"velocity_autocorr read skipped: {e!r}")
    return out


def compute_trajectory(scalar: ScalarStore, array: ArrayStore, inline_df, *,
                       condition: str, run_id: str, write_arrays: bool = True,
                       per_layer: bool = True) -> Dict[str, Any]:
    """§6.3: straightness, MSD(tau)+slope+plateau, velocity autocorrelation over the run's
    accumulated projected-coordinate trajectory + inline log. Writes run-level scalars (keyed at
    the LAST trajectory step) + the MSD(tau) curve array. Returns a summary. Never raises."""
    summary: Dict[str, Any] = {"section": "6.3_trajectory"}
    existing: Dict[Tuple[str, Optional[str]], set] = {}

    traj, steps = _read_proj_trajectory_from_arrays(array, layer=None)
    if traj is None or traj.shape[0] < 2:
        _warn(f"global proj_coords trajectory has < 2 points "
              f"(n={0 if traj is None else traj.shape[0]}); §6.3 global skipped.")
        summary["global_skipped"] = True
    else:
        last_step = int(steps[-1])
        summary["n_points"] = int(traj.shape[0])
        summary["step_span"] = [int(steps[0]), last_step]

        # --- straightness ---
        pl = _inline_path_length(inline_df, step_cap=last_step)
        straight, basis = _straightness(traj, pl)
        if np.isfinite(straight):
            scalar.add(condition, run_id, last_step, 0.0,
                       quantity="straightness", value=straight)
        summary["straightness"] = straight
        summary["straightness_basis"] = basis

        # --- MSD(tau) (true step gaps; non-uniform spacing handled) ---
        taus, msd = _msd_curve_uneven(traj, steps)
        if msd.size:
            if write_arrays:
                _put_array_once(array, "msd_curve", last_step,
                                np.stack([taus, msd], axis=-1).astype(np.float32),
                                None, existing)
            slope, plateau = _msd_slope_and_plateau(taus, msd)
            if np.isfinite(slope):
                scalar.add(condition, run_id, last_step, 0.0,
                           quantity="msd_slope", value=slope)
            scalar.add(condition, run_id, last_step, 0.0,
                       quantity="msd_plateau", value=1.0 if plateau else 0.0)
            summary["msd_slope"] = slope
            summary["msd_plateau"] = bool(plateau)
            summary["msd_n_lags"] = int(taus.size)
        else:
            summary["msd_slope"] = float("nan")

        # --- velocity autocorrelation (inline; final per lag) ---
        ac_final = _velocity_autocorr_final(inline_df)
        for lag, val in sorted(ac_final.items()):
            scalar.add(condition, run_id, last_step, 0.0,
                       quantity="velocity_autocorr_final", value=val, head=lag)
        summary["n_velocity_autocorr_lags"] = len(ac_final)

    # --- per-layer MSD slope (where the churn lives) ---
    if per_layer:
        n_layer = 0
        layers = _group_layer_keys(array, _PROJ_COORDS_LAYER_GROUP)
        for layer in layers:
            if layer is None:
                continue
            ltraj, lsteps = _read_proj_trajectory_from_arrays(array, layer=layer)
            if ltraj is None or ltraj.shape[0] < _MSD_MIN_POINTS:
                continue
            ltaus, lmsd = _msd_curve_uneven(ltraj, lsteps)
            if not lmsd.size:
                continue
            lslope, _lplat = _msd_slope_and_plateau(ltaus, lmsd)
            if np.isfinite(lslope):
                scalar.add(condition, run_id, int(lsteps[-1]), 0.0,
                           quantity="msd_slope", value=lslope, layer=layer)
                n_layer += 1
        summary["n_layer_msd"] = n_layer

    return summary


# =============================================================================
# top-level run
# =============================================================================

def _resolve_condition_run_id(out_store_run_dir: str, run_id: Optional[str]
                              ) -> Tuple[str, str]:
    """Resolve ``(condition, run_id)`` for the rows we write. ``run_id`` is taken from the CLI if
    given, else from a manifest near the store dir, else the store dir basename. ``condition`` is
    parsed from the run_id (``<condition>__<variant>__s<seed>``) or read from the manifest, else
    "unknown". The tidy-long rows MUST carry the same identity the live blocks used."""
    rid = run_id
    condition = None
    # manifest may sit next to the store, or near the run's output dir; try a couple of places.
    for cand in (os.path.join(out_store_run_dir, "manifest.json"),
                 os.path.join(os.path.dirname(out_store_run_dir.rstrip("/")), "manifest.json")):
        if os.path.exists(cand):
            try:
                man = read_manifest(cand)
                rid = rid or man.get("run_id")
                condition = man.get("condition")
                break
            except Exception:
                pass
    if not rid:
        rid = os.path.basename(out_store_run_dir.rstrip("/"))
    if not condition:
        # run_id convention: <condition>__<variant>__s<seed>
        condition = rid.split("__", 1)[0] if "__" in rid else "unknown"
    return condition, rid


def run_posthoc(out_store_run_dir: str, *, inline_scalars: Optional[str] = None,
                run_id: Optional[str] = None, write_arrays: bool = True,
                per_layer: bool = True) -> Dict[str, Any]:
    """Run the full post-hoc sequential pass on ONE run's per-run store.

    Args:
        out_store_run_dir: the per-run store directory (``<out-store>/<run_id>``) holding
            ``arrays.zarr`` + ``scalars.parquet`` (the dataset dir ``scalars_parts/``).
        inline_scalars: the trainer's inline log (``<run output_dir>/metrics/scalars.parquet``)
            for path_length / velocity_autocorr. Optional; if absent straightness falls back to
            the proj-space path length and velocity_autocorr is skipped.
        run_id: override the run_id stamped on the rows (else resolved from manifest / dir name).
        write_arrays: also write the full angle / MSD-curve arrays (not just the scalars).
        per_layer: also compute per-layer MSD slopes from ``proj_coords_layer``.

    Returns a summary dict ``{run_id, condition, rotation:{...}, trajectory:{...}, ...}``. Never
    raises out of the top level (each section is independently guarded)."""
    t0 = time.time()
    out_store_run_dir = os.path.abspath(out_store_run_dir)
    condition, rid = _resolve_condition_run_id(out_store_run_dir, run_id)
    _log(f"run_id={rid} condition={condition} store={out_store_run_dir}")

    arrays_path = os.path.join(out_store_run_dir, "arrays.zarr")
    scalars_path = os.path.join(out_store_run_dir, "scalars.parquet")

    # Open stores. The ArrayStore opens in append mode ("a"); the ScalarStore appends a new part.
    try:
        array = ArrayStore(arrays_path)
    except Exception as e:
        _warn(f"could not open array store {arrays_path!r}: {e!r}; aborting.")
        return {"run_id": rid, "condition": condition, "error": f"array_store: {e!r}"}
    try:
        scalar = ScalarStore(scalars_path)
    except Exception as e:
        _warn(f"could not open scalar store {scalars_path!r}: {e!r}; aborting.")
        return {"run_id": rid, "condition": condition, "error": f"scalar_store: {e!r}"}

    # Inline trainer log (path_length / velocity_autocorr). Optional.
    inline_df = None
    if inline_scalars:
        try:
            inline_df = read_scalars(inline_scalars)
            if inline_df is not None and len(inline_df) and "run_id" in inline_df:
                sub = inline_df[inline_df["run_id"] == rid]
                inline_df = sub if len(sub) > 0 else inline_df
            _log(f"inline log loaded: {0 if inline_df is None else len(inline_df)} rows "
                 f"from {inline_scalars}")
        except Exception as e:
            _warn(f"could not read inline scalars {inline_scalars!r}: {e!r}; "
                  "straightness uses proj-space path length, velocity_autocorr skipped.")
            inline_df = None
    else:
        _warn("no --inline-scalars given; straightness uses proj-space path length, "
              "velocity_autocorr skipped.")

    summary: Dict[str, Any] = {"run_id": rid, "condition": condition}

    # §6.1 rotation
    try:
        summary["rotation"] = compute_rotation(
            scalar, array, condition=condition, run_id=rid, write_arrays=write_arrays)
    except Exception as e:
        _warn(f"§6.1 rotation pass failed entirely: {e!r}")
        summary["rotation"] = {"error": repr(e)}

    # §6.3 trajectory
    try:
        summary["trajectory"] = compute_trajectory(
            scalar, array, inline_df, condition=condition, run_id=rid,
            write_arrays=write_arrays, per_layer=per_layer)
    except Exception as e:
        _warn(f"§6.3 trajectory pass failed entirely: {e!r}")
        summary["trajectory"] = {"error": repr(e)}

    # Flush the new scalar part durably.
    try:
        part = scalar.flush()
        summary["scalar_part"] = part
    except Exception as e:
        _warn(f"scalar flush failed: {e!r}")
        summary["scalar_part_error"] = repr(e)

    summary["t_seconds"] = round(time.time() - t0, 3)
    _log(f"DONE run_id={rid} in {summary['t_seconds']}s "
         f"rotation={summary.get('rotation', {}).get('ql', summary.get('rotation'))} "
         f"trajectory_straightness={summary.get('trajectory', {}).get('straightness')}")
    return summary


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="POST-HOC SEQUENTIAL cross-checkpoint diagnostics (REDESIGN CONTRACT (D)): "
                    "§6.1 QL/QR rotation + §6.3 trajectory, run ONCE per run after the trainer + "
                    "collectors finish. Reads the per-run store + the trainer inline log; writes "
                    "the cross-checkpoint diagnostics back into the same store.")
    p.add_argument("--out-store", required=True,
                   help="the per-run store directory <out-store>/<run_id> (holds arrays.zarr + "
                        "scalars.parquet / scalars_parts/). The cross-checkpoint diagnostics are "
                        "written back here.")
    p.add_argument("--inline-scalars", default=None,
                   help="the trainer's inline log <run output_dir>/metrics/scalars.parquet "
                        "(path_length / velocity / velocity_autocorr per step). Optional; "
                        "straightness falls back to the proj-space path length without it.")
    p.add_argument("--run-id", default=None,
                   help="override the run_id stamped on the written rows (else resolved from a "
                        "manifest near the store / the store dir basename).")
    p.add_argument("--no-arrays", action="store_true",
                   help="write only the scalar diagnostics, not the full angle / MSD-curve arrays.")
    p.add_argument("--no-per-layer", action="store_true",
                   help="skip the per-layer MSD slopes (proj_coords_layer).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_posthoc(
        args.out_store,
        inline_scalars=args.inline_scalars,
        run_id=args.run_id,
        write_arrays=not args.no_arrays,
        per_layer=not args.no_per_layer,
    )
    if summary.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
