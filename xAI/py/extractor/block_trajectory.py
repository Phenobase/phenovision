#!/usr/bin/env python3
"""
block_trajectory.py — extractor block §6.3: Trajectory geometry (the *type* of movement).

Per the briefing (``xAI/notes/phenovision_full_briefing.md`` Part II §6.3). This block answers the
question a prior experiment raised: a PlantCLEF-pretrained model changed its loss at the same rate
as the ImageNet model but traveled nearly 3x as far in weight space — directed travel into a more
distant basin, or churning in place (fast but going nowhere)? Four diagnostics distinguish
directed, diffusive, confined, and oscillatory motion, computed GLOBALLY and PER-LAYER.

  * **(a) Straightness ratio** ``R = net_displacement / path_length`` over a sliding window.
    ``net_displacement`` is ``||theta(t) - theta(prev)||`` (and vs init), from ``ctx.prev_model_sd``
    / ``ctx.init_model_sd`` over the trainable params (frozen stage excluded). ``path_length`` is
    the inline ``sum ||v_t||`` the trainer accumulates every step (briefing §5); we read the run's
    inline ``path_length`` / ``velocity_norm`` history from the ScalarStore when locatable.
    ``R ~ 1`` is directed (ballistic); ``R ~ 0`` is churning. **Fallback (documented):** if the
    inline path-length history is unavailable, we fall back to a net-displacement-based proxy
    straightness ``net_disp(prev->t) / (net_disp(init->prev) + net_disp(prev->t))`` (a coarse
    same-units stand-in that is 1 for perfectly directed motion away from init and ->0 for motion
    that returns toward where it came from). Both are written as ``straightness`` with the basis
    recorded in the summary; the fallback never raises.

  * **(b) Mean-squared-displacement scaling** ``MSD(tau) = mean_t ||theta_{t+tau} - theta_t||^2``
    vs lag ``tau``, computed on the trajectory of PROJECTED coordinates accumulated so far (the
    §6.2 ``proj_coords`` arrays, read back via ``ctx.array.read('proj_coords')``). We report the
    log-log slope (``msd_slope``: ~2 ballistic/directed, ~1 diffusive, plateau = confined) and a
    crude plateau-detection flag, and store the full ``MSD(tau)`` curve to the array store. **Early
    checkpoints (few accumulated points) yield NaN / a partial curve** — documented and handled.

  * **(c) Velocity autocorrelation** ``C(tau) = mean_t (v_t . v_{t+tau}) / mean_t ||v_t||^2`` for
    small lags, accumulated INLINE by the trainer (briefing §5/§6.3c) and logged to the ScalarStore
    as ``velocity_autocorr`` with the lag in the ``head`` column. Positive over many steps =
    persistent directed motion; negative at a lag = back-and-forth oscillation; fast decay to ~0 =
    diffusive/circling. We pass these through / aggregate them (re-emit the latest per-lag value
    under ``velocity_autocorr`` so the extractor timeline carries it) when the inline history is
    locatable; otherwise this piece is skipped.

  * **(d) Spectral flat/steep split of MOTION (direct test, reuses §6.1).** Project the per-layer
    displacement-since-last-checkpoint ``theta_l(t) - theta_l(prev)`` (or, when present, the
    optimizer's rotated first moment ``exp_avg`` from ``ctx.opt['state']`` — already the smoothed
    update direction in the eigenbasis) into the init curvature basis ``ctx.Q0`` and report the
    fraction of MOTION energy in flat vs steep directions (``motion_frac_flat`` /
    ``motion_frac_steep``), plus a loss-change-per-unit-motion proxy by direction
    (``loss_per_motion_flat`` / ``loss_per_motion_steep``) using the inline train-loss change
    between checkpoints. Churning is motion piled into flat directions where moving costs no loss;
    progress is motion along loss-reducing (steep) directions.

    **HYPOTHESIS, NOT ASSERTION (briefing §6.3d / "Do not assert flat-direction churn").** Persistent
    late motion in flat directions is what the exploration / natural-gradient stationary (OU)
    regime predicts — large standing variance in flat directions while the iterate sits near the
    optimum. If PlantCLEF does that, "fast but going nowhere" is a regime signature, not a
    pathology. These diagnostics EVALUATE that hypothesis; this block records the numbers and does
    not assert the reading.

Robustness / RNG
----------------
Deterministic (no stochastic ops), so no ``torch.Generator`` is threaded (mirrors ``block_optim``).
The block degrades gracefully for the ``adamw`` variant (no ``ctx.Q0`` -> part (d) skipped, no
``exp_avg`` rotated-moment path) and is robust to ``None`` ``prev_model_sd`` / ``Q0`` and to an
empty accumulated trajectory / missing inline history — it returns a partial summary and NEVER
raises out of ``extract``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Project paths so preadapt_common imports cleanly when run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.dirname(_HERE)  # xAI/py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PY))  # repo root
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
_TWO_NOISE = "/blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise"
for _p in (_PY, _PROJECT_ROOT, _PLANTCLEF, _TWO_NOISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if TYPE_CHECKING:  # the dataclass is defined by the collector/driver; import lazily.
    from extractor._ctx import ExtractCtx  # noqa: F401


# Inline-logged quantity names the trainer writes (preadapt_train.py §5 / VELOCITY_LAGS).
_PATH_LENGTH_Q = "path_length"
_VELOCITY_NORM_Q = "velocity_norm"
_VELOCITY_AUTOCORR_Q = "velocity_autocorr"  # lag is stored in the `head` column
_TRAIN_LOSS_Q = "train_loss"

# Minimum number of accumulated trajectory points needed to estimate an MSD slope.
_MSD_MIN_POINTS = 4
_EPS = 1e-12


# =============================================================================
# locating the run's inline ScalarStore (path_length / velocity_norm / autocorr)
# =============================================================================

def _candidate_scalar_paths(ctx) -> List[str]:
    """Candidate ScalarStore locations for the run's INLINE history (§5), most-specific first.

    The trainer writes its inline stream to ``{output_dir}/metrics/scalars.parquet``
    (``preadapt_train.main``). The driver may surface that path in the manifest under any of a few
    keys; we try them, then fall back to the extractor's own store (``ctx.scalar``) which shares the
    tidy-long schema and may carry re-logged inline quantities. Every candidate is checked for
    existence by :func:`_read_inline_scalars` before use; non-existent ones are skipped silently.
    """
    cands: List[str] = []
    man = getattr(ctx, "manifest", None) or {}
    # explicit path keys a driver might record
    for key in ("train_scalar_path", "scalar_path", "scalars_path", "inline_scalar_path"):
        v = man.get(key)
        if isinstance(v, str) and v:
            cands.append(v)
    # output_dir -> metrics/scalars.parquet (the trainer's canonical layout)
    for key in ("output_dir", "run_output_dir", "train_output_dir"):
        v = man.get(key)
        if isinstance(v, str) and v:
            cands.append(os.path.join(v, "metrics", "scalars.parquet"))
    # the extractor's own store (shares the schema) as a last resort
    own = getattr(ctx, "scalar", None)
    own_dir = getattr(own, "dataset_dir", None)
    if isinstance(own_dir, str) and own_dir:
        cands.append(own_dir)
    # de-dup preserving order
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _read_inline_scalars(ctx):
    """Read the run's inline ScalarStore as a tidy-long DataFrame filtered to this ``run_id``, or
    None if no candidate path is readable. Never raises (returns None on any failure)."""
    try:
        from preadapt_common import read_scalars
    except Exception:
        return None
    for path in _candidate_scalar_paths(ctx):
        try:
            exists = (path.endswith(".parquet") and os.path.exists(
                path[: -len(".parquet")] + "_parts")) or os.path.isdir(path) or os.path.isfile(path)
            if not exists:
                continue
            df = read_scalars(path)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        try:
            sub = df[df["run_id"] == ctx.run_id]
        except Exception:
            sub = df
        if len(sub) > 0:
            return sub
    return None


# =============================================================================
# trainable-parameter displacement helpers
# =============================================================================

def _net_displacement(model, ref_sd: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """``||theta(t) - theta_ref||`` over trainable params (frozen stage excluded), returning the
    global net displacement and a per-layer dict. Returns ``(nan, {})`` if ``ref_sd`` is None."""
    from preadapt_common import trainable_named_parameters
    if ref_sd is None:
        return float("nan"), {}
    per_layer: Dict[str, float] = {}
    total_sq = 0.0
    for name, p in trainable_named_parameters(model, include_requires_grad_false=True):
        if name not in ref_sd:
            continue
        ref = ref_sd[name]
        if tuple(ref.shape) != tuple(p.shape):
            continue
        d = (p.detach().to("cpu", dtype=torch.float64).reshape(-1)
             - ref.detach().to("cpu", dtype=torch.float64).reshape(-1))
        nrm = float(d.norm().item())
        per_layer[name] = nrm
        total_sq += nrm * nrm
    return float(total_sq ** 0.5), per_layer


# =============================================================================
# (b) MSD(tau) from the accumulated projected-coordinate trajectory
# =============================================================================

def _read_proj_trajectory(ctx, layer: Optional[str] = None
                          ) -> Tuple[Optional[np.ndarray], List[int]]:
    """Read the accumulated §6.2 projected-coordinate trajectory for this run as
    ``(stacked [n_points, d], steps)``. The current checkpoint's coords were written by
    ``block_weights`` earlier in the same extraction, so they are included.

    Group convention mirrors ``block_weights``: the GLOBAL trajectory is the root group
    ``proj_coords`` (``layer=None``); PER-LAYER trajectories live in the distinct group
    ``proj_coords_layer`` keyed by ``layer=`` (a separate group is required because the Zarr
    backend cannot make a key both a root array and a parent of ``layer=`` children). Returns
    ``(None, [])`` if the group is absent / empty / the array store is unavailable. Never raises."""
    arr = getattr(ctx, "array", None)
    if arr is None:
        return None, []
    group = "proj_coords" if layer is None else "proj_coords_layer"
    try:
        stacked, steps = arr.read(group, layer=layer)
    except Exception:
        return None, []
    if stacked is None or np.asarray(stacked).size == 0:
        return None, []
    stacked = np.asarray(stacked, dtype=np.float64)
    if stacked.ndim == 1:
        stacked = stacked.reshape(1, -1)
    # sort by step so MSD lags are in time order regardless of append order
    order = np.argsort(np.asarray(steps))
    return stacked[order], [int(steps[i]) for i in order]


def _msd_curve(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``MSD(tau) = mean_t ||x_{t+tau} - x_t||^2`` over the trajectory ``traj`` ([n_points, d]), for
    lags ``tau = 1..n_points-1``. Returns ``(taus, msd)`` (both 1-D). Empty if < 2 points."""
    n = traj.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    taus = np.arange(1, n, dtype=np.int64)
    msd = np.empty(taus.size, dtype=np.float64)
    for i, tau in enumerate(taus):
        diff = traj[tau:] - traj[:-tau]            # [n-tau, d]
        msd[i] = float(np.mean(np.sum(diff * diff, axis=1)))
    return taus.astype(np.float64), msd


def _msd_slope_and_plateau(taus: np.ndarray, msd: np.ndarray) -> Tuple[float, bool]:
    """Log-log slope of ``MSD(tau)`` and a crude plateau flag. Slope ~2 = ballistic/directed,
    ~1 = diffusive, ~0 / saturating = confined. Plateau detected when the MSD over the upper half
    of the lag range is essentially flat (top-half mean within 5% of the overall max). Returns
    ``(nan, False)`` when too few points (early checkpoints) — documented partial result."""
    valid = (taus > 0) & np.isfinite(msd) & (msd > _EPS)
    t = taus[valid]
    m = msd[valid]
    if t.size < _MSD_MIN_POINTS:
        return float("nan"), False
    # log-log least-squares slope
    lt = np.log(t)
    lm = np.log(m)
    try:
        slope = float(np.polyfit(lt, lm, 1)[0])
    except Exception:
        slope = float("nan")
    # plateau: the upper-half lags barely grow relative to the max
    half = max(1, t.size // 2)
    upper = m[-half:]
    mmax = float(np.max(m))
    plateau = bool(mmax > 0 and (float(np.mean(upper)) >= 0.95 * mmax) and slope < 0.5)
    return slope, plateau


# =============================================================================
# (d) spectral flat/steep split of MOTION (reuses §6.1 Q0)
# =============================================================================

def _motion_flat_steep(motion_2d: torch.Tensor, q0: Dict[str, Any]
                       ) -> Optional[Tuple[float, float]]:
    """Project a per-layer 2-D MOTION matrix ``[m, n]`` (displacement-since-last-ckpt or rotated
    ``exp_avg``) onto the init Kronecker eigenbasis ``Q0 = {QL (m x m), QR (n x n)}`` and return
    ``(frac_flat, frac_steep)`` of the motion energy. Same combined-rank split as
    ``block_weights._disp_on_curvature_basis`` (eigenvectors in DESCENDING curvature order; steep
    half = top-curvature combined ranks). Returns None if no usable basis. Never raises here."""
    if motion_2d is None or motion_2d.dim() != 2:
        return None
    m, n = motion_2d.shape
    Dm = motion_2d.detach().to("cpu", dtype=torch.float64)

    QL = q0.get("QL") if isinstance(q0, dict) else None
    QR = q0.get("QR") if isinstance(q0, dict) else None
    QL = QL.detach().to("cpu", dtype=torch.float64) if isinstance(QL, torch.Tensor) else None
    QR = QR.detach().to("cpu", dtype=torch.float64) if isinstance(QR, torch.Tensor) else None

    left = QL.t() if (QL is not None and QL.dim() == 2 and QL.shape == (m, m)) else None
    right = QR if (QR is not None and QR.dim() == 2 and QR.shape == (n, n)) else None
    if left is None and right is None:
        return None

    coeff = Dm
    if left is not None:
        coeff = left @ coeff
    if right is not None:
        coeff = coeff @ right
    energy = coeff * coeff
    total = float(energy.sum().item())
    if total <= _EPS:
        return None

    ii = torch.arange(coeff.shape[0], dtype=torch.float64).reshape(-1, 1)
    jj = torch.arange(coeff.shape[1], dtype=torch.float64).reshape(1, -1)
    di = ii / max(coeff.shape[0] - 1, 1)
    dj = jj / max(coeff.shape[1] - 1, 1)
    depth = di + dj
    median_depth = float(torch.median(depth).item())
    steep_mask = (depth <= median_depth)
    frac_steep = float(energy[steep_mask].sum().item()) / total
    return 1.0 - frac_steep, frac_steep


def _layer_motion_2d(ctx, name: str, p: torch.Tensor,
                     prev_sd: Optional[Dict[str, Any]]) -> Tuple[Optional[torch.Tensor], str]:
    """Return the 2-D MOTION matrix for layer ``name`` and a tag of its source.

    Prefer the optimizer's rotated first moment ``exp_avg`` (already the smoothed update direction
    in the eigenbasis — briefing §6.3d) when present and 2-D; otherwise the
    displacement-since-last-checkpoint ``theta_l(t) - theta_l(prev)``. Returns ``(None, 'none')``
    if neither is available / shaped 2-D."""
    if p.dim() != 2:
        return None, "none"
    st = ((ctx.opt or {}).get("state", {}) or {}).get(name) if getattr(ctx, "opt", None) else None
    if isinstance(st, dict):
        ea = st.get("exp_avg")
        if isinstance(ea, torch.Tensor) and ea.dim() == 2 and ea.shape == p.shape:
            return ea, "exp_avg"
    if prev_sd is not None and name in prev_sd:
        prev = prev_sd[name]
        if tuple(prev.shape) == tuple(p.shape):
            motion = (p.detach().to("cpu", dtype=torch.float64)
                      - prev.detach().to("cpu", dtype=torch.float64))
            return motion, "disp_since_prev"
    return None, "none"


# =============================================================================
# entry point
# =============================================================================

def extract(ctx: "ExtractCtx") -> Dict[str, object]:
    """§6.3 trajectory geometry. See module docstring.

    Writes scalars: ``straightness`` (global + per-layer), ``msd_slope`` (global + per-layer),
    ``velocity_autocorr`` (passthrough per lag), ``motion_frac_flat`` / ``motion_frac_steep`` and
    ``loss_per_motion_flat`` / ``loss_per_motion_steep`` (per layer, when ``ctx.Q0`` has it).
    Writes arrays: ``msd_curve`` (global + per-layer). Returns a compact summary dict for the
    collector log. Robust to None prev/Q0 and empty history (partial result, never raises).
    """
    cond = ctx.condition
    run_id = ctx.run_id
    step = ctx.step
    wt = ctx.wall_time
    model = ctx.model
    prev_sd = getattr(ctx, "prev_model_sd", None)
    init_sd = getattr(ctx, "init_model_sd", None)

    summary: Dict[str, object] = {"block": "block_trajectory"}

    # Inline history (path_length / velocity_norm / autocorr / loss). May be None.
    inline = _read_inline_scalars(ctx)
    summary["inline_history"] = bool(inline is not None)

    # Net displacements (vs prev, vs init), global + per-layer.
    net_prev, per_layer_prev = _net_displacement(model, prev_sd)
    net_init, _ = _net_displacement(model, init_sd)

    # ---------------------------------------------------------------------
    # (a) Straightness ratio R = net_displacement / path_length.
    # ---------------------------------------------------------------------
    path_length = float("nan")
    straightness_basis = "none"
    if inline is not None:
        try:
            pl = inline[(inline["quantity"] == _PATH_LENGTH_Q) & (inline["step"] <= step)]
            if len(pl) > 0:
                path_length = float(pl.sort_values("step")["value"].iloc[-1])
        except Exception:
            path_length = float("nan")

    straightness = float("nan")
    if np.isfinite(path_length) and path_length > _EPS and np.isfinite(net_init):
        # Directed-ness of the whole run so far: net travel from init / total path walked.
        straightness = float(net_init / path_length)
        straightness_basis = "path_length"
    elif np.isfinite(net_prev) and np.isfinite(net_init):
        # Documented fallback: net-displacement-only proxy when no inline path length is available.
        denom = net_init + net_prev
        straightness = float(net_init / denom) if denom > _EPS else float("nan")
        straightness_basis = "net_disp_fallback"

    if np.isfinite(straightness):
        ctx.scalar.add(cond, run_id, step, wt, quantity="straightness", value=straightness)
    summary["straightness"] = straightness
    summary["straightness_basis"] = straightness_basis

    # Per-layer straightness fallback: per-layer net(prev)/(net(prev)+net(init_layer)) is not
    # available without per-layer path length, so we emit the per-layer net-disp-since-prev as the
    # per-layer travel signal (where the churn lives is a modularity signal — briefing §6.3 final).
    for name, nrm in per_layer_prev.items():
        ctx.scalar.add(cond, run_id, step, wt, quantity="net_disp_since_prev",
                       value=float(nrm), layer=name)

    # ---------------------------------------------------------------------
    # (b) MSD(tau) from the accumulated projected-coordinate trajectory.
    # ---------------------------------------------------------------------
    traj, steps = _read_proj_trajectory(ctx, layer=None)
    if traj is not None and traj.shape[0] >= 2:
        taus, msd = _msd_curve(traj)
        if msd.size:
            # store the MSD(tau) curve as [n_lags, 2] = (tau, msd) for post-hoc plateau analysis
            ctx.array.put(group="msd_curve", step=step,
                          array=np.stack([taus, msd], axis=-1).astype(np.float32))
            slope, plateau = _msd_slope_and_plateau(taus, msd)
            if np.isfinite(slope):
                ctx.scalar.add(cond, run_id, step, wt, quantity="msd_slope", value=slope)
            ctx.scalar.add(cond, run_id, step, wt, quantity="msd_plateau",
                           value=1.0 if plateau else 0.0)
            summary["msd_slope"] = slope
            summary["msd_plateau"] = plateau
            summary["msd_n_points"] = int(traj.shape[0])
    else:
        # Early checkpoints: too few accumulated proj_coords points for an MSD curve. Documented.
        summary["msd_slope"] = float("nan")
        summary["msd_n_points"] = int(traj.shape[0]) if traj is not None else 0

    # Per-layer MSD (where the churn lives). The per-layer proj_coords were written by
    # block_weights under the ``proj_coords_layer`` group for every trainable layer; we enumerate
    # those names from the live model's trainable layout (the same selector block_weights used)
    # rather than from ArrayStore.keys(), which on the Zarr backend lists only ROOT arrays and not
    # the nested per-layer keys. Layers with too few accumulated points (early checkpoints) skip.
    n_layer_msd = 0
    if getattr(ctx, "array", None) is not None:
        try:
            from preadapt_common import trainable_named_parameters
            layer_names = [n for n, _ in trainable_named_parameters(
                model, include_requires_grad_false=True)]
        except Exception:
            layer_names = []
        for name in layer_names:
            ltraj, _ = _read_proj_trajectory(ctx, layer=name)
            if ltraj is None or ltraj.shape[0] < _MSD_MIN_POINTS:
                continue
            ltaus, lmsd = _msd_curve(ltraj)
            if not lmsd.size:
                continue
            lslope, _lplat = _msd_slope_and_plateau(ltaus, lmsd)
            if np.isfinite(lslope):
                ctx.scalar.add(cond, run_id, step, wt, quantity="msd_slope",
                               value=lslope, layer=name)
                n_layer_msd += 1
    summary["n_layer_msd"] = n_layer_msd

    # ---------------------------------------------------------------------
    # (c) Velocity autocorrelation passthrough (inline ring buffer, §5/§6.3c).
    # ---------------------------------------------------------------------
    n_autocorr = 0
    if inline is not None:
        try:
            ac = inline[(inline["quantity"] == _VELOCITY_AUTOCORR_Q) & (inline["step"] <= step)]
            if len(ac) > 0:
                # latest value per lag (the `head` column holds the lag tau)
                ac = ac.sort_values("step")
                for lag, grp in ac.groupby("head"):
                    val = float(grp["value"].iloc[-1])
                    if np.isfinite(val):
                        try:
                            lag_i = int(lag)
                        except (TypeError, ValueError):
                            continue
                        ctx.scalar.add(cond, run_id, step, wt,
                                       quantity="velocity_autocorr", value=val, head=lag_i)
                        n_autocorr += 1
        except Exception as e:
            print(f"[block_trajectory] velocity_autocorr passthrough skipped: {e}",
                  file=sys.stderr)
    summary["n_velocity_autocorr"] = n_autocorr

    # ---------------------------------------------------------------------
    # (d) Spectral flat/steep split of MOTION (direct test, reuses §6.1 Q0).
    #     HYPOTHESIS, NOT ASSERTION (see module docstring / briefing §6.3d).
    # ---------------------------------------------------------------------
    q0 = ctx.Q0 or {}
    n_motion_split = 0
    # loss change between this checkpoint and the previous one (for loss-per-motion proxy)
    delta_loss = float("nan")
    if inline is not None:
        try:
            tl = inline[(inline["quantity"] == _TRAIN_LOSS_Q) & (inline["step"] <= step)]
            tl = tl.sort_values("step")
            if len(tl) >= 2:
                # change from the previous checkpoint's step to this step (loss DECREASE positive)
                prev_step = None
                if prev_sd is not None:
                    # best effort: use the largest logged step strictly below this one
                    below = tl[tl["step"] < step]
                    if len(below) > 0:
                        prev_step = int(below["step"].iloc[-1])
                cur_loss = float(tl[tl["step"] <= step]["value"].iloc[-1])
                if prev_step is not None:
                    prev_loss = float(tl[tl["step"] <= prev_step]["value"].iloc[-1])
                    delta_loss = prev_loss - cur_loss  # positive = loss reduced
        except Exception:
            delta_loss = float("nan")

    if q0:
        from preadapt_common import trainable_named_parameters
        for name, p in trainable_named_parameters(model, include_requires_grad_false=True):
            if name not in q0 or not isinstance(q0[name], dict):
                continue
            motion, src = _layer_motion_2d(ctx, name, p, prev_sd)
            if motion is None:
                continue
            try:
                res = _motion_flat_steep(motion, q0[name])
            except Exception:
                res = None
            if res is None:
                continue
            frac_flat, frac_steep = res
            ctx.scalar.add(cond, run_id, step, wt, quantity="motion_frac_flat",
                           value=frac_flat, layer=name)
            ctx.scalar.add(cond, run_id, step, wt, quantity="motion_frac_steep",
                           value=frac_steep, layer=name)
            # loss-change-per-unit-motion resolved by direction (proxy: split the global
            # loss change between checkpoints by the motion fraction in each direction band).
            if np.isfinite(delta_loss):
                motion_energy = float((motion * motion).sum().item()) if isinstance(
                    motion, torch.Tensor) else float(np.sum(np.asarray(motion) ** 2))
                if motion_energy > _EPS:
                    flat_motion = (frac_flat * motion_energy) ** 0.5
                    steep_motion = (frac_steep * motion_energy) ** 0.5
                    if flat_motion > _EPS:
                        ctx.scalar.add(cond, run_id, step, wt, quantity="loss_per_motion_flat",
                                       value=float((delta_loss * frac_flat) / flat_motion),
                                       layer=name)
                    if steep_motion > _EPS:
                        ctx.scalar.add(cond, run_id, step, wt, quantity="loss_per_motion_steep",
                                       value=float((delta_loss * frac_steep) / steep_motion),
                                       layer=name)
            n_motion_split += 1
            summary.setdefault("motion_source", src)

    summary["n_motion_split"] = n_motion_split
    summary["q0_available"] = bool(q0)
    summary["prev_available"] = prev_sd is not None
    summary["net_disp_vs_prev"] = net_prev
    summary["net_disp_vs_init"] = net_init
    summary["path_length"] = path_length
    # Hypothesis flag (NOT an assertion): record that flat-direction churn is to be evaluated, not
    # claimed, by downstream analysis (briefing §6.3d / global "Do not").
    summary["flat_churn_is_hypothesis"] = True
    return summary
