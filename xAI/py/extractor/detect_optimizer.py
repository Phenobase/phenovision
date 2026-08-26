#!/usr/bin/env python3
"""
extractor/detect_optimizer.py — §8 optimizer-variant detection wiring.

Thin wrapper around the §8 primitives that already live in ``preadapt_common``
(``detect_evolving_M``, ``optimizer_geometry_fields``), plus the one piece the extractor
needs that those cannot provide from a *live* optimizer: ``remap_optimizer_state`` — the
function that turns a *saved* ``optimizer.state_dict()`` (whose per-parameter ``state`` is
keyed by **integer indices** into the flattened param-group order) back into a
``{param_name: {key: tensor}}`` dict, using the ordered ``param_names`` list the trainer
writes into every checkpoint (FROZEN INTERFACE CONTRACT (A)).

Why this lives here and not in ``preadapt_common``: ``optimizer_geometry_fields`` operates on
a live ``torch.optim.Optimizer`` (it walks ``optimizer.param_groups`` and ``optimizer.state``
with real Parameter objects as keys). The extractor never reconstructs the optimizer object —
it only ever sees the serialized ``state_dict``. So the remap is a state-dict-level operation,
kept next to the build_ctx code that consumes it.

§8 reconciliation result (verified against the real ``StableEvolutionSOAP`` class, see the
plan Component C4): neither ``StableEvolutionSOAP`` nor ``AdamW`` carries an evolving-M
meta-loop (no ``M`` / ``M_target`` / ``source`` / ``precond_target`` / ``G_target`` buffer),
so ``detect_evolving_M -> False`` for both. That is the **constant-isotropic
canalization-floor** case (briefing §6.1): the extractor records the scalar
``damping``/``kappa``/``alpha_min``/``alpha_max`` from the param group and skips the
matrix-valued M block. Both the StableEvo (Kronecker QL/QR + generated ``precond`` + dynamic
``alpha_last``) and AdamW (diagonal ``exp_avg_sq`` only) paths produce well-formed records.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import torch

# Ensure the foundation module (one directory up) is importable whether this package is
# imported as ``extractor.detect_optimizer`` from ``xAI/py`` on sys.path or run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.dirname(_HERE)  # .../xAI/py
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)

# Re-export the §8 primitives unchanged so consumers import them from one obvious place.
from preadapt_common import (  # noqa: E402
    detect_evolving_M,
    optimizer_geometry_fields,
    _GEOMETRY_STATE_KEYS,
    _M_CANDIDATES,
)

__all__ = [
    "detect_evolving_M",
    "optimizer_geometry_fields",
    "remap_optimizer_state",
    "detect_variant_from_state",
    "detect_M_present_from_state",
    "GEOMETRY_STATE_KEYS",
    "M_CANDIDATE_NAMES",
]

#: Public aliases of the foundation module's tuples (avoid leading-underscore imports
#: leaking through downstream).
GEOMETRY_STATE_KEYS = _GEOMETRY_STATE_KEYS
M_CANDIDATE_NAMES = _M_CANDIDATES


def remap_optimizer_state(
    optimizer_state_dict: Dict[str, Any],
    param_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Remap a saved ``optimizer.state_dict()`` to ``{param_name: {state_key: tensor}}``.

    A torch optimizer ``state_dict`` has the shape::

        {"state": {0: {...}, 1: {...}, ...},
         "param_groups": [{"params": [0, 1, ...], "lr": ..., ...}, ...]}

    where the integer keys of ``state`` index the parameters in the order they were added to
    the optimizer (the flattened concatenation of every group's ``params`` list). The trainer
    writes the ordered ``param_names`` list into the checkpoint (CONTRACT (A)) so that index
    ``i`` corresponds to ``param_names[i]``. This function inverts that mapping.

    Robustness:
      * Only ``state`` entries whose integer index is ``< len(param_names)`` are mapped; any
        extra/unknown indices are skipped (logged to stderr) rather than raising, so a minor
        ordering drift never kills the record.
      * Parameters with no optimizer state yet (e.g. a 1-D bias that has not been stepped, or
        the AdamW first step before state init) simply do not appear in the result.
      * Tensor values are returned as-is (already CPU/GPU per how the checkpoint was loaded);
        non-tensor scalars (``step``, ``use_precond``) pass through unchanged.

    Args:
        optimizer_state_dict: the dict stored under ``checkpoint["optimizer_state_dict"]``.
        param_names: ordered trainable-parameter names matching the optimizer's param order
            (``ckpt["param_names"]``).

    Returns:
        ``{param_name: {state_key: value, ...}}`` for every parameter that has optimizer state.
    """
    if optimizer_state_dict is None:
        return {}
    state = optimizer_state_dict.get("state", {})
    out: Dict[str, Dict[str, Any]] = {}
    n_names = len(param_names)
    for idx, per_param in state.items():
        try:
            i = int(idx)
        except (TypeError, ValueError):
            print(f"[detect_optimizer] remap: non-integer state key {idx!r}; skipping.",
                  file=sys.stderr)
            continue
        if i < 0 or i >= n_names:
            print(f"[detect_optimizer] remap: state index {i} out of range "
                  f"[0,{n_names}); skipping (param_names length mismatch?).",
                  file=sys.stderr)
            continue
        name = param_names[i]
        if isinstance(per_param, dict):
            out[name] = dict(per_param)
        else:
            # Unexpected, but keep something well-formed.
            out[name] = {"_value": per_param}
    return out


def detect_M_present_from_state(
    remapped_state: Dict[str, Dict[str, Any]],
    group: Optional[Dict[str, Any]] = None,
) -> bool:
    """Run :func:`detect_evolving_M` over a *remapped* (name->state) dict.

    The foundation ``detect_evolving_M`` takes a single parameter's state plus its group; this
    convenience scans every parameter's state and returns True if ANY carries an evolving-M
    matrix-valued target buffer. For ``StableEvolutionSOAP``/``AdamW`` this is False.
    """
    grp = group if group is not None else {}
    for st in remapped_state.values():
        if isinstance(st, dict) and detect_evolving_M(st, grp):
            return True
    return False


def detect_variant_from_state(remapped_state: Dict[str, Dict[str, Any]]) -> str:
    """Classify the optimizer variant from a remapped (name->state) dict, without a live
    optimizer object.

    Mirrors :func:`optimizer_geometry_fields`'s ``variant`` logic on the saved state:
      * "stable_evo" if any param carries a Kronecker eigenbasis / generated preconditioner /
        dynamic exponent (``QL``/``QR``/``precond``/``alpha_last``),
      * "adamw" if only diagonal moments (``exp_avg_sq``) are present,
      * "unknown" otherwise.
    """
    has_kron = False
    has_diag = False
    for st in remapped_state.values():
        if not isinstance(st, dict):
            continue
        if (st.get("QL") is not None or st.get("QR") is not None
                or "precond" in st or "alpha_last" in st):
            has_kron = True
        if "exp_avg_sq" in st:
            has_diag = True
    if has_kron:
        return "stable_evo"
    if has_diag:
        return "adamw"
    return "unknown"
