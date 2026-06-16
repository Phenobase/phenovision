#!/usr/bin/env python3
"""
extractor/_ctx.py — the ExtractCtx dataclass (the FROZEN block interface, CONTRACT (B))
and ``build_ctx``, the per-checkpoint context builder.

Every §6 block module is a pure function ``extract(ctx: ExtractCtx) -> dict`` that reads from
``ctx`` and writes rows via ``ctx.scalar.add`` / arrays via ``ctx.array.put``. ``ExtractCtx``
is therefore the one shared surface across the parallel block builds; its field set is the
contract and must not drift. ``build_ctx`` is the only place that knows how a checkpoint on
disk becomes a context: it loads the ``.pt`` (already claimed/renamed by the collector),
rebuilds the model, remaps the optimizer state from integer indices back to parameter names
(using ``ckpt['param_names']``), and threads in the shared, run-invariant resources (probe
loaders, the fixed Hessian batch, the seeded random projection, and the scalar/array stores).

The shared resources are owned by the :class:`~extractor.extract.Extractor` (one per collector
process); ``build_ctx`` receives them via a ``ctx_resources`` mapping so this module stays
free of any polling / lifecycle logic.

STATELESS per-checkpoint extraction (CONTRACT (B)/(C), REDESIGN 2026-06-16): ``build_ctx`` no
longer carries any cross-checkpoint in-memory state. The only init-relative reference a block
needs (E(0), the ancestral weights, for §6.2 "vs init" / §6.6 CKA-vs-init) is loaded ONCE from a
FIXED FILE ON DISK (``<run>/kept/init_model.pt`` or the run's ``init_model.pt``) into the shared
resources and threaded READ-ONLY into every ctx as ``init_model_sd`` — it is identical for every
checkpoint of the run and depends on no previously-processed checkpoint. The previous-checkpoint
fields (``prev_model_sd``, ``prev_opt``) and the prev-derived metric basis (``Q0``) are RETAINED
on :class:`ExtractCtx` as ALWAYS-``None`` optional fields for back-compat (so blocks that still
read them via ``getattr(ctx, 'prev_model_sd', None)`` degrade cleanly to "skipped"); the
inherently cross-checkpoint diagnostics that used them (§6.1 QL/QR rotation-vs-prev/init, §6.3
trajectory MSD/straightness/velocity-autocorr/motion-split) have MOVED OUT of the live path to a
POST-HOC pass that reads the stored QL/QR eigvecs + projected coords from the per-run store. Two
collector processes can therefore run :func:`build_ctx` + the per-checkpoint blocks on DIFFERENT
checkpoints concurrently and out of order — there is no shared mutable state to race on.

Architecture facts baked in here: ViT-L/16, num_classes=2 (reproductive: flowers, fruits),
D=1024, 24 blocks, 16 heads, head_dim 64 (briefing §4 invariant 4 / plan ARCH line).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# --- project paths: mirror xai_train.py / preadapt_models.py so PlantCLEF2022 + xAI/py import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.dirname(_HERE)              # .../xAI/py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PY_DIR))  # .../phenovision
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
for _p in (_PY_DIR, _PROJECT_ROOT, _PLANTCLEF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models_vit import vit_large_patch16  # noqa: E402

from preadapt_common import (  # noqa: E402
    ScalarStore,
    ArrayStore,
    SparseRandomProjection,
    trainable_named_parameters,
)
from extractor.detect_optimizer import (  # noqa: E402
    remap_optimizer_state,
    detect_M_present_from_state,
    detect_variant_from_state,
)


# =============================================================================
# The ExtractCtx dataclass — CONTRACT (B). DO NOT change field names/shapes lightly:
# every block module depends on this exact surface.
# =============================================================================

@dataclass
class ExtractCtx:
    """Per-checkpoint extraction context handed to every §6 block's ``extract(ctx)``.

    Identity / timeline (used as the tidy-long key on every scalar row):
        condition : 'imagenet'|'mae'|'plantclef'|'naive'
        variant   : 'stable_evo'|'adamw'
        run_id    : '{condition}__{variant}__s{seed}'
        seed      : int
        step      : int   — the training step this checkpoint was emitted at
        wall_time : float — the training wall-time stamp on this checkpoint

    Model / loss:
        device      : torch.device the model + probe passes run on
        model       : nn.Module — eval-mode ViT-L/16 with this checkpoint's weights loaded
        criterion   : nn.BCEWithLogitsLoss (multi-label: flowers, fruits)

    Optimizer state (saved-state view; see CONTRACT (A)/(B)):
        opt : {
            'variant'   : 'stable_evo'|'adamw',
            'M_present' : bool,                # detect_evolving_M over this checkpoint
            'state'     : { param_name: { state_key: tensor/scalar } },
                          # state_key in {QL,QR,L,R,exp_avg,exp_avg_sq,precond,alpha_last,
                          #               step,use_precond} where present
            'group'     : { config keys: alpha_min,alpha_max,kappa,damping,
                            precondition_frequency,max_precond_dim,lr,weight_decay,
                            demographic_* ... }   # representative param-group config
        }

    Fixed init reference (loaded ONCE from disk; READ-ONLY; identical every ckpt):
        init_model_sd : dict|None — the run's INIT (E(0)) model_state_dict, loaded from the
                        fixed ``<run>/kept/init_model.pt`` / ``init_model.pt`` file (NOT from a
                        previously-processed checkpoint). Threaded in unchanged on every ckpt so
                        §6.2 "vs init" / §6.6 CKA-vs-init are per-checkpoint independent.

    DEPRECATED cross-checkpoint fields (REDESIGN 2026-06-16; ALWAYS ``None`` now):
        prev_model_sd : None — the previous-checkpoint comparison moved to the POST-HOC pass.
        prev_opt      : None — the QL/QR rotation-vs-prev moved to the POST-HOC pass.
        Q0            : None — the init metric basis is reconstructed POST-HOC from the stored
                        ``ql_eigvecs``/``qr_eigvecs`` arrays, not carried in memory.
        These remain on the dataclass so the unchanged blocks (block_weights / block_interp) that
        read them via ``getattr(ctx, ..., None)`` skip their cross-checkpoint piece cleanly.

    Fixed comparability resources (briefing §4 invariants; identical every ckpt/condition):
        probe_eval_loader  : DataLoader — fixed held-out probe-eval split (fitness, §6.8)
        probe_train_loader : DataLoader — fixed probe-train split (linear probes, §6.7)
        hessian_batch      : (images, targets) on ``device`` — the fixed curvature batch (§6.4)
        projection         : SparseRandomProjection — the shared seeded reducing map (§6.2)

    Stores + manifest:
        scalar   : ScalarStore — tidy-long Parquet sink (ctx.scalar.add(...))
        array    : ArrayStore  — Zarr array sink (ctx.array.put(...))
        manifest : dict        — the run manifest (arch, optimizer config, projection meta, ...)

    Convenience cache: blocks that need per-layer features (e.g. §6.7 probes) can stash a
    one-time forward result in ``cache`` so two blocks don't re-run the same forward pass on
    the same probe batch. It is reset per checkpoint.
    """

    # --- identity / timeline -------------------------------------------------------------
    condition: str
    variant: str
    run_id: str
    seed: int
    step: int
    wall_time: float

    # --- model / loss --------------------------------------------------------------------
    device: torch.device
    model: nn.Module
    criterion: nn.Module

    # --- optimizer state view ------------------------------------------------------------
    opt: Dict[str, Any]

    # --- fixed init reference (loaded once from disk; read-only) -------------------------
    init_model_sd: Optional[Dict[str, Any]] = None
    # --- DEPRECATED cross-checkpoint fields (REDESIGN 2026-06-16; always None) ------------
    # Kept as optional fields for back-compat so the unchanged blocks that read them skip
    # cleanly. The diagnostics that used them are now computed POST-HOC (CONTRACT (C)/(D)).
    prev_model_sd: Optional[Dict[str, Any]] = None
    prev_opt: Optional[Dict[str, Any]] = None
    Q0: Optional[Dict[str, Any]] = None

    # --- fixed comparability resources ---------------------------------------------------
    probe_eval_loader: Any = None
    probe_train_loader: Any = None
    hessian_batch: Any = None
    projection: Optional[SparseRandomProjection] = None

    # --- inline §6.2 global projection coords (collector-speed optimization) -------------
    # When the trainer computed the global projected coordinates P^T theta(t) in R^d at
    # checkpoint EMISSION and stored them in the checkpoint (key ``proj_coords``, with
    # ``proj_meta`` = projection.to_meta()), build_ctx surfaces them here so block_weights can
    # use them DIRECTLY instead of re-projecting the ~302M-coord trainable vector. Both default
    # None (backward-compatible: old checkpoints without them recompute as before).
    inline_proj_coords: Optional[torch.Tensor] = None   # R^d global coords, or None
    inline_proj_meta: Optional[Dict[str, Any]] = None   # {method,seed,d,D_trainable,density}, or None

    # --- stores + manifest ---------------------------------------------------------------
    scalar: Optional[ScalarStore] = None
    array: Optional[ArrayStore] = None
    manifest: Dict[str, Any] = field(default_factory=dict)

    # --- per-checkpoint scratch cache (reset each ckpt) ----------------------------------
    cache: Dict[str, Any] = field(default_factory=dict)

    # --- small convenience accessors blocks may use --------------------------------------
    def add(self, quantity: str, value: float, layer: Optional[str] = None,
            head: Optional[int] = None) -> None:
        """Shorthand for ``ctx.scalar.add(condition, run_id, step, wall_time, quantity, value,
        layer, head)`` — the identity columns are filled from this context. Blocks may use this
        or call ``ctx.scalar.add(...)`` directly; both write identical rows."""
        if self.scalar is None:
            return
        self.scalar.add(self.condition, self.run_id, self.step, self.wall_time,
                        quantity=quantity, value=float(value), layer=layer, head=head)

    def put(self, group: str, array: Any, layer: Optional[str] = None,
            head: Optional[int] = None) -> None:
        """Shorthand for ``ctx.array.put(group, step, array, layer, head)``."""
        if self.array is None:
            return
        self.array.put(group=group, step=self.step, array=array, layer=layer, head=head)


# =============================================================================
# build_ctx — checkpoint on disk -> ExtractCtx
# =============================================================================

#: Param-group config keys we surface into ``ctx.opt['group']`` (the canalization-floor scalars
#: plus the standard SOAP/AdamW knobs). Anything else present in the group is also copied through
#: verbatim, but these are the ones blocks rely on.
_GROUP_KEYS_OF_INTEREST = (
    "lr", "weight_decay", "betas", "eps",
    "alpha_min", "alpha_max", "kappa", "damping",
    "precondition_frequency", "max_precond_dim", "precondition_1d",
    "demographic_noise", "demographic_temperature", "demographic_warmup",
    "demographic_match_grad", "demographic_kappa", "demographic_batch",
)


def _reconstruct_opt(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Build ``ctx.opt`` from a loaded checkpoint dict.

    Uses ``ckpt['param_names']`` (CONTRACT (A)) to remap the saved
    ``optimizer_state_dict['state']`` integer indices back to parameter names, detects the
    variant + M-present boolean from the remapped state, and pulls a representative param-group
    config. Robust to AdamW (no QL/QR/precond/alpha_last — only ``exp_avg``/``exp_avg_sq``) and
    to a missing optimizer state (returns an empty but well-formed ``opt``).
    """
    osd = ckpt.get("optimizer_state_dict")
    param_names: List[str] = list(ckpt.get("param_names", []) or [])

    if osd is None:
        # No optimizer state in this checkpoint — record an explicit empty view so blocks can
        # branch on it cleanly rather than crashing. (Briefing escalation: the trainer MUST save
        # optimizer state; this guard keeps the extractor alive if a stray ckpt lacks it.)
        return {"variant": "unknown", "M_present": False, "state": {}, "group": {}}

    remapped = remap_optimizer_state(osd, param_names)

    # Representative group config: the first param_group, narrowed to keys of interest plus any
    # extra keys present (so demographic_* / shrinkage knobs are not silently dropped).
    group_cfg: Dict[str, Any] = {}
    groups = osd.get("param_groups", [])
    if groups:
        g0 = groups[0]
        for k, v in g0.items():
            if k == "params":
                continue
            group_cfg[k] = v
        # Guarantee the keys of interest exist (None if absent) so blocks can index them.
        for k in _GROUP_KEYS_OF_INTEREST:
            group_cfg.setdefault(k, None)

    # Prefer the explicit variant recorded by the trainer if present; else infer from state.
    variant = ckpt.get("variant") or detect_variant_from_state(remapped)
    m_present = detect_M_present_from_state(remapped, group_cfg)

    return {
        "variant": variant,
        "M_present": bool(m_present),
        "state": remapped,
        "group": group_cfg,
    }


#: Optimizer per-param state keys retained in the CARRIED cross-checkpoint views (``prev_opt``
#: and ``Q0``). §6.1 rotation diagnostics (block_optim's ``ql_rotation_vs_prev/vs_init`` and
#: ``qr_rotation_vs_*``) only ever read the eigenbases QL/QR from the previous / init views — NOT
#: the large Kronecker accumulators (``L``/``R``), the second moments (``exp_avg``/``exp_avg_sq``),
#: or the generated preconditioner (``precond``). Retaining only QL/QR drops the bulk of a
#: full-Kronecker ViT-L optimizer state (~2 full copies of a 15 GB checkpoint) from host RAM.
#: NOTE: QL/QR for an [m×m]/[n×n] factor are themselves [m×m]/[n×n], so they are NOT free — the
#: ``--max_precond_dim`` cap on the trainer side (default 2048) is what keeps them small.
_BASIS_STATE_KEYS = ("QL", "QR")


def _cpu_detach_clone(t: Any) -> Any:
    """Detach + move-to-CPU + clone a tensor so the retained copy holds NO reference to the
    just-loaded checkpoint's storage (clone breaks the view/slice aliasing that would otherwise
    pin the whole loaded blob in RAM). Non-tensors pass through unchanged."""
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu").clone()
    return t


def _trim_opt_to_bases(opt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a SAME-STRUCTURED ``opt`` view (``{'variant','M_present','state','group'}``) whose
    per-param ``state`` keeps ONLY the eigenbasis tensors in ``_BASIS_STATE_KEYS`` (QL/QR), each
    detached + CPU + cloned.

    REDESIGN 2026-06-16: the Extractor no longer carries a previous-checkpoint ``prev_opt`` (the
    §6.1 rotation-vs-prev is POST-HOC now), so this is no longer called on the live path. It is
    kept importable because a future POST-HOC pass / test may want a QL/QR-only view of an opt
    dict. Returns None for a None / stateless input; the result is still a well-formed ``opt``
    view (the same keys present), just with trimmed per-param state dicts."""
    if opt is None:
        return None
    trimmed_state: Dict[str, Any] = {}
    for name, st in (opt.get("state", {}) or {}).items():
        if not isinstance(st, dict):
            continue
        entry: Dict[str, Any] = {}
        for k in _BASIS_STATE_KEYS:
            v = st.get(k)
            if v is not None:
                entry[k] = _cpu_detach_clone(v)
        if entry:
            trimmed_state[name] = entry
    return {
        "variant": opt.get("variant", "unknown"),
        "M_present": bool(opt.get("M_present", False)),
        "state": trimmed_state,
        "group": opt.get("group", {}) or {},
    }


def _extract_Q0(opt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """From an ``opt`` view (typically the init checkpoint's), pull the per-param Kronecker
    eigenbasis ``{name: {'QL':..,'QR':..}}`` to serve as the ancestral metric basis ``Q0``
    (briefing §6.1 init-relative rotation). Returns None if no preconditioned param carries a
    basis yet (e.g. AdamW, or step 0 before the first preconditioned step).

    REDESIGN 2026-06-16: the live path no longer carries ``Q0`` in memory (the §6.1 init-relative
    rotation is POST-HOC, reconstructed from the per-step ``ql_eigvecs``/``qr_eigvecs`` arrays
    block_optim now stores). This helper is kept importable for the POST-HOC pass / tests.

    The retained tensors are detached + CPU + CLONED so ``Q0`` holds no reference to the loaded
    checkpoint's storage."""
    state = opt.get("state", {})
    q0: Dict[str, Any] = {}
    for name, st in state.items():
        if not isinstance(st, dict):
            continue
        ql, qr = st.get("QL"), st.get("QR")
        if ql is not None or qr is not None:
            entry: Dict[str, Any] = {}
            if ql is not None:
                entry["QL"] = _cpu_detach_clone(ql)
            if qr is not None:
                entry["QR"] = _cpu_detach_clone(qr)
            q0[name] = entry
    return q0 or None


# =============================================================================
# Fixed INIT reference loading (CONTRACT (C): "vs init" loads from a fixed file on disk)
# =============================================================================

#: Candidate file names for the run's fixed INIT (E(0)) reference, most-preferred first. The
#: kept-storage redesign MOVES kept ladder checkpoints to ``<run>/kept/``; the init/Phase-2-start
#: reference is ``init_model.pt`` (the trainer's Phase-2 init) or ``phase1_final.pt`` (the
#: equalization endpoint). We look in ``<run>/kept/`` first, then the run output dir itself.
_INIT_REFERENCE_NAMES: Tuple[str, ...] = ("init_model.pt", "phase1_final.pt")


def resolve_init_reference_path(run_dir: str) -> Optional[str]:
    """Resolve the FIXED on-disk INIT (E(0)) reference for a run (CONTRACT (C)).

    ``run_dir`` is the run's watch dir (``<output_dir>/checkpoints``) or the run output dir. We
    search, in order: ``<run>/kept/{init_model.pt,phase1_final.pt}`` (kept-storage redesign
    location), then ``<output_dir>/kept/...`` and ``<output_dir>/{init_model.pt,phase1_final.pt}``.
    Returns the first existing path, or None if no init reference is on disk yet (the extractor
    then runs with ``init_model_sd=None`` and the "vs init" pieces skip cleanly).
    """
    run_dir = os.path.abspath(run_dir.rstrip("/"))
    output_dir = (os.path.dirname(run_dir)
                  if os.path.basename(run_dir) == "checkpoints" else run_dir)
    search_dirs = [
        os.path.join(run_dir, "kept"),
        os.path.join(output_dir, "kept"),
        run_dir,
        output_dir,
    ]
    seen: set = set()
    for d in search_dirs:
        if d in seen:
            continue
        seen.add(d)
        for name in _INIT_REFERENCE_NAMES:
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return cand
    return None


def load_init_model_sd(init_ref_path: Optional[str],
                       map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    """Load the FIXED INIT (E(0)) model_state_dict from ``init_ref_path`` ONCE (the Extractor
    calls this when building shared resources, then threads the result READ-ONLY into every ctx).

    The tensors are detached + CPU + cloned so the returned dict holds no reference to a loaded
    checkpoint container. Returns None if the path is None / missing / unreadable / carries no
    ``model_state_dict`` (the "vs init" blocks then skip cleanly). NEVER raises — a bad init
    reference must not take down the whole extractor."""
    if not init_ref_path or not os.path.exists(init_ref_path):
        return None
    try:
        ckpt = torch.load(init_ref_path, map_location=map_location, weights_only=False)
    except Exception as e:
        print(f"[build_ctx] init reference {init_ref_path!r} unreadable ({e!r}); "
              f"'vs init' blocks will skip.", file=sys.stderr, flush=True)
        return None
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else None
    if not isinstance(sd, dict):
        print(f"[build_ctx] init reference {init_ref_path!r} has no model_state_dict; "
              f"'vs init' blocks will skip.", file=sys.stderr, flush=True)
        return None
    out = {k: _cpu_detach_clone(v) for k, v in sd.items()}
    del ckpt
    return out


def build_ctx(
    ckpt_path: str,
    ctx_resources: Dict[str, Any],
    *,
    init_model_sd: Optional[Dict[str, Any]] = None,
    prev_model_sd: Optional[Dict[str, Any]] = None,
    prev_opt: Optional[Dict[str, Any]] = None,
    Q0: Optional[Dict[str, Any]] = None,
    num_classes: int = 2,
    map_location: str = "cpu",
) -> ExtractCtx:
    """Load a training checkpoint and assemble its :class:`ExtractCtx`.

    The collector renames a ready checkpoint to ``<ckpt>.processing`` and passes that path here.
    This loads it with ``torch.load`` (``weights_only=False`` — the checkpoint carries the
    optimizer state dict, which is not a plain tensor bag), rebuilds the ViT-L/16, loads the
    model weights, sets eval mode, and reconstructs ``ctx.opt`` from the saved optimizer state.

    Args:
        ckpt_path: path to the (already-claimed) checkpoint ``.pt`` on disk.
        ctx_resources: shared, run-invariant resources owned by the Extractor. Required keys:
            ``device`` (torch.device or str), ``criterion`` (nn.Module), ``projection``
            (SparseRandomProjection), ``scalar`` (ScalarStore), ``array`` (ArrayStore),
            ``manifest`` (dict). Optional: ``probe_eval_loader``, ``probe_train_loader``,
            ``hessian_batch`` (a ``(images, targets)`` tuple; moved to ``device`` here if on CPU).
            Optional ``init_model_sd``: the FIXED INIT (E(0)) reference loaded ONCE from disk and
            threaded read-only into every ctx (CONTRACT (C)); if present it is used directly,
            otherwise build_ctx leaves ``ctx.init_model_sd`` None and the "vs init" blocks skip.
        init_model_sd / prev_model_sd / prev_opt / Q0: REDESIGN 2026-06-16 — these are no longer
            carried cross-checkpoint. ``init_model_sd`` may be passed (it equals
            ``ctx_resources['init_model_sd']`` when present); ``prev_model_sd`` / ``prev_opt`` /
            ``Q0`` are always None on the live path (the diagnostics that used them are POST-HOC)
            and are accepted only for back-compat with older callers / tests.
        num_classes: model output dim (2 for reproductive).
        map_location: where ``torch.load`` puts tensors. Default "cpu" — the model weights are
            moved to ``device`` after loading, while large optimizer tensors stay on CPU until a
            block needs them on-device (keeps peak GPU memory to one model + its activations).

    Returns:
        A fully populated :class:`ExtractCtx`.
    """
    device = torch.device(ctx_resources["device"]) if not isinstance(
        ctx_resources["device"], torch.device) else ctx_resources["device"]

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # --- model ---------------------------------------------------------------------------
    model = vit_large_patch16(num_classes=num_classes)
    model_sd = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing or unexpected:
        print(f"[build_ctx] load_state_dict(strict=False): {len(missing)} missing, "
              f"{len(unexpected)} unexpected (first missing={missing[:4]}, "
              f"first unexpected={list(unexpected)[:4]}).", file=sys.stderr)
    model = model.to(device).eval()
    # ``load_state_dict`` has copied the weights INTO the model's own params; drop the loaded
    # checkpoint's model_state_dict reference now so the host-RAM copy of the model weights is
    # freed before the (large) optimizer-state remap, rather than lingering until ``ckpt`` dies.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt["model_state_dict"] = None
    del model_sd

    # --- optimizer state view ------------------------------------------------------------
    # _reconstruct_opt REFERENCES the loaded optimizer tensors (it does not copy them), so the
    # remapped ``opt['state']`` aliases ``ckpt['optimizer_state_dict']``; the checkpoint's tensor
    # storage stays alive via ctx.opt and is freed when the Extractor frees ctx.opt per ckpt.
    opt = _reconstruct_opt(ckpt)

    # --- identity / timeline -------------------------------------------------------------
    manifest = ctx_resources.get("manifest", {}) or {}
    condition = ckpt.get("condition", manifest.get("condition", "unknown"))
    variant = ckpt.get("variant", opt.get("variant", manifest.get("variant", "unknown")))
    seed = int(ckpt.get("seed", manifest.get("seed", 0)) or 0)
    run_id = ckpt.get("run_id", manifest.get("run_id",
                                             f"{condition}__{variant}__s{seed}"))
    step = int(ckpt.get("step", ckpt.get("global_step", 0)) or 0)
    wall_time = float(ckpt.get("wall_time", time.time()))

    # --- fixed Hessian batch onto device -------------------------------------------------
    hessian_batch = ctx_resources.get("hessian_batch")
    if hessian_batch is not None:
        imgs, tgts = hessian_batch
        if isinstance(imgs, torch.Tensor) and imgs.device != device:
            imgs = imgs.to(device)
        if isinstance(tgts, torch.Tensor) and tgts.device != device:
            tgts = tgts.to(device)
        hessian_batch = (imgs, tgts)

    # --- inline §6.2 global projection coords (collector-speed) --------------------------
    # If the trainer stored the global projected coords at emission, surface them so
    # block_weights can skip the dominant 302M-coord rmatvec. Detach + CPU + float32 the coords
    # (a tiny R^d vector) so they hold no reference to the loaded checkpoint blob. Absent on
    # old checkpoints / Phase-1 saves -> both stay None (block_weights then recomputes).
    inline_proj_coords = ckpt.get("proj_coords")
    if isinstance(inline_proj_coords, torch.Tensor):
        inline_proj_coords = inline_proj_coords.detach().to("cpu", dtype=torch.float32)
    elif inline_proj_coords is not None:
        # Tolerate a non-tensor (e.g. a numpy array) by coercing; None stays None.
        try:
            inline_proj_coords = torch.as_tensor(inline_proj_coords, dtype=torch.float32)
        except Exception:
            inline_proj_coords = None
    inline_proj_meta = ckpt.get("proj_meta")
    if inline_proj_meta is not None and not isinstance(inline_proj_meta, dict):
        inline_proj_meta = None

    # --- fixed INIT (E(0)) reference (CONTRACT (C)): prefer the explicit kwarg, else the
    # shared resource the Extractor loaded ONCE from disk. Read-only; identical every ckpt.
    if init_model_sd is None:
        init_model_sd = ctx_resources.get("init_model_sd")

    ctx = ExtractCtx(
        condition=condition,
        variant=variant,
        run_id=run_id,
        seed=seed,
        step=step,
        wall_time=wall_time,
        device=device,
        model=model,
        criterion=ctx_resources["criterion"],
        opt=opt,
        init_model_sd=init_model_sd,
        prev_model_sd=prev_model_sd,
        prev_opt=prev_opt,
        Q0=Q0,
        probe_eval_loader=ctx_resources.get("probe_eval_loader"),
        probe_train_loader=ctx_resources.get("probe_train_loader"),
        hessian_batch=hessian_batch,
        projection=ctx_resources.get("projection"),
        inline_proj_coords=inline_proj_coords,
        inline_proj_meta=inline_proj_meta,
        scalar=ctx_resources.get("scalar"),
        array=ctx_resources.get("array"),
        manifest=manifest,
    )
    # Surface the checkpoint's OWN deterministic emit-ordinal + sampler floor into the
    # per-checkpoint scratch cache (CONTRACT (B)): the driver uses them for the order-independent
    # heavy-tier decision without re-loading the checkpoint and without a new dataclass field.
    # Absent fields stay absent and the driver falls back to the step rule. ``cache`` is reset
    # per checkpoint, so this is pure per-checkpoint metadata (no cross-checkpoint state).
    for _meta_key in ("emit_ordinal", "ckpt_index", "emit_index", "sample_index",
                      "sampler_min_step"):
        if isinstance(ckpt, dict) and _meta_key in ckpt and ckpt[_meta_key] is not None:
            ctx.cache[_meta_key] = ckpt[_meta_key]
    # Drop the local checkpoint-dict reference: the only big tensors we still need (the optimizer
    # state) are reachable via ctx.opt['state'], and the model weights are now in the model. This
    # lets the loaded ``ckpt`` container (its metadata + the now-None model_state_dict slot) be
    # reclaimed promptly instead of surviving on the call stack until the caller returns.
    del ckpt
    return ctx


def trainable_names_in_order(model: nn.Module) -> List[str]:
    """Convenience: the ordered trainable-parameter names (frozen input stage excluded), the
    same order the trainer must write as ``ckpt['param_names']``. Lets the extractor or a test
    cross-check that a checkpoint's ``param_names`` matches the live model's trainable set."""
    return [name for name, _ in trainable_named_parameters(model, include_requires_grad_false=True)]
