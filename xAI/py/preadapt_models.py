#!/usr/bin/env python3
"""
preadapt_models.py — model-init / freezing for the pretraining-as-preadaptation experiment.

This module builds the ViT-L/16 used by every condition of the experiment and enforces the
frozen input-encoding discipline from the briefing (Part I §3, §4; Part II §6.1, §6.6).

Four starting conditions ("evolutionary histories"), named identically everywhere in metadata:
  * 'imagenet'  — supervised      : timm vit_large_patch16_224 (pretrained)
  * 'plantclef' — VT / Virtual Taxonomist (MAE-pretrained on PlantCLEF, species supervised head)
  * 'mae'       — self-supervised : timm vit_large_patch16_224.mae (pretrained)
  * 'naive'     — Very Naive Baseline: fresh random ViT, with the ImageNet patch-embed tokenizer
                  + positional embeddings copied in; everything else (CLS, blocks, norm, head) random.
                  This is NOT "from scratch"/"random init"/"neutral" — it carries a pretrained
                  tokenizer (briefing §3.3). Do not relabel it.

FROZEN INPUT-ENCODING STAGE (briefing §4.1): in ALL conditions the patch-embedding projection
(`patch_embed.proj.weight` / `patch_embed.proj.bias`) and the positional embeddings (`pos_embed`)
are frozen (`requires_grad=False`) AND excluded from the optimizer param groups entirely — never
given lr=0, so no weight decay or optimizer state ever touches them. The CLS token, transformer
blocks, final norm, and head all stay trainable.

EXACT-CONTROL OPTION (briefing §6.1): when `exact_control_tokenizer=True` (the default), AFTER
loading each condition's source weights we OVERWRITE `patch_embed.proj.weight/bias` and `pos_embed`
with the ImageNet ones, so the frozen input stage is BYTE-IDENTICAL across all four conditions and
input-stage variance is exactly zero by construction. For 'naive' this is already the case (the
tokenizer/pos copied in are ImageNet's); for the other conditions this resets their (typically
barely-moved) input stage to ImageNet's.

Phase-2 uses a UNIFORM learning rate across trainable params (NO layer-wise decay; briefing §4.3).
`build_uniform_param_groups` splits trainable params into a weight-decay group (ndim>=2) and a
no-decay group (ndim==1: LayerNorm, biases, and the 3-D `cls_token`/`pos_embed`-shaped vectors),
and excludes the frozen input stage entirely.

RNG: head reinit uses `nn.init.trunc_normal_`, which draws from the global generator. To keep
condition-to-condition reproducibility independent of global RNG state (per
xAI/two_noise/CONVENTIONS.md), pass an explicit `torch.Generator` via `head_generator` and it will
be threaded through the head reinit; otherwise the global RNG is used and a warning-free fallback
applies.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Optional

import torch
import torch.nn as nn

# --- project paths (mirror xai_train.py so PlantCLEF2022 modules import cleanly) ---------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLANTCLEF = os.path.join(_HERE, "..", "..", "PlantCLEF2022")
_PROJECT_ROOT = os.path.join(_HERE, "..", "..")
for _p in (_PLANTCLEF, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models_vit import vit_large_patch16  # noqa: E402
from util.pos_embed import interpolate_pos_embed  # noqa: E402


# =============================================================================
# Constants
# =============================================================================

#: Path to the PlantCLEF (VT / Virtual Taxonomist) checkpoint.
PLANTCLEF_CKPT = (
    "/blue/guralnick/share/r.dinnage/Projects/phenovision/"
    "models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"
)

#: Conditions that have an external source state-dict ('naive' is handled in-code).
SOURCE_CONDITIONS = ("imagenet", "mae", "plantclef")

#: All valid condition names.
CONDITIONS = ("imagenet", "mae", "plantclef", "naive")

#: Parameter names that make up the frozen input-encoding stage. Names follow the timm
#: VisionTransformer convention (the same one models_vit.py subclasses).
_INPUT_STAGE_PREFIX = "patch_embed."
_POS_EMBED_NAME = "pos_embed"

#: The exact tensor names copied for the exact-control tokenizer + the 'naive' build.
_TOKENIZER_KEYS = ("patch_embed.proj.weight", "patch_embed.proj.bias")
_INPUT_STAGE_KEYS = _TOKENIZER_KEYS + (_POS_EMBED_NAME,)


# =============================================================================
# Source state dicts
# =============================================================================

def load_source_state_dict(condition: str) -> dict:
    """Return a state_dict for one of the source-pretrained conditions.

    Args:
        condition: one of 'imagenet', 'mae', 'plantclef'. ('naive' has no source and is
            handled directly in :func:`init_model`.)

    Returns:
        A plain ``{name: Tensor}`` state dict on CPU. For 'plantclef', shape-mismatched
        ``head.*`` entries are stripped (we reinitialize the head for a 2-class task).
    """
    if condition == "imagenet":
        import timm

        return timm.create_model("vit_large_patch16_224", pretrained=True).state_dict()

    if condition == "mae":
        import timm

        return timm.create_model("vit_large_patch16_224.mae", pretrained=True).state_dict()

    if condition == "plantclef":
        if not os.path.exists(PLANTCLEF_CKPT):
            raise FileNotFoundError(
                f"PlantCLEF (VT) checkpoint not found: {PLANTCLEF_CKPT}"
            )
        ckpt = torch.load(PLANTCLEF_CKPT, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # Strip the pretraining head (species classifier) — it shape-mismatches the 2-class head.
        sd = dict(sd)
        for k in ("head.weight", "head.bias"):
            sd.pop(k, None)
        return sd

    raise ValueError(
        f"load_source_state_dict: condition {condition!r} has no source state dict. "
        f"Expected one of {SOURCE_CONDITIONS} (use init_model for 'naive')."
    )


# =============================================================================
# Head reinit (RNG-threaded)
# =============================================================================

def _reinit_head(model: nn.Module, generator: Optional[torch.Generator] = None) -> None:
    """Reinitialize the classification head: trunc_normal_ weight (std=1e-5), zeros bias.

    If ``generator`` is provided, it is threaded through the weight draw so head init is
    reproducible independently of global RNG state (per CONVENTIONS.md). Older torch
    ``nn.init.trunc_normal_`` signatures do not accept ``generator``; in that case we fall
    back to a manual generator-driven draw to honor the contract.
    """
    with torch.no_grad():
        if generator is not None:
            try:
                nn.init.trunc_normal_(model.head.weight, std=1e-5, generator=generator)
            except TypeError:
                # Manual truncated-normal in [-2, 2] sigma, generator-driven.
                w = model.head.weight
                tmp = torch.empty_like(w)
                # Rejection-free approx via clamp is acceptable here (std is 1e-5, head only).
                tmp.normal_(mean=0.0, std=1e-5, generator=generator).clamp_(-2e-5, 2e-5)
                w.copy_(tmp)
        else:
            nn.init.trunc_normal_(model.head.weight, std=1e-5)
        nn.init.zeros_(model.head.bias)


# =============================================================================
# Model construction
# =============================================================================

def init_model(
    condition: str,
    num_classes: int = 2,
    device: str = "cuda",
    exact_control_tokenizer: bool = True,
    head_generator: Optional[torch.Generator] = None,
) -> nn.Module:
    """Build a ViT-L/16 initialized for the given condition, with the frozen input stage applied.

    Args:
        condition: 'imagenet' | 'mae' | 'plantclef' | 'naive'.
        num_classes: output dim (2 for reproductive: flowers+fruits).
        device: device to move the model to ("cuda" / "cpu").
        exact_control_tokenizer: if True (default), after loading the condition's source weights,
            overwrite ``patch_embed.proj.weight/bias`` and ``pos_embed`` with the ImageNet ones so
            the frozen input stage is byte-identical across all four conditions (briefing §6.1).
            For 'naive' this is already satisfied (ImageNet tokenizer/pos are what gets copied in).
        head_generator: optional ``torch.Generator`` threaded through the head reinit for
            reproducibility independent of global RNG.

    Returns:
        The model on ``device``. Note: this function does NOT freeze anything — call
        :func:`freeze_input_stage` afterward (kept separate so the caller controls phase ordering).

    Raises:
        ValueError: unknown ``condition``.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}.")

    model = vit_large_patch16(num_classes=num_classes)

    if condition in SOURCE_CONDITIONS:
        src = load_source_state_dict(condition)
        # Interpolate pos_embed if the checkpoint grid differs from this model's grid.
        interpolate_pos_embed(model, src)
        missing, unexpected = model.load_state_dict(src, strict=False)
        _report_load(condition, missing, unexpected)
        _reinit_head(model, head_generator)

    elif condition == "naive":
        # Fresh random ViT (blocks/CLS/norm/head random); copy in ImageNet tokenizer + pos_embed.
        imagenet_sd = load_source_state_dict("imagenet")
        _copy_input_stage(model, imagenet_sd, source_name="imagenet")
        # Head is already fresh-random from construction, but reinit to match the documented
        # head-init recipe and to thread the generator. (Blocks/CLS/norm keep their random init.)
        _reinit_head(model, head_generator)

    # EXACT-CONTROL: force the frozen input stage to be byte-identical = ImageNet's, in every
    # condition. For 'naive' this is a no-op in value (already ImageNet's) but harmless; we still
    # run it for the source conditions so VT/MAE input drift is removed by construction.
    if exact_control_tokenizer:
        imagenet_sd = (
            imagenet_sd if condition == "naive" else load_source_state_dict("imagenet")
        )
        _copy_input_stage(model, imagenet_sd, source_name="imagenet (exact-control)")

    model = model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[preadapt_models] init_model(condition={condition!r}, "
        f"exact_control_tokenizer={exact_control_tokenizer}): "
        f"{n_total:,} params ({n_train:,} currently trainable, pre-freeze)."
    )
    return model


def _copy_input_stage(model: nn.Module, source_sd: dict, source_name: str) -> None:
    """Copy patch_embed.proj.weight/bias and pos_embed from ``source_sd`` into ``model`` in-place.

    Shapes must match exactly (same arch/resolution). pos_embed must already be at the model's
    grid size — interpolate the source dict first if needed before calling this.
    """
    target_sd = model.state_dict()
    with torch.no_grad():
        for k in _INPUT_STAGE_KEYS:
            if k not in source_sd:
                raise KeyError(f"_copy_input_stage: source ({source_name}) missing key {k!r}.")
            if source_sd[k].shape != target_sd[k].shape:
                raise ValueError(
                    f"_copy_input_stage: shape mismatch for {k!r}: "
                    f"source {tuple(source_sd[k].shape)} vs model {tuple(target_sd[k].shape)}."
                )
            target_sd[k].copy_(source_sd[k].to(target_sd[k].dtype))


def _report_load(condition: str, missing: list, unexpected: list) -> None:
    """Print a concise summary of a non-strict load (expected: head missing; some unexpected)."""
    # Expected-missing: head.* (we reinit). Anything else is worth surfacing.
    nontrivial_missing = [m for m in missing if not m.startswith("head.")]
    if nontrivial_missing or unexpected:
        print(
            f"[preadapt_models] load_state_dict(condition={condition!r}, strict=False): "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"non-head-missing={nontrivial_missing[:8]}"
            + (" ..." if len(nontrivial_missing) > 8 else "")
            + f" unexpected={list(unexpected)[:8]}"
            + (" ..." if len(unexpected) > 8 else "")
        )


# =============================================================================
# Freeze discipline
# =============================================================================

def is_input_stage_param(name: str) -> bool:
    """True iff ``name`` belongs to the frozen input-encoding stage.

    Matches any ``patch_embed.*`` parameter (proj weight/bias) and exactly ``pos_embed``.
    The CLS token, blocks, norm, and head are NOT part of the input stage.
    """
    return name.startswith(_INPUT_STAGE_PREFIX) or name == _POS_EMBED_NAME


def freeze_input_stage(model: nn.Module) -> None:
    """Set ``requires_grad=False`` on every input-stage param (patch_embed.* and pos_embed).

    The CLS token, transformer blocks, final norm, and head are left as-is (trainable). This only
    flips ``requires_grad``; the optimizer-exclusion guarantee is enforced separately by building
    param groups via :func:`build_uniform_param_groups` / :func:`head_only_param_groups`, which
    skip frozen params entirely (no lr=0 hack).
    """
    n_frozen = 0
    for name, p in model.named_parameters():
        if is_input_stage_param(name):
            p.requires_grad_(False)
            n_frozen += 1
    print(f"[preadapt_models] freeze_input_stage: froze {n_frozen} input-stage tensors.")


# =============================================================================
# Param groups (uniform LR, frozen stage excluded)
# =============================================================================

def build_uniform_param_groups(model: nn.Module, weight_decay: float) -> list:
    """Build ONE uniform-LR param grouping over trainable params, excluding the frozen input stage.

    Phase-2 grouping (briefing §4.3): no layer-wise LR decay, no ``lr_scale``. Trainable params
    are split into:
      * a weight-decay group  (``ndim >= 2``: linear/qkv/proj/mlp/patch-free weights),
      * a no-decay group      (``ndim == 1`` and the ``cls_token``: LayerNorm weights, all biases,
        and the CLS token — 1D-effective params that should not be weight-decayed).

    Frozen input-stage params (``patch_embed.*``, ``pos_embed``) are EXCLUDED entirely: they never
    appear in any returned group, so no optimizer state or weight decay touches them.

    Args:
        model: the model (after :func:`freeze_input_stage`).
        weight_decay: weight decay for the decay group; the no-decay group always gets 0.

    Returns:
        ``[{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]``
        (empty groups omitted).
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen input stage (and anything else explicitly frozen) — excluded
        if is_input_stage_param(name):
            # Defensive: the frozen stage should already have requires_grad=False, but never let
            # an input-stage param into the optimizer regardless of its grad flag.
            continue
        # cls_token is [1,1,D] (ndim 3) but should be no-decay; biases/LN are ndim 1.
        if p.ndim <= 1 or name == "cls_token" or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def head_only_param_groups(model: nn.Module) -> list:
    """Return the head-only trainable param group for Phase-1 equalization (backbone frozen).

    Used during Phase 1, where the entire backbone (including the input stage) is frozen and only
    ``head.*`` is trained. Returns ``[{"params": [head.weight, head.bias]}]`` (only those that are
    currently trainable). No weight-decay split is imposed here (the head is tiny); the caller's
    Phase-1 optimizer (Adam) sets lr/wd.
    """
    head_params = [
        p for name, p in model.named_parameters()
        if name.startswith("head.") and p.requires_grad
    ]
    return [{"params": head_params}] if head_params else []


# =============================================================================
# Correctness assertion
# =============================================================================

def input_stage_state(model: nn.Module) -> dict:
    """Snapshot the input-stage tensors (detached CPU clones) for a later drift assertion.

    Returns a dict ``{name: Tensor}`` over ``patch_embed.proj.weight/bias`` and ``pos_embed``.
    Use this to capture E(0) right after init/freeze, then pass it to
    :func:`assert_frozen_input_stage`.
    """
    sd = model.state_dict()
    return {k: sd[k].detach().to("cpu").clone() for k in _INPUT_STAGE_KEYS}


def assert_frozen_input_stage(model: nn.Module, init_state: dict) -> float:
    """Return ``||E(t) - E(0)||`` over the input stage; must be ~0 (briefing §6.6 correctness check).

    Computes the L2 norm of the concatenated displacement across
    ``patch_embed.proj.weight``, ``patch_embed.proj.bias``, and ``pos_embed`` between the current
    model state and ``init_state`` (as produced by :func:`input_stage_state`). A nonzero value
    means the freezing/exclusion is broken.

    Args:
        model: the model at time t.
        init_state: snapshot dict from :func:`input_stage_state` (E(0)).

    Returns:
        The scalar Frobenius/L2 displacement norm as a Python float.
    """
    cur = model.state_dict()
    total_sq = 0.0
    for k in _INPUT_STAGE_KEYS:
        if k not in init_state:
            raise KeyError(f"assert_frozen_input_stage: init_state missing key {k!r}.")
        e0 = init_state[k].to(dtype=torch.float64)
        et = cur[k].detach().to("cpu", dtype=torch.float64)
        if e0.shape != et.shape:
            raise ValueError(
                f"assert_frozen_input_stage: shape mismatch for {k!r}: "
                f"{tuple(e0.shape)} vs {tuple(et.shape)}."
            )
        total_sq += float(torch.sum((et - e0) ** 2).item())
    return float(total_sq ** 0.5)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    # Lightweight self-test. Uses 'naive' so only the ImageNet tokenizer needs downloading
    # (still requires network for the timm ImageNet weights). Falls back to a freeze-only test
    # if the network/weights are unavailable.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="naive", choices=list(CONDITIONS))
    ap.add_argument("--no-net", action="store_true",
                    help="skip source-weight loading; test freeze/param-group plumbing only")
    args = ap.parse_args()

    if args.no_net:
        print("[selftest] --no-net: building bare vit_large_patch16 (no source weights).")
        m = vit_large_patch16(num_classes=2)
    else:
        m = init_model(args.condition, num_classes=2, device="cpu",
                       exact_control_tokenizer=True,
                       head_generator=torch.Generator().manual_seed(1234))

    e0 = input_stage_state(m)
    freeze_input_stage(m)

    # Param-group exclusion check.
    groups = build_uniform_param_groups(m, weight_decay=0.05)
    grouped_ids = {id(p) for g in groups for p in g["params"]}
    input_ids = {id(p) for n, p in m.named_parameters() if is_input_stage_param(n)}
    leaked = grouped_ids & input_ids
    assert not leaked, f"[selftest] FAIL: {len(leaked)} input-stage params leaked into optimizer groups"
    n_grouped = sum(len(g["params"]) for g in groups)
    n_trainable = sum(1 for p in m.parameters() if p.requires_grad)
    print(f"[selftest] OK: {len(groups)} uniform groups, {n_grouped} params grouped, "
          f"{n_trainable} trainable, 0 input-stage leaks.")

    # Head-only group.
    hg = head_only_param_groups(m)
    print(f"[selftest] head_only_param_groups: {sum(len(g['params']) for g in hg)} params.")

    # Frozen assertion at t0 (must be exactly 0).
    drift0 = assert_frozen_input_stage(m, e0)
    assert drift0 == 0.0, f"[selftest] FAIL: nonzero drift at t0: {drift0}"
    print(f"[selftest] frozen-input drift at t0 = {drift0} (expected 0.0).")

    # Simulate a step: perturb a trainable block param, confirm input stage still 0 drift.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.requires_grad and n.startswith("blocks.0."):
                p.add_(torch.ones_like(p))
                break
    drift1 = assert_frozen_input_stage(m, e0)
    assert drift1 == 0.0, f"[selftest] FAIL: input stage drifted after perturbing a block: {drift1}"
    print(f"[selftest] frozen-input drift after perturbing blocks.0 = {drift1} (expected 0.0).")
    print("[selftest] all assertions passed.")
