#!/usr/bin/env python3
"""
PhenoVision pretraining-as-preadaptation: design verification gates (Part I §6).

Gates the validity of the controlled-constant input-encoding design BEFORE any
fine-tuning run. The experiment freezes the input-encoding stage (patch_embed +
pos_embed) and holds it identical across all four conditions; that only makes sense
if the pretrained input stages are near-identical to begin with. This script measures
the drift and prints a verdict.

What it does
------------
1. Loads patch_embed.proj.weight, patch_embed.proj.bias, pos_embed from three REAL
   checkpoints:
     - imagenet  : timm.create_model("vit_large_patch16_224", pretrained=True)   (supervised)
     - mae       : timm.create_model("vit_large_patch16_224.mae", pretrained=True) (self-supervised MIM)
     - plantclef : torch.load(<...epoch100.pth>)["model"]  (VT / Virtual Taxonomist, supervised species)
2. Input-encoding drift vs ImageNet, for BOTH plantclef and mae:
     - relative Frobenius  ||E - E_IN|| / ||E_IN||  for proj.weight, proj.bias, pos_embed separately.
     - per-filter cosine similarity for proj.weight (each of the 1024 output filters); mean and min.
     - pos_embed shape mismatch handled by interpolation (PlantCLEF2022/util/pos_embed.interpolate_pos_embed).
3. Architecture sanity across all three real checkpoints (proj.weight [1024,3,16,16],
   pos_embed [1,197,1024], 24 blocks).
4. Pretraining-type note (plantclef supervised -> uniform-LR fair; MAE self-supervised MIM -> flag caveat).
5. VERDICT: drift "small" (=> recommend exact-control copy of ImageNet tokenizer+pos into all four)
   vs "large" (=> ESCALATE). Threshold: relative Frobenius < 0.15 AND mean per-filter cosine > 0.9.

Run
---
    mamba run -n reticulate-gpu2 python xAI/py/verify_design.py

CPU only, no training. Downloads ImageNet + MAE weights from timm/HF on first run.
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import torch

# Project paths (PlantCLEF2022 for models_vit + pos_embed util).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "PlantCLEF2022"))
sys.path.insert(0, _PROJECT_ROOT)

import timm  # noqa: E402

# Expected ViT-L/16 architecture facts (verified from PlantCLEF2022/models_vit.py::vit_large_patch16).
EMBED_DIM = 1024
DEPTH = 24
NUM_HEADS = 16
HEAD_DIM = 64
PATCH = 16
RESOLUTION = 224
N_FILTERS = EMBED_DIM            # 1024 patch-embed output filters
EXPECTED_PROJ_W = (1024, 3, 16, 16)
EXPECTED_PROJ_B = (1024,)
EXPECTED_POS = (1, 197, 1024)    # cls token + 196 patches (14x14)

# Verdict thresholds (briefing Part I §6 / plan C1).
FROB_SMALL = 0.15        # relative Frobenius below this counts as "small"
COS_SMALL = 0.90         # mean per-filter cosine above this counts as "small"

PLANTCLEF_PATH = os.path.join(
    _PROJECT_ROOT, "models", "PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"
)

# Keys we extract from every state_dict.
PROJ_W = "patch_embed.proj.weight"
PROJ_B = "patch_embed.proj.bias"
POS = "pos_embed"


# =============================================================================
# Loading the three real input-encoding stages
# =============================================================================

def _count_blocks(sd: Dict[str, torch.Tensor]) -> int:
    """Count transformer blocks by scanning blocks.<i>.* keys."""
    idxs = set()
    for k in sd:
        if k.startswith("blocks."):
            try:
                idxs.add(int(k.split(".")[1]))
            except (IndexError, ValueError):
                pass
    return (max(idxs) + 1) if idxs else 0


def _input_stage(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pull the three frozen-stage tensors from a state_dict, in fp32 on CPU."""
    out: Dict[str, torch.Tensor] = {}
    for k in (PROJ_W, PROJ_B, POS):
        if k not in sd:
            raise KeyError(f"missing key '{k}' in state_dict (have e.g. {list(sd)[:5]} ...)")
        out[k] = sd[k].detach().float().cpu().clone()
    out["_n_blocks"] = _count_blocks(sd)  # type: ignore[assignment]
    return out


def load_imagenet() -> Dict[str, torch.Tensor]:
    """timm supervised ImageNet ViT-L/16."""
    m = timm.create_model("vit_large_patch16_224", pretrained=True)
    return _input_stage(m.state_dict())


def load_mae() -> Dict[str, torch.Tensor]:
    """timm MAE self-supervised ViT-L/16."""
    m = timm.create_model("vit_large_patch16_224.mae", pretrained=True)
    return _input_stage(m.state_dict())


def load_plantclef(path: str = PLANTCLEF_PATH) -> Dict[str, torch.Tensor]:
    """PlantCLEF / VT checkpoint on disk; weights under the 'model' key."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PlantCLEF checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"expected a 'model' key in {path}; got {list(ckpt)[:10]}")
    return _input_stage(ckpt["model"])


# =============================================================================
# Drift metrics
# =============================================================================

def relative_frobenius(e: torch.Tensor, e_in: torch.Tensor) -> float:
    """||e - e_in||_F / ||e_in||_F."""
    denom = torch.linalg.vector_norm(e_in.reshape(-1)).item()
    if denom == 0.0:
        return float("nan")
    num = torch.linalg.vector_norm((e - e_in).reshape(-1)).item()
    return num / denom


def per_filter_cosine(w: torch.Tensor, w_in: torch.Tensor) -> Tuple[float, float, float]:
    """
    Per-filter cosine similarity for patch_embed.proj.weight [n_filters, 3, 16, 16].
    Flatten each output filter to a vector and compare. Returns (mean, min, median).
    """
    n = w.shape[0]
    a = w.reshape(n, -1)
    b = w_in.reshape(n, -1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)  # [n_filters]
    return cos.mean().item(), cos.min().item(), cos.median().item()


def align_pos_embed(
    pos: torch.Tensor, pos_in: torch.Tensor
) -> Tuple[torch.Tensor, Optional[str]]:
    """
    Align `pos` to the spatial grid of `pos_in` (the ImageNet reference) so a drift
    norm is well-defined. If the patch-token counts already match, no-op. Otherwise
    bicubically interpolate the patch tokens (matching the project's
    util.pos_embed.interpolate_pos_embed convention: 1 extra cls token, square grid).
    Returns (aligned_pos, note) where note is non-None when interpolation/mismatch occurred.
    """
    if pos.shape == pos_in.shape:
        return pos, None

    num_extra = 1  # cls token
    n_tok, n_tok_in = pos.shape[1], pos_in.shape[1]
    n_patch, n_patch_in = n_tok - num_extra, n_tok_in - num_extra
    dim, dim_in = pos.shape[2], pos_in.shape[2]

    if dim != dim_in:
        return pos, (
            f"embed-dim mismatch ({dim} vs {dim_in}); cannot align pos_embed -- "
            f"reporting raw mismatch as NaN"
        )

    orig = int(round(n_patch ** 0.5))
    new = int(round(n_patch_in ** 0.5))
    if orig * orig != n_patch or new * new != n_patch_in:
        return pos, (
            f"non-square patch grid ({n_patch} -> {orig}^2, {n_patch_in} -> {new}^2); "
            f"cannot interpolate cleanly"
        )

    extra = pos[:, :num_extra]
    tok = pos[:, num_extra:]
    tok = tok.reshape(1, orig, orig, dim).permute(0, 3, 1, 2)
    tok = torch.nn.functional.interpolate(
        tok, size=(new, new), mode="bicubic", align_corners=False
    )
    tok = tok.permute(0, 2, 3, 1).reshape(1, new * new, dim)
    aligned = torch.cat([extra, tok], dim=1)
    note = f"pos_embed interpolated {orig}x{orig} -> {new}x{new} to match ImageNet"
    return aligned, note


# =============================================================================
# Per-condition drift computation
# =============================================================================

def compute_drift(
    stage: Dict[str, torch.Tensor], imagenet: Dict[str, torch.Tensor]
) -> Dict[str, object]:
    """All drift metrics for one condition vs the ImageNet reference."""
    res: Dict[str, object] = {}

    res["frob_proj_weight"] = relative_frobenius(stage[PROJ_W], imagenet[PROJ_W])
    res["frob_proj_bias"] = relative_frobenius(stage[PROJ_B], imagenet[PROJ_B])

    cos_mean, cos_min, cos_med = per_filter_cosine(stage[PROJ_W], imagenet[PROJ_W])
    res["cos_mean"] = cos_mean
    res["cos_min"] = cos_min
    res["cos_median"] = cos_med

    pos_aligned, note = align_pos_embed(stage[POS], imagenet[POS])
    res["pos_note"] = note
    if pos_aligned.shape == imagenet[POS].shape:
        res["frob_pos_embed"] = relative_frobenius(pos_aligned, imagenet[POS])
    else:
        res["frob_pos_embed"] = float("nan")  # unalignable mismatch

    return res


def condition_is_small(drift: Dict[str, object]) -> bool:
    """A condition's input stage is 'small'-drift iff all drift Frobenii are below
    FROB_SMALL AND its mean per-filter cosine exceeds COS_SMALL. NaN Frobenius
    (unalignable pos_embed) does NOT count as small."""
    frobs = [
        drift["frob_proj_weight"],
        drift["frob_proj_bias"],
        drift["frob_pos_embed"],
    ]
    for f in frobs:
        if not (isinstance(f, float) and f == f):  # reject NaN
            return False
        if f >= FROB_SMALL:
            return False
    return float(drift["cos_mean"]) > COS_SMALL  # type: ignore[arg-type]


# =============================================================================
# Architecture sanity
# =============================================================================

def architecture_report(name: str, stage: Dict[str, torch.Tensor]) -> Tuple[str, bool]:
    """Verify proj.weight/proj.bias/pos_embed shapes and block count; return (text, ok)."""
    pw = tuple(stage[PROJ_W].shape)
    pb = tuple(stage[PROJ_B].shape)
    pe = tuple(stage[POS].shape)
    nb = int(stage["_n_blocks"])  # type: ignore[arg-type]

    ok_pw = pw == EXPECTED_PROJ_W
    ok_pb = pb == EXPECTED_PROJ_B
    ok_pe = pe == EXPECTED_POS
    ok_nb = nb == DEPTH
    ok = ok_pw and ok_pb and ok_pe and ok_nb

    def mark(b: bool) -> str:
        return "OK " if b else "BAD"

    text = (
        f"  {name:<10} proj.weight={pw} [{mark(ok_pw)}]  "
        f"proj.bias={pb} [{mark(ok_pb)}]  "
        f"pos_embed={pe} [{mark(ok_pe)}]  "
        f"blocks={nb} [{mark(ok_nb)}]"
    )
    return text, ok


# =============================================================================
# Reporting
# =============================================================================

def print_summary_table(drifts: Dict[str, Dict[str, object]]) -> None:
    print("\n" + "=" * 78)
    print("INPUT-ENCODING DRIFT vs ImageNet  (E = patch_embed/pos_embed)")
    print("=" * 78)
    header = (
        f"{'condition':<10} {'frob(proj.W)':>13} {'frob(proj.b)':>13} "
        f"{'frob(pos)':>11} {'cos_mean':>9} {'cos_min':>9}"
    )
    print(header)
    print("-" * len(header))
    for cond, d in drifts.items():
        def fmt(x: object) -> str:
            return f"{x:.4f}" if isinstance(x, float) and x == x else "  NaN "
        print(
            f"{cond:<10} {fmt(d['frob_proj_weight']):>13} {fmt(d['frob_proj_bias']):>13} "
            f"{fmt(d['frob_pos_embed']):>11} {fmt(d['cos_mean']):>9} {fmt(d['cos_min']):>9}"
        )
    for cond, d in drifts.items():
        if d.get("pos_note"):
            print(f"  note [{cond}]: {d['pos_note']}")
    print()
    print(f"  thresholds for 'small': relative Frobenius < {FROB_SMALL} (all of proj.W/proj.b/pos) "
          f"AND mean per-filter cosine > {COS_SMALL}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser(description="PhenoVision design verification gates (Part I §6)")
    p.add_argument("--plantclef_path", type=str, default=PLANTCLEF_PATH)
    args = p.parse_args()

    torch.manual_seed(0)  # determinism; this script has no stochastic ops, but be explicit.

    print("=" * 78)
    print("PhenoVision pretraining-as-preadaptation: DESIGN VERIFICATION (Part I §6)")
    print("=" * 78)
    print("Architecture under test: ViT-L/16  D=1024  depth=24  heads=16  head_dim=64  "
          f"patch={PATCH}  res={RESOLUTION}")
    print()

    # --- Load the three real input-encoding stages ---
    print("Loading input-encoding stages (downloads ImageNet+MAE weights on first run)...")
    print("  [1/3] imagenet  (timm vit_large_patch16_224, supervised) ...")
    imagenet = load_imagenet()
    print("  [2/3] mae       (timm vit_large_patch16_224.mae, self-supervised MIM) ...")
    mae = load_mae()
    print(f"  [3/3] plantclef (VT, on disk: {args.plantclef_path}) ...")
    plantclef = load_plantclef(args.plantclef_path)
    print("  all three loaded.")

    # --- Architecture sanity (gate 3) ---
    print("\n" + "=" * 78)
    print("ARCHITECTURE SANITY (gate 3)")
    print("=" * 78)
    arch_ok = True
    for name, stage in (("imagenet", imagenet), ("mae", mae), ("plantclef", plantclef)):
        text, ok = architecture_report(name, stage)
        print(text)
        arch_ok = arch_ok and ok
    print(f"\n  architecture sanity: {'PASS' if arch_ok else 'FAIL'}")

    # --- Drift metrics (gate 1) ---
    drifts = {
        "plantclef": compute_drift(plantclef, imagenet),
        "mae": compute_drift(mae, imagenet),
    }
    print_summary_table(drifts)

    # --- Pretraining type note (gate 2) ---
    print("\n" + "=" * 78)
    print("PRETRAINING TYPE (gate 2)")
    print("=" * 78)
    print("  plantclef / VT (Virtual Taxonomist): SUPERVISED species classification.")
    print("    -> uniform-LR fine-tuning (no layer-wise decay) is FAIR for this start.")
    print("  mae: SELF-SUPERVISED masked-image modeling (MIM).")
    print("    -> CAVEAT: layer-wise LR decay genuinely matters for MIM-pretrained models;")
    print("       uniform LR may DISADVANTAGE the MAE start. Flag as a known confound for")
    print("       the MAE condition in the writeup; do NOT silently re-enable layer-wise decay")
    print("       (it would break cross-condition comparability). Escalate if visibly severe.")

    # --- Verdict (gate 5) ---
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    per_cond_small = {c: condition_is_small(d) for c, d in drifts.items()}
    for c, small in per_cond_small.items():
        print(f"  {c:<10} drift is {'SMALL' if small else 'LARGE'}")

    all_small = all(per_cond_small.values()) and arch_ok
    print()
    if all_small:
        print("  ==> OVERALL: SMALL drift. The controlled-constant input-stage assumption HOLDS.")
        print("      RECOMMENDATION: adopt the EXACT-CONTROL option (briefing Part II §6.1):")
        print("      copy the SAME ImageNet patch_embed + pos_embed into ALL FOUR conditions")
        print("      (imagenet, plantclef, mae, naive) before fine-tuning, so input-stage")
        print("      variance across conditions is exactly zero by construction.")
        verdict = "SMALL -> adopt exact-control copy"
    else:
        print("  ==> OVERALL: LARGE drift (or architecture mismatch). The controlled-constant")
        print("      input-stage assumption FAILS for the flagged component(s).")
        print("      ACTION: ESCALATE -- the design needs revisiting before running (briefing")
        print("      Part I §6 / Escalate list). Do not launch the main block.")
        verdict = "LARGE -> ESCALATE"

    print(f"\n  FINAL VERDICT: {verdict}")
    print("=" * 78)

    # Exit 0 on small (gate passed), 2 on large/escalate, so a launcher can branch on it.
    return 0 if all_small else 2


if __name__ == "__main__":
    sys.exit(main())
