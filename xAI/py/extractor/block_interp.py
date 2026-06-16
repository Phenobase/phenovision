#!/usr/bin/env python3
"""
block_interp.py — extractor block §6.6: ViT interpretability (FORWARD/BACKWARD on the probe set).

Per the briefing (``xAI/notes/phenovision_full_briefing.md`` Part II §6.6). All metrics here
run on the **fixed probe-eval set** (``ctx.probe_eval_loader``; §4 invariant 1 — same images,
same order, every checkpoint, every condition) plus one backward pass on the probe-train set
(``ctx.probe_train_loader``) for the gradient-covariance rank.

Metrics
-------
* **Mean attention distance** per head per layer — the average spatial (grid) distance
  between a query patch and the patches it attends to (local vs global). Tracking it over
  training is heads specializing / despecializing — the most framework-relevant ViT metric.
  Scalar ``attn_mean_distance`` (per layer, head=h) + layer mean (head=None).
* **Attention entropy** per head — sharp vs diffuse routing. Scalar ``attn_entropy``.
* **CLS-token attention concentration** — where the trainable readout query points;
  reported as ``1 - normalized_entropy`` of the CLS row's attention over patches (higher =
  more concentrated). Scalar ``cls_attn_concentration``.
* **Per-layer / per-head gradient-covariance rank** — a backward pass on the probe-train set
  yields per-example gradients of each block's attention weights; the participation-ratio
  effective rank of their covariance is ``gradcov_rank`` (per layer, head=None) and the
  per-head slice is ``gradcov_rank`` (per layer, head=h). Direct test of per-module
  dimensionality reduction during fine-tuning. The full eigenspectrum goes to array
  ``gradcov_spectrum``.
* **CKA** (linear + RBF, implemented from scratch) of the CLS features vs init
  (``ctx.init_model_sd``-derived) and vs the previous checkpoint (``ctx.prev_model_sd``,
  when present): scalars ``cka_vs_init`` / ``cka_rbf_vs_init`` / ``cka_vs_prev`` /
  ``cka_rbf_vs_prev``. The probe CLS features themselves are stored to array
  ``probe_features`` so **cross-condition CKA at matched training fraction is computed
  post-hoc** by a separate step that reads the stored features (documented here; not done
  inline because it needs all conditions' features simultaneously).
* **Frozen-embedder assertion** — ``||E(t) - E(0)||`` over ``patch_embed.*`` + ``pos_embed``
  must be **exactly zero** (the input stage is excluded from the optimizer; §6.6 correctness
  check). Scalar ``frozen_embedder_drift``. A nonzero value is a BUG in the freezing and is
  logged loudly (summary flag + stderr).

Collector speed / memory cuts (L4-safe)
---------------------------------------
The naive implementation materialized the full softmax attention map for **every** probe image
over **all** 24 layers: ``[B=1024, heads=16, tokens=197, 197]`` is ~51 GB/layer, ~858 GB host
RAM across 24 layers if accumulated — it would OOM the L4 collector. This block now:

* **Chunks the probe set** (``ATTN_CHUNK``, default 32): attention statistics are computed
  chunk-by-chunk and accumulated incrementally. Mean attention distance, attention entropy and
  CLS concentration are all *means over images and heads*, so summing per-head numerators over
  chunks and dividing by the total image count at the end is **mathematically EXACT**, not an
  approximation. Peak tensor held is one chunk's maps: ``[32,16,197,197]`` ≈ 1.6 GB (freed each
  chunk). This is what removes the OOM.
* **Subsamples layers** (``INTERP_LAYER_STRIDE``, default 3): attention stats + CKA-relevant
  layer work run on every 3rd block (≈8 of 24). Depth profiles are therefore at *stride-3
  resolution*; the sampled block indices are recorded in the summary.
* **Caps the gradient-covariance backward** (``GRADCOV_MAX_IMAGES``, default 256) and only
  tracks the strided attention layers, so the (expensive) per-example backward pass touches a
  bounded, deterministic subset of the probe-train images and a bounded set of layers.

All subsampling is **deterministic** (fixed layer stride; a fixed prefix of the probe loaders,
which are themselves fixed-order per §4 invariant 1) and the chosen knobs + sampled layers are
recorded in the returned summary dict for reproducibility / cross-checkpoint comparability.

Attention-map capture (documented approach)
-------------------------------------------
timm's ``Attention`` (timm 1.0.x) uses **fused** ``F.scaled_dot_product_attention`` and does
**not** expose the softmax attention map. We therefore *recompute* the attention map on the
probe batch by registering a forward hook on each ``blocks.{i}.attn.qkv`` Linear to capture
its output ``[B, N, 3D]``, then reproduce timm's head split
(``reshape(B, N, 3, num_heads, head_dim)``), apply the module's ``q_norm`` / ``k_norm`` (Identity
for these ViT-L models, but applied generically for faithfulness) and ``scale``, and take
``softmax(q @ k^T)`` ourselves. This is mathematically identical to the fused kernel's implicit
map and needs no monkeypatch of the forward method. CLS token is index 0; patch tokens are
1..196 on a 14x14 grid (ViT-L/16 @ 224). Hooks are registered only on the **strided** blocks so
the other blocks' qkv outputs are never materialized into maps.

Robustness / RNG
----------------
This block reads model weights and runs forward/backward; it does not depend on the optimizer
variant (``ctx.opt`` is untouched), so it is identical for ``adamw`` and ``stable_evo``. The RBF
CKA bandwidth is the median heuristic (deterministic). All subsampling here is deterministic; a
``ctx.gen_interp`` generator (or a fresh one seeded from ``(seed, step)``) is resolved for
consistency with the other blocks but never drives a stochastic choice (we use fixed strides /
fixed loader prefixes). It never touches global RNG.
"""

from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project paths so preadapt_models / xai_train import cleanly when run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.dirname(_HERE)                  # xAI/py
_PROJECT_ROOT = os.path.join(_PY, "..", "..")  # repo root
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
for _p in (_PY, _PROJECT_ROOT, _PLANTCLEF):
    _ap = os.path.abspath(_p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

# Frozen-embedder assertion (single source of truth for the input stage).
try:
    from preadapt_models import assert_frozen_input_stage, input_stage_state, _INPUT_STAGE_KEYS
    _HAVE_PREADAPT = True
except Exception:  # pragma: no cover - environment guard
    _HAVE_PREADAPT = False
    _INPUT_STAGE_KEYS = ("patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed")

if TYPE_CHECKING:  # the dataclass is defined by the collector/driver; import lazily.
    from extractor._ctx import ExtractCtx  # noqa: F401


# ViT-L/16 architecture facts (verified against the checkpoints).
N_BLOCKS = 24
N_HEADS = 16
HEAD_DIM = 64
D_MODEL = 1024
GRID = 14          # 14x14 patch grid (224 / 16)
N_PATCH = GRID * GRID  # 196
# Token layout: index 0 = CLS, indices 1..196 = patches in row-major grid order.

# --- collector speed / memory cuts (L4-safe; see module docstring) ----------------------
#: Probe images processed per forward chunk for attention statistics. Per-chunk peak attention
#: tensor is [ATTN_CHUNK, N_HEADS, N_TOKENS, N_TOKENS] ≈ 1.6 GB at 32; the full probe-set map is
#: NEVER materialized. Means are accumulated incrementally (exact).
ATTN_CHUNK = 32
#: Run attention statistics + per-layer interp work on every Nth block (≈8 of 24 at stride 3).
#: Depth profiles are reported at this stride resolution; sampled blocks recorded in the summary.
INTERP_LAYER_STRIDE = 3
#: Cap on probe-train images feeding the (expensive) per-example gradient-covariance backward.
GRADCOV_MAX_IMAGES = 256

# Cap features used for CKA to keep the N x N Gram matrices tractable.
CKA_MAX_EXAMPLES = 2048


def _strided_blocks(n_blocks: int = N_BLOCKS, stride: int = INTERP_LAYER_STRIDE) -> List[int]:
    """Deterministic strided block indices ``[0, stride, 2*stride, ...]`` within ``n_blocks``.
    This is the fixed layer subset used for attention stats + gradcov so the depth profile is
    comparable across every checkpoint and condition."""
    stride = max(1, int(stride))
    return list(range(0, int(n_blocks), stride))


# =============================================================================
# 1. Attention-map capture (recompute from a qkv forward hook; fused attn hides the map)
# =============================================================================

class _QKVCapture:
    """Registers forward hooks on the SELECTED ``blocks.{i}.attn.qkv`` Linears and recomputes the
    softmax attention map per head, since the fused kernel does not expose it.

    Only the blocks in ``block_indices`` are hooked (layer subsampling), so non-selected blocks'
    qkv outputs are never turned into [B,h,N,N] maps. After a forward pass on a batch,
    :attr:`maps` holds, per selected block index, a tensor ``[B, n_heads, N, N]`` of attention
    probabilities (rows = query token, cols = key token).
    """

    def __init__(self, model: nn.Module, block_indices: Optional[List[int]] = None):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.maps: Dict[int, torch.Tensor] = {}
        self._attn_modules: Dict[int, nn.Module] = {}
        if block_indices is None:
            block_indices = list(range(len(model.blocks)))
        self.block_indices = list(block_indices)
        sel = set(self.block_indices)
        for i, blk in enumerate(model.blocks):
            if i not in sel:
                continue
            attn = blk.attn
            self._attn_modules[i] = attn
            self.handles.append(attn.qkv.register_forward_hook(self._make_hook(i, attn)))

    def _make_hook(self, block_idx: int, attn: nn.Module):
        def hook(module: nn.Module, inputs, output: torch.Tensor) -> None:
            # output: [B, N, 3D] (the fused qkv projection of the LayerNorm'd input).
            B, N, _ = output.shape
            n_heads = getattr(attn, "num_heads", N_HEADS)
            head_dim = getattr(attn, "head_dim", D_MODEL // n_heads)
            qkv = output.reshape(B, N, 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _v = qkv.unbind(0)  # each [B, n_heads, N, head_dim]
            # Apply the module's q/k norm (Identity for ViT-L, but faithful in general).
            q = attn.q_norm(q) if hasattr(attn, "q_norm") else q
            k = attn.k_norm(k) if hasattr(attn, "k_norm") else k
            scale = getattr(attn, "scale", head_dim ** -0.5)
            attn_logits = (q.float() * scale) @ k.float().transpose(-2, -1)  # [B, h, N, N]
            self.maps[block_idx] = attn_logits.softmax(dim=-1).detach()
        return hook

    def clear_maps(self) -> None:
        """Drop the captured maps (call after consuming a chunk so the ~1.6 GB tensors are freed
        before the next chunk's forward pass repopulates them)."""
        self.maps.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "_QKVCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def _grid_distance_matrix() -> torch.Tensor:
    """Patch-to-patch Euclidean distance on the 14x14 grid, ``[N_PATCH, N_PATCH]`` (float)."""
    coords = torch.stack([
        torch.arange(N_PATCH) // GRID,  # row
        torch.arange(N_PATCH) % GRID,   # col
    ], dim=1).float()                   # [N_PATCH, 2]
    diff = coords[:, None, :] - coords[None, :, :]
    return diff.norm(dim=-1)            # [N_PATCH, N_PATCH]


@torch.no_grad()
def _attention_chunk_sums(maps: Dict[int, torch.Tensor], dist: torch.Tensor
                          ) -> Dict[int, Dict[str, np.ndarray]]:
    """From one chunk's captured attention maps compute, per block, per-head, the **SUMS over
    images** (NOT means) of:
      * mean attention distance over PATCH query->key pairs (CLS excluded from both axes),
        already averaged over query patches (the per-image scalar), then summed over images,
      * attention entropy (full row, natural log, averaged over query tokens) summed over images,
      * CLS-token attention concentration = ``1 - H(CLS row over patches)/log(N_patch)`` summed
        over images.

    Returns ``{block: {"mean_distance": [h], "entropy": [h], "cls_concentration": [h], "n": int}}``
    where the three arrays are per-head SUMS over the ``n`` images in this chunk. Summing these
    across chunks and dividing by the total image count yields the EXACT per-head means (the same
    value a single full-batch pass would produce), because every reduction over images is a plain
    arithmetic mean.
    """
    out: Dict[int, Dict[str, np.ndarray]] = {}
    if not maps:
        return out
    dist = dist.to(next(iter(maps.values())).device)
    for blk, A in maps.items():
        # A: [B, h, N, N]; token 0 = CLS, 1..N_patch = patches.
        B, H, N, _ = A.shape
        patch = A[:, :, 1:, 1:]                       # [B, h, P, P] patch->patch
        # mean attention distance: sum_j A[i,j] * dist[i,j], averaged over query patches i.
        # patch rows are not renormalized (they exclude CLS+self mass); this is the standard
        # ViT mean-attention-distance using the patch-restricted weights as-is.
        md = (patch * dist[None, None, :, :]).sum(dim=-1)   # [B, h, P]
        # per-image mean over query patches, then SUM over images in the chunk.
        mean_distance_sum = md.mean(dim=2).sum(dim=0)        # [h]
        # entropy over the FULL attention row (all N keys), averaged over query tokens per image,
        # then summed over images.
        ent = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(dim=-1)  # [B, h, N]
        entropy_sum = ent.mean(dim=2).sum(dim=0)             # [h]
        # CLS concentration: 1 - normalized entropy of the CLS query row over patch keys (per
        # image), then summed over images.
        cls_row = A[:, :, 0, 1:]                              # [B, h, P]
        cls_row = cls_row / cls_row.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cls_ent = -(cls_row.clamp_min(1e-12) * cls_row.clamp_min(1e-12).log()).sum(dim=-1)
        cls_conc = 1.0 - cls_ent / math.log(N_PATCH)         # [B, h]
        cls_conc_sum = cls_conc.sum(dim=0)                   # [h]
        out[blk] = {
            "mean_distance": mean_distance_sum.cpu().numpy().astype("float64"),
            "entropy": entropy_sum.cpu().numpy().astype("float64"),
            "cls_concentration": cls_conc_sum.cpu().numpy().astype("float64"),
            "n": int(B),
        }
    return out


@torch.no_grad()
def _attention_statistics_chunked(model: nn.Module, loader, device, dist: torch.Tensor,
                                  block_indices: List[int], attn_chunk: int,
                                  ) -> Tuple[Dict[int, Dict[str, np.ndarray]], int, int]:
    """Compute per-head attention statistics over the WHOLE probe-eval loader by chunking and
    accumulating EXACT per-head means, never holding more than one chunk's [b,h,N,N] maps.

    The probe loader's batches are re-chunked to ``attn_chunk`` images so peak memory is bounded
    regardless of the loader's own batch size (a probe loader batch may be 1024 images, which is
    exactly the tensor we must avoid materializing as a map). Hooks are registered only on
    ``block_indices``.

    Returns ``(stats, n_images, n_chunks)`` where ``stats`` is
    ``{block: {"mean_distance":[h], "entropy":[h], "cls_concentration":[h]}}`` of per-head MEANS.
    """
    model.eval()
    attn_chunk = max(1, int(attn_chunk))
    # Per-block running sums + a single running image count (same images feed every block).
    sums: Dict[int, Dict[str, np.ndarray]] = {}
    total_n = 0
    n_chunks = 0

    with _QKVCapture(model, block_indices=block_indices) as cap:
        for batch in loader:
            images = batch[0]
            # Re-chunk this loader batch to attn_chunk so peak map memory is bounded by the
            # chunk, not the loader's (possibly huge) batch size.
            for start in range(0, images.shape[0], attn_chunk):
                sub = images[start:start + attn_chunk].to(device, non_blocking=True)
                cap.clear_maps()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(sub)
                chunk = _attention_chunk_sums(cap.maps, dist)
                cap.clear_maps()   # free the ~1.6 GB chunk maps before the next forward
                del sub
                if not chunk:
                    continue
                # n is identical across blocks within a chunk; track it once.
                this_n = next(iter(chunk.values()))["n"]
                for blk, s in chunk.items():
                    acc = sums.setdefault(blk, {
                        "mean_distance": np.zeros_like(s["mean_distance"]),
                        "entropy": np.zeros_like(s["entropy"]),
                        "cls_concentration": np.zeros_like(s["cls_concentration"]),
                    })
                    acc["mean_distance"] += s["mean_distance"]
                    acc["entropy"] += s["entropy"]
                    acc["cls_concentration"] += s["cls_concentration"]
                total_n += this_n
                n_chunks += 1

    stats: Dict[int, Dict[str, np.ndarray]] = {}
    if total_n > 0:
        for blk, acc in sums.items():
            stats[blk] = {
                "mean_distance": (acc["mean_distance"] / total_n).astype("float32"),
                "entropy": (acc["entropy"] / total_n).astype("float32"),
                "cls_concentration": (acc["cls_concentration"] / total_n).astype("float32"),
            }
    return stats, total_n, n_chunks


# =============================================================================
# 2. CKA (linear + RBF), implemented from scratch
# =============================================================================

def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix: ``H K H`` with ``H = I - 1/n``."""
    n = K.shape[0]
    unit = torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    H = I - unit
    return H @ K @ H


def _hsic(Kc: torch.Tensor, Lc: torch.Tensor) -> float:
    """HSIC of two centered Gram matrices: ``<Kc, Lc>_F`` (a Frobenius inner product)."""
    return float((Kc * Lc).sum().item())


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between feature matrices ``X`` ([n, d1]) and ``Y`` ([n, d2]).

    CKA = HSIC(K, L) / sqrt(HSIC(K,K) HSIC(L,L)) with linear kernels K = X X^T, L = Y Y^T.
    Invariant to orthogonal transforms and isotropic scaling; in [0, 1]. From scratch.
    """
    X = X.float()
    Y = Y.float()
    K = X @ X.t()
    L = Y @ Y.t()
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    num = _hsic(Kc, Lc)
    den = math.sqrt(max(_hsic(Kc, Kc), 1e-12) * max(_hsic(Lc, Lc), 1e-12))
    return num / den if den > 0 else float("nan")


def _rbf_gram(X: torch.Tensor) -> torch.Tensor:
    """RBF Gram matrix with the median-distance bandwidth heuristic (deterministic)."""
    X = X.float()
    sq = (X * X).sum(dim=1, keepdim=True)
    d2 = (sq + sq.t() - 2.0 * (X @ X.t())).clamp_min(0.0)
    n = X.shape[0]
    if n > 1:
        med = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0, device=X.device)
        sigma2 = float(med.item()) if float(med.item()) > 0 else 1.0
    else:
        sigma2 = 1.0
    return torch.exp(-d2 / (2.0 * sigma2))


def rbf_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """RBF (Gaussian-kernel) CKA between ``X`` and ``Y`` with the median bandwidth. From scratch."""
    K = _rbf_gram(X)
    L = _rbf_gram(Y)
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    num = _hsic(Kc, Lc)
    den = math.sqrt(max(_hsic(Kc, Kc), 1e-12) * max(_hsic(Lc, Lc), 1e-12))
    return num / den if den > 0 else float("nan")


# =============================================================================
# 3. Feature extraction (reuse xai_train.extract_features semantics)
# =============================================================================

@torch.no_grad()
def _extract_cls_features(model: nn.Module, loader, device, max_examples: int,
                          chunk: int = ATTN_CHUNK) -> torch.Tensor:
    """CLS features ``[n, D]`` from ``model.forward_features`` over ``loader`` (capped).

    Mirrors ``xai_train.extract_features`` (same penultimate CLS readout) but with a cap so
    the CKA Gram matrices stay tractable, and with the loader batches re-chunked to ``chunk``
    images so peak activation memory is bounded (a probe loader batch can be 1024 images). CLS
    features are pooled to ``[b, D]`` per image, so feature extraction is cheap memory-wise; the
    chunking simply mirrors the attention path and keeps the forward activations small. Imported
    lazily so this module loads without the full training stack present.
    """
    model.eval()
    chunk = max(1, int(chunk))
    feats: List[torch.Tensor] = []
    n = 0
    for batch in loader:
        images = batch[0]
        for start in range(0, images.shape[0], chunk):
            sub = images[start:start + chunk].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                f = model.forward_features(sub)
            feats.append(f.float().cpu())
            n += f.shape[0]
            del sub
            if n >= max_examples:
                break
        if n >= max_examples:
            break
    if not feats:
        return torch.empty(0, D_MODEL)
    out = torch.cat(feats, dim=0)
    return out[:max_examples]


def _features_from_state_dict(model: nn.Module, state_dict: dict, loader, device,
                              max_examples: int) -> torch.Tensor:
    """Load ``state_dict`` into a temporary copy of the model's weights, extract CLS features,
    then restore the current weights. Used for vs-init / vs-prev CKA without mutating ``model``.
    """
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        model.load_state_dict(state_dict, strict=False)
        feats = _extract_cls_features(model, loader, device, max_examples)
    finally:
        model.load_state_dict(saved, strict=False)
    return feats


# =============================================================================
# 4. Per-layer / per-head gradient-covariance rank
# =============================================================================

def _participation_ratio(eigs: np.ndarray) -> float:
    """Effective rank as participation ratio of nonneg eigenvalues: ``(sum)^2 / sum(sq)``."""
    e = np.clip(eigs, 0.0, None)
    num = float(e.sum()) ** 2
    den = float((e * e).sum())
    return num / den if den > 0 else 0.0


def _gradcov_rank(model: nn.Module, criterion, loader, device,
                  max_examples: int, block_indices: Optional[List[int]] = None,
                  ) -> Tuple[Dict[str, float], Dict[str, np.ndarray],
                             Dict[Tuple[str, int], float]]:
    """Per-example gradients of the SELECTED blocks' attention ``qkv``+``proj`` weights -> per-layer
    gradient covariance; return its participation-ratio effective rank and full spectrum,
    plus a per-head rank (slicing the qkv gradient by head rows).

    Only the layers in ``block_indices`` are tracked (reusing the attention layer stride) and at
    most ``max_examples`` probe-train images feed the backward pass — both bound the cost of the
    per-example backward sweep. The image subset is the FIXED first ``max_examples`` images of the
    fixed-order probe-train loader (deterministic; the loader order is a §4 invariant), so the
    same examples feed every checkpoint and condition.

    We accumulate one flattened gradient vector per example (microbatch size 1) for each
    tracked attention layer, form the ``[n_examples, n_params_layer]`` matrix, and compute the
    eigenspectrum of its (example) covariance via the small Gram matrix
    ``G G^T / n`` ([n, n]) — whose nonzero eigenvalues equal those of the parameter
    covariance, so the effective rank is exact and cheap (n << n_params).

    Returns:
      layer_rank : {layer_name -> effective rank}
      layer_spec : {layer_name -> sorted eigenvalue spectrum (np.ndarray)}
      head_rank  : {(layer_name, head) -> effective rank from the head's qkv gradient rows}
    """
    model.eval()  # we want gradients but not dropout/BN noise on the probe set
    for p in model.parameters():
        p.requires_grad_(p.requires_grad)  # keep the trainer's grad flags

    if block_indices is None:
        block_indices = list(range(len(model.blocks)))
    block_indices = [i for i in block_indices if 0 <= i < len(model.blocks)]

    # Which parameters to track per (selected) attention layer (qkv.weight + proj.weight).
    layer_param_names = {}
    for i in block_indices:
        layer = f"blocks.{i}.attn"
        layer_param_names[layer] = [f"blocks.{i}.attn.qkv.weight",
                                    f"blocks.{i}.attn.proj.weight"]
    name_to_param = dict(model.named_parameters())

    # Per-layer list of per-example flattened gradient vectors, and per-head (qkv only).
    grads: Dict[str, List[np.ndarray]] = {ly: [] for ly in layer_param_names}
    head_grads: Dict[Tuple[str, int], List[np.ndarray]] = {}

    n_seen = 0
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[-1].to(device, non_blocking=True)
        for bi in range(images.shape[0]):
            if n_seen >= max_examples:
                break
            model.zero_grad(set_to_none=True)
            img = images[bi:bi + 1]
            tgt = targets[bi:bi + 1]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(img)
                loss = criterion(out, tgt)
            loss.backward()
            for layer, pnames in layer_param_names.items():
                vecs = []
                for pn in pnames:
                    p = name_to_param.get(pn)
                    if p is None or p.grad is None:
                        continue
                    vecs.append(p.grad.detach().float().reshape(-1).cpu())
                if vecs:
                    grads[layer].append(torch.cat(vecs).numpy().astype("float32"))
                # per-head: slice qkv.weight grad rows for head h (q,k,v rows each d_h).
                qkv_name = pnames[0]
                qp = name_to_param.get(qkv_name)
                if qp is not None and qp.grad is not None:
                    g = qp.grad.detach().float().cpu()  # [3D, D]
                    for h in range(N_HEADS):
                        r = h * HEAD_DIM
                        rows = torch.cat([
                            g[r:r + HEAD_DIM],
                            g[D_MODEL + r:D_MODEL + r + HEAD_DIM],
                            g[2 * D_MODEL + r:2 * D_MODEL + r + HEAD_DIM],
                        ], dim=0).reshape(-1).numpy().astype("float32")
                        head_grads.setdefault((layer, h), []).append(rows)
            n_seen += 1
        if n_seen >= max_examples:
            break
    model.zero_grad(set_to_none=True)

    def _rank_from_grad_list(vlist: List[np.ndarray]) -> Tuple[float, np.ndarray]:
        if len(vlist) < 2:
            return float("nan"), np.zeros(0, dtype="float32")
        G = np.stack(vlist, axis=0)              # [n, p]
        G = G - G.mean(axis=0, keepdims=True)    # center over examples
        n = G.shape[0]
        gram = (G @ G.T) / max(n - 1, 1)         # [n, n], same nonzero spectrum as cov
        try:
            eigs = np.linalg.eigvalsh(0.5 * (gram + gram.T))
        except np.linalg.LinAlgError:
            return float("nan"), np.zeros(0, dtype="float32")
        eigs = np.sort(np.clip(eigs, 0.0, None))[::-1]
        return _participation_ratio(eigs), eigs.astype("float32")

    layer_rank: Dict[str, float] = {}
    layer_spec: Dict[str, np.ndarray] = {}
    for layer, vlist in grads.items():
        r, spec = _rank_from_grad_list(vlist)
        layer_rank[layer] = r
        layer_spec[layer] = spec

    head_rank: Dict[Tuple[str, int], float] = {}
    for key, vlist in head_grads.items():
        r, _spec = _rank_from_grad_list(vlist)
        head_rank[key] = r

    return layer_rank, layer_spec, head_rank


# =============================================================================
# 5. Main entry point
# =============================================================================

def extract(ctx: "ExtractCtx") -> Dict[str, object]:
    """Run §6.6 ViT-interpretability extraction on the fixed probe set.

    Writes attention statistics, gradient-covariance ranks, CKA-vs-init/prev, the
    frozen-embedder assertion (scalars) and the probe CLS features + gradcov spectra (arrays).
    Returns a small summary dict (including the frozen-embedder drift, a ``bug`` flag, and the
    subsampling knobs + sampled layers for reproducibility).
    """
    model = ctx.model
    cond = ctx.condition
    run_id = ctx.run_id
    step = ctx.step
    wt = ctx.wall_time
    device = ctx.device
    criterion = ctx.criterion
    eval_loader = ctx.probe_eval_loader
    train_loader = getattr(ctx, "probe_train_loader", None)

    model.eval()
    summary: Dict[str, object] = {"block": "block_interp"}

    # Deterministic layer subsample reused by attention stats + gradcov.
    sampled_blocks = _strided_blocks(len(model.blocks), INTERP_LAYER_STRIDE)
    # Record the subsampling knobs up front (present even if a later stage errors).
    summary["attn_chunk"] = int(ATTN_CHUNK)
    summary["interp_layer_stride"] = int(INTERP_LAYER_STRIDE)
    summary["gradcov_max_images"] = int(GRADCOV_MAX_IMAGES)
    summary["cka_max_examples"] = int(CKA_MAX_EXAMPLES)
    summary["sampled_layers"] = [int(i) for i in sampled_blocks]
    summary["n_sampled_layers"] = int(len(sampled_blocks))
    summary["n_blocks_total"] = int(len(model.blocks))

    # ---------------------------------------------------------------------
    # (0) Frozen-embedder assertion (correctness check; must be exactly 0).
    # ---------------------------------------------------------------------
    init_sd = getattr(ctx, "init_model_sd", None)
    frozen_drift = float("nan")
    if init_sd is not None:
        cur = model.state_dict()
        total_sq = 0.0
        for k in _INPUT_STAGE_KEYS:
            if k in init_sd and k in cur:
                e0 = init_sd[k].detach().to("cpu", dtype=torch.float64)
                et = cur[k].detach().to("cpu", dtype=torch.float64)
                if e0.shape == et.shape:
                    total_sq += float(((et - e0) ** 2).sum().item())
        frozen_drift = float(total_sq ** 0.5)
        ctx.scalar.add(cond, run_id, step, wt, quantity="frozen_embedder_drift",
                       value=frozen_drift, layer=None, head=None)
        if frozen_drift > 1e-6:
            # LOUD: the input stage moved — the freezing is broken (briefing escalation).
            msg = (f"[block_interp] FROZEN-EMBEDDER BUG: ||E(t)-E(0)|| = {frozen_drift:.3e} "
                   f"!= 0 at run_id={run_id} step={step}. The input stage is NOT frozen; "
                   f"every cross-condition contrast is confounded. ESCALATE.")
            print(msg, file=sys.stderr, flush=True)
            summary["frozen_embedder_bug"] = True
    summary["frozen_embedder_drift"] = frozen_drift

    # ---------------------------------------------------------------------
    # (1) Attention statistics over the WHOLE probe-eval set, chunked + accumulated EXACTLY,
    #     on the strided layer subset. Peak map tensor held: one [ATTN_CHUNK,h,N,N] chunk.
    # ---------------------------------------------------------------------
    dist = _grid_distance_matrix()
    n_attn_images = 0
    n_attn_chunks = 0
    try:
        stats, n_attn_images, n_attn_chunks = _attention_statistics_chunked(
            model, eval_loader, device, dist, sampled_blocks, ATTN_CHUNK)
        for blk, s in stats.items():
            layer = f"blocks.{blk}.attn"
            ctx.array.put(group="attn_mean_distance_heads", step=step,
                          array=s["mean_distance"], layer=layer)
            ctx.array.put(group="attn_entropy_heads", step=step,
                          array=s["entropy"], layer=layer)
            for h in range(len(s["mean_distance"])):
                ctx.scalar.add(cond, run_id, step, wt, quantity="attn_mean_distance",
                               value=float(s["mean_distance"][h]), layer=layer, head=h)
                ctx.scalar.add(cond, run_id, step, wt, quantity="attn_entropy",
                               value=float(s["entropy"][h]), layer=layer, head=h)
                ctx.scalar.add(cond, run_id, step, wt, quantity="cls_attn_concentration",
                               value=float(s["cls_concentration"][h]), layer=layer, head=h)
            ctx.scalar.add(cond, run_id, step, wt, quantity="attn_mean_distance_layer_mean",
                           value=float(np.mean(s["mean_distance"])), layer=layer, head=None)
            ctx.scalar.add(cond, run_id, step, wt, quantity="attn_entropy_layer_mean",
                           value=float(np.mean(s["entropy"])), layer=layer, head=None)
    except Exception as e:  # never kill extraction on an attention hiccup
        print(f"[block_interp] attention statistics skipped: {e}", file=sys.stderr)
    summary["attn_images"] = int(n_attn_images)
    summary["attn_chunks"] = int(n_attn_chunks)

    # ---------------------------------------------------------------------
    # (2) Probe CLS features (stored for CKA + cross-condition post-hoc) and CKA.
    #     Feature extraction is chunked the same way (peak activations bounded).
    # ---------------------------------------------------------------------
    cur_feats = _extract_cls_features(model, eval_loader, device, CKA_MAX_EXAMPLES)
    if cur_feats.numel() > 0:
        # Store the probe features so cross-condition CKA at matched fraction is possible
        # later (documented: a separate post-hoc step reads all conditions' stored features).
        ctx.array.put(group="probe_features", step=step,
                      array=cur_feats.numpy().astype("float32"))

        if init_sd is not None:
            try:
                init_feats = _features_from_state_dict(model, init_sd, eval_loader, device,
                                                       CKA_MAX_EXAMPLES)
                if init_feats.numel() > 0:
                    n = min(cur_feats.shape[0], init_feats.shape[0])
                    ctx.scalar.add(cond, run_id, step, wt, quantity="cka_vs_init",
                                   value=linear_cka(cur_feats[:n], init_feats[:n]))
                    ctx.scalar.add(cond, run_id, step, wt, quantity="cka_rbf_vs_init",
                                   value=rbf_cka(cur_feats[:n], init_feats[:n]))
            except Exception as e:  # never kill extraction on a CKA hiccup
                print(f"[block_interp] cka_vs_init skipped: {e}", file=sys.stderr)

        prev_sd = getattr(ctx, "prev_model_sd", None)
        if prev_sd is not None:
            try:
                prev_feats = _features_from_state_dict(model, prev_sd, eval_loader, device,
                                                       CKA_MAX_EXAMPLES)
                if prev_feats.numel() > 0:
                    n = min(cur_feats.shape[0], prev_feats.shape[0])
                    ctx.scalar.add(cond, run_id, step, wt, quantity="cka_vs_prev",
                                   value=linear_cka(cur_feats[:n], prev_feats[:n]))
                    ctx.scalar.add(cond, run_id, step, wt, quantity="cka_rbf_vs_prev",
                                   value=rbf_cka(cur_feats[:n], prev_feats[:n]))
            except Exception as e:
                print(f"[block_interp] cka_vs_prev skipped: {e}", file=sys.stderr)
    summary["n_probe_features"] = int(cur_feats.shape[0]) if cur_feats.numel() else 0

    # ---------------------------------------------------------------------
    # (3) Per-layer / per-head gradient-covariance rank (backward on probe-train).
    #     Bounded to GRADCOV_MAX_IMAGES images and the strided layer subset.
    # ---------------------------------------------------------------------
    gradcov_loader = train_loader if train_loader is not None else eval_loader
    try:
        layer_rank, layer_spec, head_rank = _gradcov_rank(
            model, criterion, gradcov_loader, device, GRADCOV_MAX_IMAGES,
            block_indices=sampled_blocks)
        for layer, r in layer_rank.items():
            if np.isfinite(r):
                ctx.scalar.add(cond, run_id, step, wt, quantity="gradcov_rank",
                               value=float(r), layer=layer, head=None)
            spec = layer_spec.get(layer)
            if spec is not None and spec.size:
                ctx.array.put(group="gradcov_spectrum", step=step, array=spec, layer=layer)
        for (layer, h), r in head_rank.items():
            if np.isfinite(r):
                ctx.scalar.add(cond, run_id, step, wt, quantity="gradcov_rank",
                               value=float(r), layer=layer, head=h)
        finite_ranks = [r for r in layer_rank.values() if np.isfinite(r)]
        summary["gradcov_rank_mean"] = float(np.mean(finite_ranks)) if finite_ranks else float("nan")
        summary["gradcov_n_layers"] = int(len(layer_rank))
    except Exception as e:
        print(f"[block_interp] gradcov_rank skipped: {e}", file=sys.stderr)
        summary["gradcov_rank_mean"] = float("nan")
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()

    return summary
