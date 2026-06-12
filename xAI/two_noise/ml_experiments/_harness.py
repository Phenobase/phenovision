"""Shared harness for the real-model optimizer benchmarks (analysis_plan.md §2.1, §2.2).

This module is the single source of truth for *how* a real-model experiment is built and run,
so `benchmarks.py` (§2.1, one-config CLI for a SLURM array) and `alpha_vs_batch.py` (§2.2,
the α×batch sweep) share identical model/data/optimizer/training code and only differ in the
configs they iterate over.

Four entry points:
  - make_model(name, num_classes/vocab_size) -> nn.Module
  - make_data(dataset, batch_size, ...)      -> (train_loader, val_loader, meta)
  - make_optimizer(name, params, alpha, lr, **kw) -> torch.optim.Optimizer
  - train_eval(model, optimizer, train_loader, val_loader, ...) -> dict (tidy per-step records
        + final val metrics + per-step time/peak memory)

Design choices (see CONVENTIONS.md):
  - α (`precond_power`) is the swept exponent: 0=SGD-ish, 0.5=whitening (vanilla SOAP),
    1.0=full inverse / natural gradient. At α=1 SOAPFullPower needs relative damping ~1e-2 and
    a ~10x smaller lr than AdamW — make_optimizer encodes those defaults but lets the caller
    override.
  - RNG discipline: an explicit torch.Generator is threaded through data shuffling and any
    stochastic op so smoke tests are bit-reproducible.
  - The training loop borrows the callback idea from xAI/py/xai_engine.py
    (train_one_epoch_with_callbacks): periodic eval, per-step tidy logging. It is rewritten
    here (not imported) because xai_engine is ViT/timm-specific and pulls PlantCLEF utils.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default on-disk locations (kept under the subproject; runs/ and data/ are gitignored).
DATA_DIR = Path(os.environ.get("TWO_NOISE_DATA_DIR", ROOT / "data"))
RUNS_DIR = ROOT / "runs"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# nanoGPT config presets (~10-30M params). These are the LM substrate for §2.1.
_NANOGPT_PRESETS = {
    # ~12M non-embedding params at vocab 4096 — small enough for fast sweeps.
    "nanogpt": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.0, bias=False),
    # ~30M, for the larger end of the §2.1 LM scale.
    "nanogpt_m": dict(n_layer=8, n_head=8, n_embd=512, block_size=256, dropout=0.0, bias=False),
}


def make_model(name: str, num_classes: int = 100, vocab_size: int = 4096) -> nn.Module:
    """Build a model by short name.

    Args:
        name: 'vit_s' | 'vit_b' (vision, timm) | 'nanogpt' | 'nanogpt_m' (LM, _vendor/nanogpt).
        num_classes: vision head size (e.g. 100 for CIFAR-100, 200 for Tiny-ImageNet).
        vocab_size: LM vocab (must match the tokenizer used by make_data for LM datasets).

    Returns:
        An nn.Module. Vision models take (B,3,224,224) -> (B,num_classes) logits. The nanoGPT
        model takes (B,T) token ids and an optional `targets` -> (logits, loss); train_eval
        detects the LM path by `isinstance` on the GPT class.
    """
    name = name.lower()
    if name in ("vit_s", "vit_small", "vit_small_patch16_224"):
        import timm
        return timm.create_model("vit_small_patch16_224", num_classes=num_classes, pretrained=False)
    if name in ("vit_b", "vit_base", "vit_base_patch16_224"):
        import timm
        return timm.create_model("vit_base_patch16_224", num_classes=num_classes, pretrained=False)
    if name in _NANOGPT_PRESETS:
        sys.path.insert(0, str(ROOT / "_vendor"))
        from nanogpt.model import GPT, GPTConfig
        preset = dict(_NANOGPT_PRESETS[name])
        preset["vocab_size"] = vocab_size
        return GPT(GPTConfig(**preset))
    if name in ("tiny_vision", "tiny_cnn"):
        # Tiny conv net for CPU smoke tests only (NOT a §2.1 model). Has 2D weight tensors so
        # SOAPFullPower's Kronecker path is exercised. Expects (B,3,32,32).
        return _TinyVision(num_classes=num_classes)
    raise ValueError(f"unknown model name {name!r}")


class _TinyVision(nn.Module):
    """Minimal conv+linear net for smoke tests (2D weights exercise the SOAP precond path)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)  # 32->16
        self.fc1 = nn.Linear(8 * 16 * 16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def is_lm_model(model: nn.Module) -> bool:
    """True if `model` is the nanoGPT LM (forward signature (idx, targets) -> (logits, loss))."""
    return type(model).__name__ == "GPT"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataMeta:
    dataset: str
    task: str               # "vision" | "lm"
    num_classes: int = 0    # vision
    vocab_size: int = 0     # lm
    block_size: int = 0     # lm sequence length
    n_train: int = 0
    n_val: int = 0


def _vision_transforms(train: bool, img_size: int = 224):
    from torchvision import transforms
    # CIFAR-style normalization; images upsampled to 224 for ViT patch16.
    mean = (0.5071, 0.4866, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    if train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def make_data(
    dataset: str,
    batch_size: int,
    num_workers: int = 4,
    img_size: int = 224,
    block_size: int = 256,
    generator: Optional[torch.Generator] = None,
    data_dir: Optional[Path] = None,
    download: bool = True,
    pin_memory: bool = True,
    synthetic_n: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, DataMeta]:
    """Build (train_loader, val_loader, meta) for a named dataset.

    Supported:
        'cifar100'       — torchvision CIFAR-100 (auto-download). WORKING end-to-end.
        'tiny_imagenet'  — Tiny-ImageNet-200 ImageFolder. Needs manual download (see below).
        'tinystories'    — HF 'roneneldan/TinyStories', GPT-2 BPE tokenized. Needs `datasets`
                           + a one-time tokenize-to-.bin step (see below).
        'synthetic_vision'/'synthetic_lm' — tiny in-memory data for CPU smoke tests.

    Manual-setup datasets (documented so the caller knows what to provision):
        Tiny-ImageNet: download from http://cs231n.stanford.edu/tiny-imagenet-200.zip,
            unzip under {data_dir}/tiny-imagenet-200/ (train/ and val/ ImageFolder dirs;
            the standard val/ needs reorganizing into class subdirs — see prepare note in
            the function body). 200 classes, 64x64 (upsampled to img_size here).
        TinyStories: requires the `datasets` package (NOT in the two_noise env yet:
            `mamba run -n two_noise pip install datasets tiktoken`). The loader tokenizes a
            subset with the GPT-2 BPE (tiktoken 'gpt2', vocab 50257) and caches token .bin
            files under {data_dir}/tinystories/. Pass synthetic_n>0 / a subset for smoke.

    Args:
        generator: torch.Generator for reproducible shuffling (RNG discipline, CONVENTIONS §5).
        synthetic_n: if >0 and dataset starts with 'synthetic', size of the in-memory set.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset.lower()

    if dataset in ("synthetic_vision", "synthetic_lm"):
        return _make_synthetic(dataset, batch_size, block_size, generator, synthetic_n or 64)

    if dataset == "cifar100":
        return _make_cifar100(batch_size, num_workers, img_size, generator, data_dir,
                              download, pin_memory)

    if dataset in ("tiny_imagenet", "tiny-imagenet", "tinyimagenet"):
        return _make_tiny_imagenet(batch_size, num_workers, img_size, generator, data_dir,
                                   pin_memory)

    if dataset == "tinystories":
        return _make_tinystories(batch_size, block_size, generator, data_dir, num_workers,
                                 pin_memory)

    raise ValueError(f"unknown dataset {dataset!r}")


def _loader(ds, batch_size, shuffle, num_workers, generator, pin_memory, drop_last):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        generator=generator if shuffle else None, pin_memory=pin_memory,
        drop_last=drop_last, persistent_workers=(num_workers > 0),
    )


def ensure_dataset(dataset: str, data_dir: str = "data") -> None:
    """Stage a dataset ONCE (download+extract if absent), single-process. Call this before
    launching parallel SLURM-array tasks so they find the data present and read it read-only,
    instead of each racing to (re-)extract into the same directory."""
    make_data(dataset, batch_size=2, num_workers=0, data_dir=data_dir, download=True)


def _cifar100_present(data_dir) -> bool:
    base = Path(data_dir) / "cifar-100-python"
    return (base / "train").exists() and (base / "test").exists()


def _make_cifar100(batch_size, num_workers, img_size, generator, data_dir, download, pin_memory):
    from torchvision.datasets import CIFAR100
    # Download/extract ONLY if the data is genuinely absent. Otherwise read-only — so parallel
    # SLURM-array tasks never re-extract into the same dir and race (the Errno-5 corruption).
    # Stage once up front with ensure_dataset() before launching an array.
    dl = bool(download) and not _cifar100_present(data_dir)
    train = CIFAR100(root=str(data_dir), train=True, download=dl,
                     transform=_vision_transforms(True, img_size))
    val = CIFAR100(root=str(data_dir), train=False, download=dl,
                   transform=_vision_transforms(False, img_size))
    meta = DataMeta("cifar100", "vision", num_classes=100,
                    n_train=len(train), n_val=len(val))
    return (
        _loader(train, batch_size, True, num_workers, generator, pin_memory, drop_last=True),
        _loader(val, batch_size, False, num_workers, generator, pin_memory, drop_last=False),
        meta,
    )


def _make_tiny_imagenet(batch_size, num_workers, img_size, generator, data_dir, pin_memory):
    """Tiny-ImageNet-200 via ImageFolder. Requires manual download.

    Expected layout after setup:
        {data_dir}/tiny-imagenet-200/train/<wnid>/images/*.JPEG
        {data_dir}/tiny-imagenet-200/val_organized/<wnid>/*.JPEG
    The raw download's val/ is a flat dir + val_annotations.txt; reorganize once with:
        python -c "from ml_experiments._harness import prepare_tiny_imagenet_val; \
                   prepare_tiny_imagenet_val('{data_dir}/tiny-imagenet-200')"
    """
    from torchvision.datasets import ImageFolder
    base = Path(data_dir) / "tiny-imagenet-200"
    train_dir = base / "train"
    val_dir = base / "val_organized"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Tiny-ImageNet not set up. Download "
            "http://cs231n.stanford.edu/tiny-imagenet-200.zip, unzip under "
            f"{base}, then run prepare_tiny_imagenet_val('{base}'). See _harness.make_data docstring."
        )
    train = ImageFolder(str(train_dir), transform=_vision_transforms(True, img_size))
    val = ImageFolder(str(val_dir), transform=_vision_transforms(False, img_size))
    meta = DataMeta("tiny_imagenet", "vision", num_classes=200,
                    n_train=len(train), n_val=len(val))
    return (
        _loader(train, batch_size, True, num_workers, generator, pin_memory, drop_last=True),
        _loader(val, batch_size, False, num_workers, generator, pin_memory, drop_last=False),
        meta,
    )


def prepare_tiny_imagenet_val(base_dir: str):
    """One-time: reorganize Tiny-ImageNet val/ (flat images + annotations) into ImageFolder dirs."""
    base = Path(base_dir)
    val = base / "val"
    out = base / "val_organized"
    ann = val / "val_annotations.txt"
    if not ann.exists():
        raise FileNotFoundError(f"{ann} not found; is the archive unzipped at {base}?")
    out.mkdir(exist_ok=True)
    with open(ann) as f:
        for line in f:
            fn, wnid = line.split("\t")[:2]
            cls_dir = out / wnid
            cls_dir.mkdir(exist_ok=True)
            src = val / "images" / fn
            dst = cls_dir / fn
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
    print(f"Tiny-ImageNet val reorganized into {out}")


class _LMBlockDataset(torch.utils.data.Dataset):
    """Contiguous-block LM dataset over a 1-D token array (nanoGPT style)."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, self.tokens.numel() - self.block_size - 1)

    def __getitem__(self, i):
        x = self.tokens[i : i + self.block_size].long()
        y = self.tokens[i + 1 : i + 1 + self.block_size].long()
        return x, y


def _make_tinystories(batch_size, block_size, generator, data_dir, num_workers, pin_memory):
    """TinyStories LM data. Tokenizes (or loads cached) GPT-2 BPE tokens to .bin, then blocks.

    Requires the `datasets` and `tiktoken` packages (install once:
    `mamba run -n two_noise pip install datasets tiktoken`). If they are missing this raises
    with the exact install command. Caches train.bin/val.bin under {data_dir}/tinystories/.
    """
    cache = Path(data_dir) / "tinystories"
    cache.mkdir(parents=True, exist_ok=True)
    train_bin = cache / "train.bin"
    val_bin = cache / "val.bin"

    if not (train_bin.exists() and val_bin.exists()):
        try:
            import numpy as np
            import tiktoken
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "TinyStories needs `datasets` + `tiktoken` (not in the two_noise env). Install: "
                "`mamba run -n two_noise pip install datasets tiktoken`, then re-run. The first "
                "run tokenizes and caches train.bin/val.bin under "
                f"{cache} (subset via TWO_NOISE_TINYSTORIES_DOCS env var)."
            ) from e
        n_docs = int(os.environ.get("TWO_NOISE_TINYSTORIES_DOCS", "20000"))
        enc = tiktoken.get_encoding("gpt2")
        ds = load_dataset("roneneldan/TinyStories")
        for split, path in (("train", train_bin), ("validation", val_bin)):
            n = min(n_docs, len(ds[split])) if split == "train" else min(n_docs // 10, len(ds[split]))
            ids = []
            for i in range(n):
                ids.extend(enc.encode_ordinary(ds[split][i]["text"]))
                ids.append(enc.eot_token)
            np.array(ids, dtype=np.uint16).tofile(str(path))
        print(f"TinyStories tokenized to {train_bin}, {val_bin}")

    import numpy as np
    train_tok = torch.from_numpy(np.fromfile(str(train_bin), dtype=np.uint16).astype(np.int64))
    val_tok = torch.from_numpy(np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int64))
    train = _LMBlockDataset(train_tok, block_size)
    val = _LMBlockDataset(val_tok, block_size)
    meta = DataMeta("tinystories", "lm", vocab_size=50257, block_size=block_size,
                    n_train=len(train), n_val=len(val))
    return (
        _loader(train, batch_size, True, num_workers, generator, pin_memory, drop_last=True),
        _loader(val, batch_size, False, num_workers, generator, pin_memory, drop_last=False),
        meta,
    )


def _make_synthetic(dataset, batch_size, block_size, generator, n):
    """Tiny in-memory dataset for CPU smoke tests (no download, no disk)."""
    g = generator or torch.Generator().manual_seed(0)
    if dataset == "synthetic_vision":
        x = torch.randn(n, 3, 32, 32, generator=g)   # small images; model resize handled by transform-free path
        y = torch.randint(0, 10, (n,), generator=g)
        ds_tr = torch.utils.data.TensorDataset(x, y)
        ds_va = torch.utils.data.TensorDataset(x[: n // 2], y[: n // 2])
        meta = DataMeta("synthetic_vision", "vision", num_classes=10, n_train=n, n_val=n // 2)
    else:
        vocab = 64
        toks = torch.randint(0, vocab, (n + block_size + 1,), generator=g)
        ds_tr = _LMBlockDataset(toks, block_size)
        ds_va = _LMBlockDataset(toks[: (n // 2) + block_size + 1], block_size)
        meta = DataMeta("synthetic_lm", "lm", vocab_size=vocab, block_size=block_size,
                        n_train=len(ds_tr), n_val=len(ds_va))
    return (
        _loader(ds_tr, batch_size, True, 0, g, False, drop_last=True),
        _loader(ds_va, batch_size, False, 0, g, False, drop_last=False),
        meta,
    )


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def default_lr(optimizer: str, alpha: float, base_lr: float = 1e-3) -> float:
    """Reasonable default lr given optimizer/alpha (CONVENTIONS / §2.1 prediction).

    AdamW: base_lr. SOAP whitening (α≈0.5): ~base_lr. SOAP full power (α→1): ~10x smaller,
    because the full inverse behaves like Newton and over-steps at the AdamW lr.
    """
    if optimizer == "adamw":
        return base_lr
    if optimizer == "sgd":
        # SGD+momentum tolerates a markedly larger lr than the adaptive methods; the benchmark
        # lr-grid tunes around this. (No alpha; whitening/Newton schedule below does not apply.)
        return base_lr * 10.0
    if optimizer == "stable_evo":
        # selection-driven exponent sits near whitening; share the whitening lr.
        return base_lr
    # soap and riccati both behave Newton-like as alpha -> 1, so share the schedule.
    # linear-in-alpha interpolation of the log10 lr from base (α=0.5) to base/10 (α=1.0).
    if alpha <= 0.5:
        return base_lr
    frac = (alpha - 0.5) / 0.5            # 0 at 0.5, 1 at 1.0
    return base_lr * (10.0 ** (-frac))   # base -> base/10


def make_optimizer(
    name: str,
    params,
    alpha: float = 0.5,
    lr: Optional[float] = None,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    base_lr: float = 1e-3,
    **kw,
):
    """Build an optimizer.

    Args:
        name: 'adamw' (torch baseline) or 'soap' (SOAPFullPower with precond_power=alpha).
        alpha: precond_power for SOAP (ignored for adamw). 0.5=whitening, 1.0=full inverse.
        lr: explicit lr; if None, default_lr(name, alpha, base_lr) is used.
        kw: forwarded to the optimizer (e.g. damping, relative_damping, precondition_frequency).

    SOAP at α=1.0 defaults to relative_damping=True, damping=1e-2 (CONVENTIONS §4 / soap docstring).
    Below α=1 the same relative damping is harmless and kept for a continuous interpolation.
    """
    name = name.lower()
    if lr is None:
        lr = default_lr(name, alpha, base_lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay), lr
    if name == "sgd":
        # plain SGD with Nesterov momentum — the first-order, no-preconditioner baseline.
        return torch.optim.SGD(params, lr=lr, momentum=kw.pop("momentum", 0.9),
                               nesterov=kw.pop("nesterov", True),
                               weight_decay=weight_decay), lr
    if name == "soap":
        from optim.soap_full_power import SOAPFullPower
        soap_kw = dict(
            lr=lr,
            betas=(0.95, 0.95),
            precond_power=alpha,
            damping=kw.pop("damping", 1e-2),
            relative_damping=kw.pop("relative_damping", True),
            weight_decay=weight_decay,
            precondition_frequency=kw.pop("precondition_frequency", 10),
        )
        soap_kw.update(kw)
        return SOAPFullPower(params, **soap_kw), lr
    if name == "stable_evo":
        # StableEvolutionSOAP: GENERATE-don't-INVERT preconditioner with a selection-driven
        # per-coordinate exponent (alpha = 1/2 + 1/2*shrink). No precond_power; the operative
        # exponent is dynamic and read back via optimizer.mean_exponent(). `alpha` here only
        # influences default_lr (treated Newton-like as alpha->1, shared with soap/riccati).
        from optim.stable_evolution_optimizer import StableEvolutionSOAP
        se_kw = dict(
            lr=lr,
            betas=(0.95, 0.95),
            alpha_max=kw.pop("alpha_max", 0.9),
            alpha_min=kw.pop("alpha_min", 0.5),
            kappa=kw.pop("kappa", 0.4),
            damping=kw.pop("damping", 1e-2),
            weight_decay=weight_decay,
            precondition_frequency=kw.pop("precondition_frequency", 10),
            selection_off=kw.pop("selection_off", False),
            max_update_norm=kw.pop("max_update_norm", 1.0),
        )
        se_kw.update(kw)
        return StableEvolutionSOAP(params, **se_kw), lr
    if name == "riccati":
        from optim.riccati_precond import RiccatiPrecond
        # alpha selects the base mode (whiten=1/2, inverse=1); --precond-mode overrides.
        mode = kw.pop("precond_mode", None) or ("whiten" if alpha <= 0.5 else "inverse")
        ric_kw = dict(
            lr=lr,
            precond=mode,
            shrink=kw.pop("shrink", 0.0),
            inner_steps=kw.pop("inner_steps", 2),
            eta_p=kw.pop("eta_p", 1.0),
            damping=kw.pop("damping", 1e-6),
            safeguard=kw.pop("safeguard", 8.0),
            # amortize the Newton-Schulz: curvature factors drift slowly, so refreshing the
            # preconditioner every 5 steps (vs every step) is ~5x cheaper at no accuracy cost
            # (the SOAP/Shampoo amortization argument).
            precond_every=kw.pop("precond_every", 5),
            weight_decay=weight_decay,
            momentum=kw.pop("momentum", 0.0),
            precond_stats_from_hook=kw.pop("precond_stats_from_hook", False),
            evolve_M=kw.pop("evolve_m", False),
            evolve_M_weighted=kw.pop("evolve_m_weighted", True),
            eta_M=kw.pop("eta_m", 1e-3),
            meta_every=kw.pop("meta_every", 20),
            langevin=kw.pop("langevin", False),
            temperature=kw.pop("temperature", 1.0),
        )
        ric_kw.update(kw)
        return RiccatiPrecond(params, **ric_kw), lr
    raise ValueError(f"unknown optimizer {name!r}")


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    records: List[dict] = field(default_factory=list)  # per-logged-step tidy rows
    final_val_metric: float = float("nan")             # accuracy (vision) or perplexity (lm)
    final_val_loss: float = float("nan")
    val_metric_name: str = ""
    peak_mem_mb: float = 0.0
    mean_step_time_ms: float = 0.0
    total_wallclock_s: float = 0.0
    steps_run: int = 0


def _forward_loss(model, batch, device, is_lm):
    if is_lm:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, loss = model(x, y)
        return loss, logits, y
    images, targets = batch
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    return loss, logits, targets


@torch.no_grad()
def evaluate(model, val_loader, device, is_lm, max_batches: int = 0) -> Tuple[float, float, str]:
    """Return (val_metric, val_loss, metric_name). metric = accuracy (vision) / perplexity (lm)."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct, total = 0, 0
    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break
        loss, logits, targets = _forward_loss(model, batch, device, is_lm)
        total_loss += float(loss.item())
        n_batches += 1
        if not is_lm:
            pred = logits.argmax(dim=-1)
            correct += int((pred == targets).sum().item())
            total += int(targets.numel())
    val_loss = total_loss / max(n_batches, 1)
    if is_lm:
        return math.exp(min(val_loss, 20.0)), val_loss, "perplexity"
    return (correct / max(total, 1)), val_loss, "accuracy"


def _grad_noise_scale(model, meas_iter, device, is_lm, k, amp, use_cuda):
    """McCandlish gradient noise scale B_simple = tr(Σ)/|g|² at the model's CURRENT weights.

    Estimated from K independent micro-batch gradients (each a b-sample estimate of the true
    gradient g, covariance Σ/b). With Ḡ = mean of the K micro-grads and sumsq = Σ_i|G_i|²:
      sample-var-trace  = (sumsq − K|Ḡ|²)/(K−1)  ≈ tr(Σ)/b
      tr(Σ)  = b · sample-var-trace
      |g|²   = |Ḡ|² − tr(Σ)/(bK)                 (debiased at the full KB-sample batch)
      B_simple = tr(Σ)/|g|²                       (the critical-batch-size scale)
    Train mode + autocast so it reflects the noise the optimizer actually sees (.grad is fp32
    because params are fp32). Leaves optimizer state untouched: no .step(), grads zeroed after,
    and a dedicated `meas_iter` so the training data stream is not perturbed.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    accum = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    sumsq, b = 0.0, 0
    for p in params:
        p.grad = None
    for _ in range(k):
        batch = next(meas_iter)
        with torch.amp.autocast("cuda", enabled=(amp and use_cuda)):
            loss, _, tgt = _forward_loss(model, batch, device, is_lm)
        loss.backward()
        b = int(tgt.shape[0]) if hasattr(tgt, "shape") and tgt.dim() > 0 else int(tgt.numel())
        gsq = 0.0
        for a, p in zip(accum, params):
            g = p.grad.detach().float()
            a.add_(g)
            gsq += float(g.pow(2).sum())
            p.grad = None
        sumsq += gsq
    if k < 2 or b == 0:
        return dict(noise_scale=float("nan"), grad_norm_sq=float("nan"), tr_sigma=float("nan"))
    gbar_sq = sum(float((a / k).pow(2).sum()) for a in accum)        # |Ḡ|²
    svt = (sumsq - k * gbar_sq) / (k - 1)                            # ≈ tr(Σ)/b
    tr_sigma = b * svt
    g2 = gbar_sq - tr_sigma / (b * k)                               # debiased |g|²
    noise = (tr_sigma / g2) if g2 > 1e-12 else float("nan")
    return dict(noise_scale=noise, grad_norm_sq=g2, tr_sigma=tr_sigma)


def train_eval(
    model: nn.Module,
    optimizer,
    train_loader,
    val_loader,
    device: torch.device,
    max_steps: int = 0,
    epochs: int = 0,
    accum_steps: int = 1,
    log_every: int = 10,
    eval_every: int = 0,
    eval_max_batches: int = 0,
    grad_clip: float = 1.0,
    amp: bool = False,
    lr: float = 0.0,
    callbacks: Optional[Dict[str, Tuple[Callable, int]]] = None,
    extra_record_fields: Optional[dict] = None,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    eval_hook: Optional[Callable] = None,
    swa_start_frac: float = 0.0,
    noise_scale_every: int = 0,
    noise_scale_k: int = 4,
) -> TrainResult:
    """Train for max_steps OPTIMIZER steps (or `epochs` epochs) and return tidy records + metrics.

    One "step" = one optimizer update (after `accum_steps` micro-batches accumulated). This is the
    unit §2.2 holds comparable across batch sizes via gradient accumulation: effective batch =
    micro_batch * accum_steps.

    Args:
        max_steps: optimizer steps to run (takes precedence over epochs if >0).
        epochs: alternative budget; run this many passes over train_loader.
        accum_steps: gradient-accumulation micro-batches per optimizer step.
        log_every: record a tidy row every N optimizer steps.
        eval_every: run a val eval every N optimizer steps (0 = only at end).
        amp: use autocast (cuda only).
        callbacks: {name: (fn(model, step, train_loss), freq)} — xai_engine-style hooks.
        extra_record_fields: static fields merged into every record (model, dataset, alpha, ...).

    Returns:
        TrainResult with per-step records [{step, train_loss, wallclock_s, lr, step_time_ms,
        peak_mem_mb, val_metric?}], final val metric, peak memory, mean step time.
    """
    callbacks = callbacks or {}
    extra = extra_record_fields or {}
    use_cuda = device.type == "cuda"
    is_lm = is_lm_model(model)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and use_cuda))

    model.to(device)
    model.train()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    result = TrainResult(val_metric_name="perplexity" if is_lm else "accuracy")
    t0 = time.time()

    def infinite_loader(loader):
        while True:
            for b in loader:
                yield b

    # Determine total optimizer steps.
    if max_steps > 0:
        total_opt_steps = max_steps
        batch_iter = infinite_loader(train_loader)
    else:
        steps_per_epoch = max(1, len(train_loader) // accum_steps)
        total_opt_steps = steps_per_epoch * max(epochs, 1)
        batch_iter = infinite_loader(train_loader)

    # SWA / iterate averaging (opt-in): equal-weight running mean of params over the tail of
    # training — the principled MAP estimate under injected noise (the posterior MEAN ≈ the mode;
    # best-of-sample can't reach the mode in high-D by concentration of measure). LayerNorm net
    # => no BatchNorm running-stats to recalibrate, so the averaged weights eval directly.
    swa_enabled = swa_start_frac > 0.0
    swa_start_step = int(swa_start_frac * total_opt_steps)
    swa_avg, swa_n = None, 0
    # dedicated iterator for the gradient-noise-scale probe (so it never perturbs the train stream)
    meas_iter = infinite_loader(train_loader) if noise_scale_every > 0 else None

    def _eval_swa():
        backup = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            for p, s in zip(model.parameters(), swa_avg):
                p.copy_(s.to(p.dtype))
        vm, vl, _ = evaluate(model, val_loader, device, is_lm, eval_max_batches)
        with torch.no_grad():
            for p, bkp in zip(model.parameters(), backup):
                p.copy_(bkp)
        model.train()
        return vm, vl

    optimizer.zero_grad(set_to_none=True)
    step = 0
    last_step_t = time.time()
    best_val = float("inf")            # early-stopping trackers (opt-in via early_stop_patience>0)
    no_improve = 0
    while step < total_opt_steps:
        # Accumulate `accum_steps` micro-batches.
        micro_loss = 0.0
        for _ in range(accum_steps):
            batch = next(batch_iter)
            with torch.amp.autocast("cuda", enabled=(amp and use_cuda)):
                loss, _, _ = _forward_loss(model, batch, device, is_lm)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            micro_loss += float(loss.item())

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            torch.cuda.synchronize()
        now = time.time()
        step_time_ms = (now - last_step_t) * 1000.0
        last_step_t = now
        step += 1

        # SWA: accumulate the equal-weight running mean once past the start fraction.
        if swa_enabled and step >= swa_start_step:
            if swa_avg is None:
                swa_avg = [p.detach().clone().float() for p in model.parameters()]
                swa_n = 1
            else:
                swa_n += 1
                for s, p in zip(swa_avg, model.parameters()):
                    s.add_((p.detach().float() - s) / swa_n)

        if not math.isfinite(micro_loss):
            raise FloatingPointError(f"non-finite train loss at step {step}: {micro_loss}")

        do_log = (step % log_every == 0) or (step == total_opt_steps)
        do_eval = eval_every > 0 and (step % eval_every == 0)
        do_ns = noise_scale_every > 0 and (step % noise_scale_every == 0)
        if do_log or do_eval or do_ns:
            peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if use_cuda else 0.0
            row = dict(extra)
            row.update(
                step=step,
                train_loss=micro_loss,
                wallclock_s=now - t0,
                lr=lr or (optimizer.param_groups[0]["lr"]),
                step_time_ms=step_time_ms,
                peak_mem_mb=peak_mb,
                val_metric=float("nan"),
                val_loss=float("nan"),
            )
            # surface the optimizer's realized operative-exponent DISTRIBUTION if it exposes one
            # (StableEvolutionSOAP) — mean + spread + tails, for the cross-substrate overlay and to
            # show how much is leaning Newton vs pinned at whitening (and how that shifts with batch).
            if hasattr(optimizer, "exponent_stats"):
                es = optimizer.exponent_stats()
                row["mean_exponent"] = es["mean"]
                row["exp_std"] = es["std"]
                row["exp_max"] = es["max"]
                row["exp_frac_high"] = es["frac_high"]
                row["exp_frac_floor"] = es["frac_floor"]
            elif hasattr(optimizer, "mean_exponent"):
                row["mean_exponent"] = optimizer.mean_exponent()
            # diagonal noise loss-tax ½·tr(H·Σ_noise) from the injection (0 when not injecting) —
            # the transient observable that varies with noise SHAPE even though its equilibrium
            # integral is trace-only (experiment B).
            if hasattr(optimizer, "loss_tax"):
                row["loss_tax"] = optimizer.loss_tax()
            # gradient noise scale B_simple = tr(Σ)/|g|² at the current weights (critical batch
            # size) — tells us where the injected demographic temperature sits vs the natural
            # minibatch-sampling temperature, and how that ratio evolves over training.
            if do_ns:
                ns = _grad_noise_scale(model, meas_iter, device, is_lm,
                                       noise_scale_k, amp, use_cuda)
                row["noise_scale"] = ns["noise_scale"]
                row["grad_norm_sq"] = ns["grad_norm_sq"]
                row["tr_sigma"] = ns["tr_sigma"]
                model.train()
            if do_eval:
                vm, vl, _ = evaluate(model, val_loader, device, is_lm, eval_max_batches)
                row["val_metric"] = vm
                row["val_loss"] = vl
                model.train()
                # SWA: also score the averaged ("posterior-mean") weights — the MAP estimate.
                if swa_enabled and swa_avg is not None:
                    vm_s, vl_s = _eval_swa()
                    row["val_metric_swa"] = vm_s
                    row["val_loss_swa"] = vl_s
                if early_stop_patience > 0:
                    if vl < best_val - early_stop_min_delta:
                        best_val = vl
                        no_improve = 0
                    else:
                        no_improve += 1
            result.records.append(row)
            # within-run visibility: stream a progress line to stdout (-> SLURM .out, tail-able)
            # and flush partial results to disk each eval (also preemption-safe).
            if do_eval:
                print(f"[train_eval] step {step}/{total_opt_steps} "
                      f"train_loss={micro_loss:.4f} val_loss={vl:.4f} "
                      f"{result.val_metric_name}={vm:.4f}"
                      + (f" early_stop[{no_improve}/{early_stop_patience}]"
                         if early_stop_patience > 0 else ""), flush=True)
                if eval_hook is not None:
                    eval_hook(result)

        for cb_name, (cb_fn, cb_freq) in callbacks.items():
            if cb_freq > 0 and step % cb_freq == 0:
                model.eval()
                cb_fn(model, step, micro_loss)
                model.train()

        # early stop: val loss has not improved by min_delta for `patience` consecutive evals.
        # Bounds the to-convergence runs (large batch converges in fewer opt-steps and stops here
        # rather than over-training at high per-step cost).
        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            print(f"[train_eval] early stop at step {step} (val plateau, best={best_val:.4f})")
            break

    # Final eval.
    vm, vl, vn = evaluate(model, val_loader, device, is_lm, eval_max_batches)
    result.final_val_metric = vm
    result.final_val_loss = vl
    result.val_metric_name = vn
    result.steps_run = step
    result.total_wallclock_s = time.time() - t0
    result.peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if use_cuda else 0.0
    step_times = [r["step_time_ms"] for r in result.records if r["step_time_ms"] > 0]
    result.mean_step_time_ms = float(sum(step_times) / len(step_times)) if step_times else 0.0

    # Stamp final val metric onto the last record so a per-config CSV is self-contained.
    if result.records:
        result.records[-1]["val_metric"] = vm
        result.records[-1]["val_loss"] = vl
        if swa_enabled and swa_avg is not None:
            vm_s, vl_s = _eval_swa()
            result.records[-1]["val_metric_swa"] = vm_s
            result.records[-1]["val_loss_swa"] = vl_s
    return result


if __name__ == "__main__":
    # CLI: stage a dataset once before launching parallel array tasks, e.g.
    #   python -m ml_experiments._harness --prestage cifar100
    import argparse
    _p = argparse.ArgumentParser(description="harness data utilities")
    _p.add_argument("--prestage", help="dataset to download+extract once (e.g. cifar100)")
    _p.add_argument("--data-dir", default="data")
    _a = _p.parse_args()
    if _a.prestage:
        ensure_dataset(_a.prestage, _a.data_dir)
        print(f"[_harness] staged dataset '{_a.prestage}' under {_a.data_dir}/")
