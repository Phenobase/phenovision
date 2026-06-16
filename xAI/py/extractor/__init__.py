"""extractor — per-checkpoint extraction blocks for the PhenoVision
"pretraining-as-preadaptation" experiment (briefing Part II §6).

Each block module exposes a single ``extract(ctx: ExtractCtx) -> dict`` entry point
that writes tidy-long scalars via ``ctx.scalar.add`` and heavy arrays via
``ctx.array.put``, and returns a small summary dict for the collector's log.

The ``ExtractCtx`` dataclass (the frozen block interface) is defined by the
collector/driver in ``extractor/_ctx.py``; block modules import it lazily (inside
``extract``-time type hints only) so they remain importable before the driver lands.

Blocks implemented across the parallel builds:
  * ``block_optim``      — §6.1 optimizer geometric state (metric evolution; headline)
  * ``block_curvature``  — §6.4 curvature / loss-landscape (Hessian spectra)
  * ``block_weights``    — §6.2 dimension-reduced weights / displacement
  * ``block_trajectory`` — §6.3 trajectory geometry
  * ``block_circuits``   — §6.5 QK/OV circuits
  * ``block_interp``     — §6.6 ViT interpretability / CKA
  * ``block_probes``     — §6.7 per-patch + per-layer linear probes
  * ``block_fitness``    — §6.8 held-out fitness

The driver/collector live in ``_ctx.py`` (the ``ExtractCtx`` interface + ``build_ctx``),
``extract.py`` (``Extractor`` / ``extract_checkpoint``), and ``collector.py`` (the
producer/consumer loop). These are imported lazily on demand rather than at package import
time, so ``import extractor`` stays cheap and works even before every sibling block module
is present on disk (``extract.py`` guards each block import individually).
"""

from __future__ import annotations

__all__ = [
    "block_optim",
    "block_curvature",
    "block_weights",
    "block_trajectory",
    "block_circuits",
    "block_interp",
    "block_probes",
    "block_fitness",
    # driver / collector entry points (import lazily, e.g. ``from extractor.extract import ...``)
    "ExtractCtx",
    "build_ctx",
    "Extractor",
    "extract_checkpoint",
]
