# Image-compression analysis scripts

Reproducible measurement scripts behind [`../image_compression_report.md`](../image_compression_report.md)
(decision: store iNat images as **WebP q82, full resolution** — ~66% smaller, model-equivalence-checked).

All are **measure-only**: they encode in memory, write no files, and never touch the originals.
Each takes one or more `batch_*` dirs (+ optional sample size) and runs in the `reticulate-gpu2`
conda env. The model checks load `phenobase/phenovision` from the local HF cache (offline).

| script | what it measures |
|---|---|
| `reencode_size_test.py` | size: original → 256px JPEG → 256px WebP (first pass; long-side target) |
| `reencode_fidelity_test.py` | size at **shorter-side** 256/320/384 (JPEG & WebP) + transform passthrough |
| `webp_only_test.py` | full-res WebP q80/85/90 + lossless vs original (format-only, no downscale) |
| `fullres_format_test.py` | full-res JPEG vs WebP at matched q (pure format bonus ≈ 14% iso-q) |
| `ssim_isoquality_test.py` | rate–distortion via SSIM vs original; WebP ≈ 23–24% smaller at equal SSIM |
| `reencode_model_equivalence.py` | production model: prediction Δ (orig vs WebP q82) |
| `reencode_model_doseresponse.py` | prediction Δ vs WebP quality q98→q75 (confirms real dose effect) |
| `reencode_model_equivocal.py` | **production 3-way** decision change (0% reversals, ~1–2% borderline) |

Example: `python notes/image_compression/reencode_model_equivocal.py <batch_dir> [more dirs] [N]`
