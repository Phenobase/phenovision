# iNaturalist image-store compression: analysis & decision

**Status:** **WebP q82 (full resolution) — ACCEPTABLE / recommended.** The model-output check (§7)
showed a ~2.5% raw point-flip rate, but the production-relevant equivocal-aware analysis found
**0% confident-call reversals** and only ~1–2% borderline Detected↔Equivocal shuffling — and that
applies only to the current original-trained model on a converted store (future q82-trained models
are unaffected). Quality stays a tunable dial (q90–q92 halves even the borderline effect) but q82
is defensible. **~65% size reduction, measured ~11–12 TB → ~4 TB** at full resolution (the earlier
"~15 TB → ~5 TB" estimate over-stated the original store; see §2). See §7.
**Date:** 2026-06-30. **Author:** R. Dinnage + Claude Code. **Scope:** the iNat "medium" image
store that feeds PhenoVision training (subset) and inference (full set).

---

## 1. Motivation

`/blue` (group quota 52 TB) runs tight. The iNat image store is the single largest consumer.
Goal: **substantially reduce its on-disk size while keeping the images usable** by the existing
pipeline (no pipeline rewrite, no loss of model performance).

## 2. What we have

| location | contents | size | files |
|---|---|---|---|
| `/blue/.../phenobase_inat_data/images/medium/batch_1..870` | 870 batches | ~10–11 TB | ~87 M |
| `/orange/.../phenobase_inat_data_TEMP_MOVED/.../batch_871..932` | 62 batches (temp-moved; restored to blue) | 1.018 TB | ~6.2 M |
| **total** | 932 batches | **~11–12 TB** | ~93 M |

> **Correction (measured during the 2026-06-30 conversion):** the original store was **~11–12 TB**,
> not the ~15 TB first estimated (that figure came from an incomplete `du` × an assumed 16 GB/batch;
> real batches average ~11–14 GB and many are partial, e.g. 48k images rather than 100k). Per-batch
> conversion measured **~64–69% reduction** (batch_5: 11.0 GB → 4.0 GB; per-image webp ≈45 KB vs
> jpg ≈147 KB), so the WebP q82 *ratio* is on target; only the absolute totals shift. Post-conversion
> store ≈ **~4–4.5 TB** (freeing ~6.5 TB), better in absolute terms than the earlier "~5 TB".

Two cost dimensions: **bytes** (~11–12 TB) and **file count / inodes** (~93 M, against a ~179 M
group file count). They have different remedies (below).

## 3. Three independent, stackable levers

A 224×224 model never benefits from the bytes spent on (a) resolution above ~256 px or (b)
JPEG quality far above the perceptual knee. So:

1. **Resolution** (downscale) — the biggest lever; the originals are ~500 px "medium" but the
   model only consumes 224 px.
2. **Quality target** — the originals are *over-encoded* (see §5); dropping to a still-excellent
   quality is nearly free perceptually.
3. **Format** (JPEG → WebP) — a modest but real per-byte efficiency gain at equal quality.

These multiply. All three were measured empirically (in-memory, no files written, originals
untouched; 2 full batches / sampled, scipy/Pillow/timm).

## 4. Resolution (downscale) measurements

Measured at **shorter-side** targets — important, because the val transform is
`Resize(256 on the shorter side) → CenterCrop(224)`, and train is
`RandomResizedCrop(224, scale=(0.08, 1.0))`. Downscaling to a **shorter side < 256** would force
the pipeline to *upscale* (fabricate detail) and is wrong; shorter-side ≥ 256 guarantees a 224
crop never upscales. (An early test mistakenly used long-side-256, which under-stores and gave an
optimistic ~88% — corrected here.)

Extrapolated to the store (~11–12 TB measured; ratios below quoted the earlier ~15 TB figure —
scale by ~0.75 for absolute TB):

| shorter-side target | JPEG q90 | WebP q85 |
|---|---|---|
| 256 | ~4.4 TB (saves 71%) | **~3.2 TB (saves 79%)** |
| 320 | ~6.2 TB (saves 58%) | ~4.4 TB (saves 70%) |
| 384 | ~8.0 TB (saves 47%) | ~5.6 TB (saves 63%) |

**Augmentation-fidelity tradeoff:** `RandomResizedCrop` takes a random *area fraction* (8%–100%)
and resizes it to 224 — so only crops whose pixel region exceeds 224² are genuine sharp
downscales; smaller crops are upscaled to 224 (softer) *even on the originals*. Downscaling
shifts more of the crop-scale range into the upscaled/soft regime:
- short-256: large crops stay sharp only above ~scale 0.6
- short-320: above ~scale 0.4
- short-384: above ~scale 0.27 (preserves almost all large/mid-crop sharpness)

So smaller targets save more but soften the big-area augmentation crops. There is no way to keep
both aggressive downscaling *and* no-upscaling across the full 0.08–1.0 range (that needs ~686 px).

## 5. Quality + format measurements (full resolution)

Iso-perceptual-quality test: encode each image across a JPEG and WebP quality sweep, decode, and
compute **SSIM vs the original** (1.0 = identical), then compare sizes at *equal SSIM*. (SSIM is a
luminance-only, single-scale perceptual proxy — good for relative codec comparison.)

Rate–distortion (full-res, 3,996 images, % of original bytes / mean SSIM):

| q | JPEG | WebP |
|---|---|---|
| 75 | 33.3% / 0.9611 | — |
| 80 | 38.1% / 0.9687 | 32.0% / 0.9759 |
| 82 (interp) | ~41% / ~0.972 | **~34% / ~0.979** |
| 85 | 44.1% / 0.9781 | 38.1% / 0.9835 |
| 90 | 54.9% / 0.9863 | 47.3% / 0.9902 |
| 95 | 74.6% / 0.9965 | — |

Two key findings:

- **The originals are over-encoded.** Even near-pristine **JPEG q95 is only 75% of the original**
  (SSIM 0.9965) — i.e. dropping to q95 is perceptually free and already saves 25%. A visually
  excellent target (SSIM ~0.98, ≈ JPEG q85 / WebP q80–82) saves ~55–60% at full resolution.
- **WebP's true format advantage is ~23–24%** at *equal perceptual quality* (SSIM-matched):
  matching JPEG q85's SSIM, WebP needs 33.8% vs 44.1% (23% smaller); matching JPEG q90, 41.9% vs
  54.9% (24% smaller). The cross-codec quality offset for these photos is ~3–4 points (JPEG q90 ≈
  WebP q87), a touch less than the folkloric "5 points". (Comparing at the *same q-number* gives
  ~14% and understates WebP; the headline "62% for WebP q85 vs original" was mostly *recompression
  to a lower quality*, not format.)

## 6. Proposed: WebP q82, full resolution — pending the §7 model check

> **Confirmed by §7:** the equivocal-aware model check cleared q82 — **0% confident-call reversals**,
> only ~1–2% borderline Detected↔Equivocal shuffling (and none of it for future q82-trained models).
> q82 is acceptable; quality remains a tunable dial (q90–q92 halves even the borderline effect) if
> collaborators prefer. Resolution and format reasoning below stands.

- **WebP q82, no downscale** → **~34% of original ≈ ~65% saved → measured ~11–12 TB → ~4 TB** on blue
  (per-batch measured 64–69%; earlier "~15 TB → ~5 TB" over-stated the original store — see §2).
- **Rationale:** keeps **all resolution**, so augmentation fidelity is fully intact (only the q82
  transcode, SSIM ≈ 0.979 ≈ JPEG-q85 quality — and the model downsamples to 224 anyway, which
  *washes out* the high-frequency artifacts SSIM penalizes, so q82 is conservative for the model).
  It also preserves the option to train at >224 px later. WebP q82 sits right at the
  rate–distortion knee.
- **Downscaling remains available as an additional, stackable lever** if more space is needed
  later (short-256 WebP → ~3.2 TB), at the documented augmentation-fidelity cost. Deferred.
- **File-count/inode reduction** (WebDataset tar shards vs the already-supported RocksDB store) is
  a *separate* decision from compression and is deferred — note WebP/downscale do not reduce file
  count, only bytes.

## 7. Model-equivalence check (the real acceptance gate)

SSIM is a human-perceptual proxy; the decision is gated on **whether the production model's
predictions change**. Test: load `phenobase/phenovision` (offline from the HF cache, snapshot
`65b0d2d9…`, a 2-label reproductive checkpoint), run sampled images through the val transform
twice (original vs WebP) and compare per-class sigmoid probabilities + decision flips at the
production thresholds (fruit 0.60, flower 0.48). Run on GPU (L4), 3,000-image (q82) and
1,500-image (dose-response) samples across 3 batches.

**Result — a small but REAL, dose-dependent effect (yellow flag, not a clean pass).** Dose-response
(orig vs WebP at decreasing quality; mean |Δp| over both classes, hard-flip rate at the point
thresholds):

| WebP q | mean &#124;Δp&#124; | p95 | max | fruit flips | flower flips |
|---|---|---|---|---|---|
| 98 (near-lossless) | 0.010 | 0.059 | 0.371 | 0.87% | 0.80% |
| 92 | 0.013 | 0.072 | 0.614 | 1.53% | 1.20% |
| 85 | 0.018 | 0.093 | 0.640 | 1.93% | 2.00% |
| **82** | 0.022 | 0.118 | 0.724 | **2.53%** | **2.47%** |
| 75 | 0.030 | 0.165 | 0.807 | 3.53% | 2.93% |

Reading: flip rate rises **monotonically** as quality drops (0.8% → 3.5%), so it is a genuine
compression effect, **not** a preprocessing/boundary-jitter artifact (a pure artifact would make
near-lossless q98 look like q82; it doesn't — q98 is ~3× lower). There is an irreducible ~0.8%
"any re-encode" floor; **q82 adds ~1.7 points on top → ~2.5% raw hard flips.** Mean Δp stays small
(~0.02) with r ≈ 0.98–0.99; it's the near-boundary cases that move.

**Two caveats that temper this:**
1. **Hard flips at the point threshold OVERSTATE the real impact.** The production pipeline applies
   the **equivocal buffer** and reports **presence-only**, so most near-threshold flips land in the
   equivocal zone and are filtered out. The decision-relevant rate (Detected↔NotDetected, crossing
   the full buffer) is **lower** than 2.5% — quantifying it exactly needs the R
   `threshold_annotations()` 3-way logic applied to the orig vs q82 probabilities (recommended
   next step).
2. SSIM had put q82 in a "safe neighborhood," and it mostly is — but the model is the arbiter, and
   it sees a measurable (if small) shift.

**Equivocal-aware (production 3-way) analysis — the decision-relevant number.** Applying the real
`threshold_annotations()` 3-way logic (Detected if pred > equivocal_upper; Not Detected if pred <
equivocal_lower; else Equivocal) to orig vs q82 (2,999 images):

| class | orig D/E/N | Detected-call change | full Detected↔NotDetected reversal |
|---|---|---|---|
| fruit | 310 / 439 / 2250 | 0.93% | **0.00% (0/2999)** |
| flower | 1309 / 418 / 1272 | 1.10% | **0.00% (0/2999)** |
| image-level (either class) | — | **2.03%** | **0.00%** |

**Zero full reversals** (<0.1% upper bound) — q82 never turns a confident "present" into a confident
"absent" or vice versa. Because reversals are 0, the ~1% per-class Detected-call changes are *all*
**Detected↔Equivocal** transitions: borderline detections moving in/out of the low-certainty filter
— the least-confident calls, which the equivocal zone exists to quarantine. So the production-
relevant impact is **~1–2% borderline detected/equivocal shuffling, no reversals** (the raw 2.5%
point-flip rate overstated it).

**Train/serve consistency caveat (reduces the stakes further).** This whole concern is a *train/serve
mismatch*: v1.1.0 was trained on the original JPEGs, so it sees a small shift on q82 inputs. A model
*trained or fine-tuned on q82 data serves on the same distribution it learned* → no shift; the
artifacts become data it's robust to. So the ~1–2% applies **only to the current original-trained
model on a converted store**; every future retrain on the q82 store is unaffected. (Lossy
compression can't be "learned back," but since the model downsamples to 224, the detail q82 drops is
in frequencies the model never uses → negligible signal loss either way.) Cheap mitigation if you
keep using the current model: a one-time realignment fine-tune on a q82 sample.

**Verdict: q82 is acceptable.** No confident-call reversals; the residual effect is confined to
borderline cases the pipeline already treats as uncertain, and vanishes for future q82-trained
models. If collaborators want to shrink even the borderline shuffling, the dose-response gives the
dial (q90–q92 ≈ halves it at modest extra size); otherwise q82 maximizes space.

## 8. Implementation plan (when approved — not yet executed)

- **Batch-by-batch, verify-before-delete, delete-original-as-you-go.** Output (~34%) ≪ input, so
  net disk usage *drops* with each batch; peak extra space = only a few in-flight batches (~tens of
  GB), not a second full copy. (Mirror the existing `move/restore_batches_*` verify pattern.)
- **Safety net:** every image is re-downloadable from the iNat open-data S3 by `photo_id` (the
  `migration_setup.sh` path), so incremental deletion of originals is recoverable (minus withdrawn
  observations). Orange has no room to archive a second full copy, so S3 is the real backstop.
- **Keep paths stable:** either keep `.jpg` names (Pillow opens WebP by content regardless) or
  update `file_name`/path logic + CSVs consistently. Pipeline works out of the box: `PhenoDataset`
  uses `Image.open().convert('RGB')` (format-agnostic), timm transforms accept any resolution, and
  Pillow WebP is enabled in `reticulate-gpu2`. (Passthrough verified: re-encoded JPEG & WebP →
  `(3,224,224)` through both train and val transforms.)

## 9. Caveats

- Lossy transcode (JPEG→WebP is double-lossy); at q82 the added loss is small and the model is
  robust, but it *is* a change to training inputs — a future *retrain* on re-encoded data could
  differ marginally (inference impact expected ~nil, gated by §7).
- SSIM here is luminance-only / single-scale → treat ~23–24% as ±a few points.
- Full-res WebP preserves the >224-px-training option; downscaling would forfeit it.
