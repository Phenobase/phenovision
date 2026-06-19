# Agent guide — preadapt-v2 on a fresh allocation (rdinnage.fiu)

You are picking up the **PhenoVision preadapt-v2 experiment** on a different HiPerGator
allocation, with none of the prior session's memory. This is the context + the lessons that
aren't obvious from the code. Read `xAI/scripts/migration_setup.sh`'s header for the literal
runbook; this doc is the *why* and the *gotchas*.

## What the experiment is
Fine-tune ViT-L/16 from 3 pretrained "evolutionary histories" with the StableEvolutionSOAP
optimizer, harvesting the optimizer/curvature geometry as a measurement of metric/basin
evolution (framed for evolutionary biology). **6 runs = {naive, plantclef(VT), mae} × {seed 42, 43}.**
- **Two phases per run:** Phase 1 = head-only on a frozen backbone ("ecological fitting"; its
  converged AUC-PR is the starting-fitness estimate). Phase 2 = unfreeze all-but-the-frozen-input-stage
  and "evolve" — this is where checkpoints emit and the collector works.
- **`naive` starts from random weights**, so its Phase-1 AUC-PR is ~chance (≈0.50). That is CORRECT,
  not a bug. plantclef/mae start much higher.
- Common frozen MAE input stage (tokenizer + pos-embed) across all 3 conditions (`--shared-tokenizer mae`).

## Quick start (the runbook lives in migration_setup.sh)
1. `git clone https://github.com/Phenobase/phenovision.git && cd phenovision && git checkout two_noise-build`
2. **[HUMAN]** drop `v2_migration_csvs.zip` in the repo root (not in git, can't be re-fetched).
3. `CONDA_OVERRIDE_CUDA="11.8" mamba env create -f xAI/environment.yml && mamba activate reticulate-gpu2`
   — slim Python-only env (no R/RStudio). The `CONDA_OVERRIDE_CUDA` prefix is only needed when
   creating on a CPU login node (the CUDA torch build otherwise won't solve: "`__cuda` missing").
4. `bash xAI/scripts/migration_setup.sh`  → unzips CSVs, gdowns PlantCLEF .pth, re-downloads the
   ~228k-image subset from iNat open-data S3 (resumable). timm MAE/ImageNet weights auto-download run 1.
5. `bash xAI/scripts/launch_v2_round.sh 1 rdinnage.fiu rdinnage.fiu rdinnage.fiu-b`  (round 1 = s42).
   Round 2 (s43): same with `2`. **Run both for parallelism** if 4 GPU / 375 GB allow.

## Config + why (committed in the grid/scripts)
- `sampler_min_step 12`, `sampler_delta 0.025` — emit a checkpoint at most every 12 steps / on a 2.5%
  weight-change. **Do not lower these.** At the old `min_step=4` the trainer flooded the collector,
  leaving the B200 ~99% idle (collector-bound) and exploding the store.
- `num_workers 4` (was 8) — 8 DataLoader workers each fork a ~7.6 GB copy of the dataset → trainer
  peaked ~155 GB host RSS. 4 workers ≈ 91 GB; we run trainers at `--mem-per-cpu=13G` (104 GB).
- `num_epochs 25` is the backstop; `--phase2_early_stop` (train-loss plateau) ends runs ~earlier.
  Phase 2 (evolution) gets its OWN budget counted from 0, independent of how many epochs Phase 1
  (ecological fitting) used — set by `--phase2_epochs` (default = `num_epochs`). (Before this fix
  Phase 2 ran `range(phase1_end+1, num_epochs)`, so a long Phase 1 silently shrank the budget.)
- `max_precond_dim 2048`, `heavy_every 8` (collector).

## Collector architecture (two-pass)
Per emitted checkpoint, TWO passes must both finish before it's finalized:
- **GPU pass** (L4, hpg-turin): `block_curvature` (Hessian-Lanczos), `block_fitness`, and the HEAVY
  blocks `block_interp` (gradcov ~45s) + `block_probes` (~37s), gated by `heavy_every` (every 8th).
- **CPU pass** (burst, `*-b` QOS): `block_optim` (§6.1 optimizer geometry), `block_weights`,
  `block_trajectory`, `block_circuits`.
- Finalize requires BOTH `.gpu.complete` AND `.cpu.complete` (filesystem refcount, exactly-once).
- Trainer **backpressure**: pauses emitting when pending > high(6), resumes < low(2).
- Heavy GPU-pass checkpoints peak ~200 GB RAM → the GPU collector runs at `--mem-per-cpu=21G x10cpu`
  (210 GB). 375 GB QOS fits trainer(104) + heavy collector(210).

## Storage rework — IMPORTANT (per-run ~15-20 GB, was ~800-980 GB)
- **§6.1 CurvSummary**: `block_optim` no longer stores the full `exp_avg_sq`/`precond` spectra (full
  param numel/layer/ckpt, ~2.4 GB/ckpt, zero readers). It stores a 139-wide summary (top-64 eigs + 64
  log-spaced decay knots + 11 shape scalars). Downstream readers must use the `*_summary` zarr groups.
- **Single-slot ladder**: `kept/` = a true log2 ladder (one `step*.pt` per octave, model-only) + ONE
  rotating `kept/latest_full.pt` (FULL, atomically overwritten) as the resume anchor. The old
  latest-keep rule re-accumulated; don't reintroduce it.
- Wide-range spectra (Hessian/Lanczos/kron eigenvalues) stay **float32**; bounded arrays (attn stats,
  probe features, accuracy curves, patch maps, basin radii) are float16.
- `PREADAPT_PILOT_FULL_SPECTRA=512` (env) optionally keeps raw spectra on a sparse step grid to
  validate the summary on a pilot run.

## Resume (the trainer is resumable; one subtlety)
- `--resume auto` reads: live `checkpoints/` newest FULL → `kept/latest_full.pt` → newest FULL kept rung
  → `phase1_final.pt`. Or pass an explicit `--resume <path>`.
- **Load checkpoints with `map_location="cpu"`** — StableEvo stores a `torch.Generator` in the optimizer
  state; `map_location=cuda` makes its `__setstate__` reject the non-CPU tensor and the whole load crashes
  (already fixed in preadapt_train.py). If a run hits the 24h limit, resume it; it's near-bit-exact.

## Hard-won gotchas
- **GPU collector hangs (history):** four causes fixed — reslim RAM bloat, B200 double-backprop
  (run the collector on **L4**, not B200), BLAS thread oversubscription (set OMP/MKL/OPENBLAS threads),
  gradcov memory. Heavy checkpoints still need ~200 GB; give the GPU collector ~210 GB.
- **B200 queue:** the shared `hpg-b200` partition is contended; heavy recent group usage depletes
  fair-share → long backfill estimates. A LOWER `--time` (we use 24h, not 96h) backfills sooner — a 96h
  job can't fit before higher-priority reservations. rdinnage.fiu is usually idle → much better priority.
- **OPEN QUESTION (measure it):** 1 GPU collector may not fully keep up with 1 trainer at `min_step=12`
  (GPU pass ~2.2/min @ he=8 vs a full-speed trainer wanting ~5-8 emits/min in the fast Phase-2 window).
  Watch the trainer's step rate + backpressure once it reaches Phase 2. If badly throttled, options:
  raise `heavy_every`, add burst CPU collectors, or (375 GB allows) a 2nd GPU collector won't fit with a
  full one — use a cheap-only 2nd collector or raise min_step. Decide from the live rate.

## Monitoring (what to watch)
- `squeue -u $USER | grep preadapt` ; `ls xAI/output/preadapt_v2/*/RUN_COMPLETE | wc -l` (target 6).
- Collector keep-up: gpu-pass `DONE` count + log mtime (a >10-min-stale log with high `watch_pt` and
  climbing `sstat -j <jid>.batch --format=MaxRSS` = a hang; idle-caught-up with `watch_pt` low is fine).
- `/blue` headroom (`blue_quota`) — diskwatch is OFF; watch it yourself.
- **Group courtesy (guralnick only):** ≤3 GPU, leave ≥2 for the group; never pause/cancel/yield jobs to
  free room without the owner's explicit OK. (On rdinnage.fiu this is your own allocation — 4 GPU.)

## After all 6 complete
Report per-condition init→converged AUC-PR for s42/s43, then the **d90 sweep**: regenerate
`gen_d90_grid` with the REAL per-condition init/converged AUC-PR from the v2 runs, then `submit_d90.sh`.

## Things that may NOT be in the push (memory-only)
- The **CSV zip** (`v2_migration_csvs.zip`, train+val_v1.1.0.csv) — 469 MB, over GitHub's limit; the only
  hard manual handoff. Also at `/blue/guralnick/share/r.dinnage/v2_migration_csvs.zip` on the guralnick side.
- **Deferred (incremental, not needed):** `block_weights` proj_coords 292→24 aggregation; `block_interp`
  every-Kth probe-feature gating. Per-run is already ~15-20 GB without them.
- The exact 228,392-image subset is committed as an EXPLICIT manifest `xAI/data/v2_image_subset.csv.gz`
  (file_name,photo_id,extension); `migration_setup.sh` downloads from it (no RNG, env-independent).
  Regenerate it from the grid seeds + trainer RNG with `python xAI/scripts/gen_image_subset.py`.
- **No-transfer alternative to the whole migration:** if RC adds the `r.dinnage` login to the
  `rdinnage.fiu` SLURM account, jobs run as r.dinnage (in the guralnick UNIX group) and read
  `/blue/guralnick` in place — submit with `--account=rdinnage.fiu` and skip the data download entirely.
