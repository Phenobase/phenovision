# PhenoVision: Experiment and Checkpoint-Extractor Briefing

**Audience:** the Claude Code agents building and running the PhenoVision fine-tuning experiment and its data-collection pipeline. Assume no prior conversation context; this document is self-contained.

**Scope:** the complete handoff. **Part I** — the experiment you are running: the three pretrained starting points, how to build the new baseline, and the freezing discipline that makes the conditions comparable. **Part II** — how you observe it: the checkpoint extractor, the four-GPU orchestration (Plan A), what to collect per checkpoint, the storage schema, and the intrinsic-dimension sweep.

**How to read this.** Part I defines the runs; Part II defines the measurement. Section numbers restart in each part — an unqualified reference such as "§6.3" means *the current part*, and cross-part references are written "Part I §6" / "Part II §6". Before launching anything, read **Part I §6 (Verification)** and the combined **Do not / Escalate / Checklist** at the end.

**The optimizer.** Training uses a curvature-adaptive SOAP-family optimizer ("SOAP StableEvo"). Part II §8 is the one place that must be reconciled against the actual optimizer class on the training machine.

---

# PART I — The experiment: three pretrained starting points

## 1. What this experiment is

PhenoVision is a Vision Transformer (ViT) fine-tuned periodically on a plant-phenology task from natural-history specimen images. We are comparing how the model adapts to that task from three different pretrained starting points, treating each starting point as a different "evolutionary history" and measuring how much each one helps downstream adaptation. The three conditions share an identical architecture and an identical fine-tuning protocol; the *only* thing that differs is the initial weights.

Throughout, the **frozen input-encoding stage** means two things together: (1) the **patch-embedding layer** — the strided convolution / linear projection that maps each pixel patch to a token vector (`patch_embed` in `timm`, a `Conv2d(kernel=stride=patch_size)`) — and (2) the **positional embeddings**. Both are frozen and held identical across all three conditions (see §4). The **CLS token** is treated separately: it is *not* part of the frozen stage and remains trainable, because it is a readout query for the classification head rather than input scaffolding (see §4.4).

## 2. The three conditions

| Condition | Initial weights | Role |
|---|---|---|
| **ImageNet** | Full ImageNet-pretrained checkpoint | Generic-vision pretraining |
| **Virtual Taxonomist (VT)** | Full VT-pretrained checkpoint | Task-adjacent pretraining |
| **Very Naive Baseline** (NEW) | ImageNet **tokenizer + positional embeddings**; everything else randomly reinitialized | Naive-downstream reference |

All three are the same ViT variant (confirm the exact variant, patch size, resolution, and embedding dimension `D` from the checkpoint before starting — the running example below assumes ViT-B/16, 224×224, `D=768`, but verify).

## 3. The new condition: "Very Naive Baseline" (read carefully)

### 3.1 What it is

Take the ImageNet checkpoint, keep the **patch-embedding (tokenizer) weights and the positional embeddings**, and **randomly reinitialize every other parameter** — all transformer blocks (attention QKV/projection, MLPs, LayerNorms), the CLS token, and the classification head. The result is a model with a competent, pretrained input-encoding stage (sensory front-end + grid geometry) sitting in front of an otherwise untrained network.

### 3.2 Why it exists

It is a feasible stand-in for a true train-from-scratch baseline. A standard ViT trained from random init on our reduced datasets, within the compute/time budget for this experiment, will not converge to anything informative — it would be a degenerate floor, not a useful comparison. Keeping a good tokenizer also removes the single most optimization-unstable part of a from-scratch ViT (the patchify stem is the documented source of ViT training fragility), so the naive-downstream network actually trains to a meaningful point.

### 3.3 Naming discipline (important for any writeup or logged metadata)

- Call it the **"very naive baseline."**
- Do **not** call it "from scratch," "random init," "neutral," or a "zero-history null." It carries an inherited, pretrained tokenizer, so it is none of those.
- Its naivety lives **entirely in the randomly reinitialized downstream weights**, not in the tokenizer.

### 3.4 What it does and does not measure

- It **does** isolate the value of *downstream* pretraining given a fixed, competent tokenizer. The contrast `ImageNet − Very Naive Baseline` estimates what ImageNet's transformer-block pretraining buys you *on top of* the shared front-end.
- It **does not** measure the value of all pretraining versus nothing, and it does not occupy an independent region of weight space in the tokenizer subspace — it shares that subspace with the other two conditions by construction. Phrase any conclusions accordingly.

### 3.5 How to construct it

1. Instantiate a fresh model with the project's default initialization.
2. Load the ImageNet checkpoint and copy **`patch_embed` weights (projection weight + bias) and the positional embeddings** into the fresh model.
3. Leave everything else at fresh random init: CLS token, all transformer blocks, head.
4. Freeze the input-encoding stage (§4) and proceed with the standard fine-tuning protocol.

Positional embeddings are **frozen** here (copied from ImageNet, not learned). The CLS token is **trainable** (reinitialized, then learned). Only the patch projection and positional embeddings are frozen.

## 4. Freezing the tokenizer in all three conditions

### 4.1 The decision

In **all three** conditions, the **patch-embedding (tokenizer) weights and the positional embeddings** are **frozen** during phenology fine-tuning: `requires_grad = False`, and the parameters are **excluded from the optimizer param groups** (not merely given LR 0 — exclude them so no weight decay or optimizer state touches them). The CLS token, transformer blocks, and head remain trainable.

### 4.2 Why, in three registers

- **ML / empirical.** Freezing the patch embedding captures most of the benefit of layer-wise LR decay (the embedding barely moves during ordinary fine-tuning and learns near-universal low-level features), and it stabilizes optimization by removing the unstable patchify stem from the training problem — which matters on reduced data.
- **Methodological.** If the tokenizer and positional embeddings are identical and frozen across all three conditions, the entire input-encoding stage is a controlled constant: it cancels out of every pairwise contrast, so any measured difference between conditions is attributable to the downstream processing weights, not the front-end. (This relies on the embedders actually being near-identical — see §6.)
- **Biological framing (for context, not code).** The tokenizer is modeled as a deeply conserved, canalized sensory front-end — analogous to the phototransduction/eye-development machinery that is shared and near-invariant across visual lineages, **not** the visual cortex (the plastic, experience-shaped part, which maps onto the trainable attention layers). The positional embeddings are the retinotopy analog — the spatial map of where each input sits — and retinotopic organization is likewise among the most deeply conserved features of visual systems, so it belongs with the conserved front-end. Freezing the stage corresponds to a non-evolvable module: directions with zero variance available to respond to selection. The positional embeddings carry only grid geometry (position (i,j) is the same regardless of content), so there is nothing domain-specific in them to shed when moving from ImageNet to herbarium specimens.

### 4.3 Companion LR setting

Because the tokenizer is frozen, the standard layer-wise learning-rate decay across the remaining layers is **removed**: use a **uniform learning rate** across all trainable parameters. Freezing the embedding is the extreme limit of LR decay, and the graded decay beyond it contributes little for supervised pretraining.

**Caveat to check (see §6):** uniform LR is well-justified for supervised/CLIP-style pretraining but penalizes masked-image-modeling / contrastive (self-supervised) pretrained models more, where layer-wise decay is genuinely important. If VT turns out to be MIM/contrastive-pretrained, flag it — uniform LR could disadvantage VT relative to ImageNet and become a confound in the opposite direction.

### 4.4 CLS token and the fixed-resolution assumption

The **CLS token stays trainable** in all three conditions. Unlike the tokenizer and positional embeddings, it is not input scaffolding — it is the readout query that pools patch information for the classification decision, so it belongs with the task head, not the conserved front-end (phenology is a different readout than ImageNet classification). It is a single `D`-dimensional vector, so this is numerically minor; the point is the clean split: **frozen input encoding (tokenizer + positions), trainable processing and readout (blocks + CLS + head)**.

Freezing the positional embeddings assumes **fixed input resolution and patch grid across all runs**, which is the current plan. Frozen positions are only valid at the resolution they were learned for. If any condition is ever fine-tuned at a different resolution, the frozen (interpolated) positions could become suboptimal and this decision must be revisited.

## 5. Storage and data loading (do not cache tokens)

It is tempting, with a frozen tokenizer, to encode every image to tokens once and store the token vectors instead of the images. **Do not do this.**

- ViT patch embedding is **dimension-preserving** (for ViT-B, 196 tokens × 768 = 224×224×3 — the same scalar count as the raw image), and dense float activations do not compress like pixels. Stored as fp16, the tokens are roughly **7–30× larger** on disk than the same image as JPEG/WebP. Token caching makes the storage problem worse, not better.
- Caching post-tokenizer outputs also **freezes augmentation**, because crops/flips/color-jitter happen in pixel space *before* the tokenizer. On reduced data, losing augmentation is costly.

**Do instead:** store images at training resolution in a modern codec (**WebP or AVIF**, e.g. quality ≈ 80), keep color (phenology/senescence cues need it), augment in pixel space, and run the frozen tokenizer **on the fly** — it is the cheapest layer in the network, so there is no meaningful compute saving from caching it.

## 6. Verification steps — run BEFORE the main experiment

These gate the validity of the design. Do them first and report the numbers.

1. **Input-encoding drift.** Compute `‖E_VT − E_ImageNet‖ / ‖E_ImageNet‖` and per-filter cosine similarities for the patch-embedding weights, **and the same for the positional embeddings** (`‖pos_VT − pos_ImageNet‖ / ‖pos_ImageNet‖`). Note that VT's positions may have moved more or less than its tokenizer depending on which optimizer LR group they were in.
   - If **small** (near-identical): the shared-constant assumption holds. Prefer the **exact-control option**: copy the *same* ImageNet patch embedding **and positional embeddings** into all three models (including VT) before fine-tuning, so input-stage variance across conditions is exactly zero by construction. Resetting VT's barely-moved input stage costs essentially nothing.
   - If **large** (diverge): the controlled-constant assumption fails for whichever component drifted. **Stop and escalate** — the design needs revisiting before running.
2. **VT pretraining type.** Confirm whether VT was supervised vs. self-supervised (MIM/contrastive). This determines whether the uniform-LR setting (§4.3) is fair to VT.
3. **Architecture sanity.** Confirm ViT variant, patch size, input resolution, and `D` match across all three checkpoints. (Part II §4 invariant 4 and the manifest record the same facts for the extractor.)

## 7. What to log along each run

Logging is specified in detail in Part II — the inline per-step stream (Part II §5) and the per-checkpoint catalog (Part II §6) — which supersedes the brief list this section held in the standalone briefing. The per-condition signals the decomposition rests on: **initial (pre-update) loss** → ecological fitting; **early-epoch slope** → short-term evolvability; **final converged loss** → long-term evolvability / basin depth (anchors: VT converged in ~4 epochs, ImageNet in ~11); **weight displacement** `‖θ_t − θ_0‖` over trainable parameters only; the **efficiency ratio** `Δloss / Δ‖θ‖`; **per-layer and per-head gradient rank** and **gradient alignment** with the PCs of pretrained activations; and the downstream **d₉₀** (Part II §9.1).

---

# PART II — Observing the runs: the checkpoint extractor and orchestration

## 1. What this is and why it exists

The experiment compares how a ViT adapts to a plant-phenology task from three pretrained starting points, treating each starting point as an evolutionary history. The decomposition we are after has three parts: **ecological fitting** (how good the starting position is), **short-term evolvability** (how efficiently the model descends per step, set by local geometry), and **long-term evolvability** (the quality and structure of the basin it can reach). The extractor's job is to record, along each training trajectory, the quantities that estimate these three things — plus the optimizer's own geometric state, which is the distinctive signal here.

The headline: **because training uses a curvature-adaptive optimizer, the optimizer's slow state is already a running estimate of the loss-landscape geometry.** Harvesting it turns each run into a direct measurement of *metric* evolution — the moving Riemannian metric on the loss landscape, the analog of the mutational/genetic covariance (M/G) — not just *position* evolution. Standard fine-tuning logging discards this. For this project it is the most important thing the extractor collects, and it is nearly free because the optimizer computes it anyway. The extractor must therefore checkpoint and read the **optimizer state**, not only the model weights. Confirm the training job saves optimizer state in each checkpoint; without it, the metric-evolution signal cannot be recovered.

The naming discipline from Part I carries over: the third condition is the **"very naive baseline,"** never "from scratch" or "random init." Use the condition names `imagenet`, `vt`, `very_naive` in all logged metadata.

---

## 2. Architecture

Producer/consumer over a shared checkpoint directory:

- **Producer** = the training process. It writes checkpoints on the adaptive schedule of §3 and logs the cheap per-step stream of §5 inline. It never blocks on the extractor.
- **Consumer** = the extractor, running as its own job **on a GPU node** that can see the checkpoint storage (shared filesystem, or a fast local copy step). It polls for new checkpoints, runs the full extraction of §6 on each, writes the records of §7, then deletes the checkpoint unless it is on the retention ladder.

Two separation decisions make this robust:

**Cheap-inline vs heavy-extracted.** Anything that is a scalar or small vector and needs only the parameter or gradient norms is logged inline by the training process (§5). It is never lost, never blocked on I/O, and it is the signal that drives the adaptive sampler. The extractor is reserved for what needs the full weight or optimizer tensors, or a forward/backward pass over the probe set.

**Extraction cadence vs retention.** Expensive extraction runs on *every* emitted checkpoint — there is no cheap/expensive tiering of the schedule, because the extractor has its own GPU. Separately, keep a thin **retention ladder** of full checkpoints that are never deleted: logarithmic spacing in steps (e.g. steps 1, 2, 4, 8, 16, … plus the most recent one or two). The ladder is O(log T) checkpoints — trivial disk — and its purpose is not throughput. It is insurance for analyses that need raw weights and cannot be reconstructed from extracted summaries: mode connectivity between the three endpoints (§9), trajectory PCA recomputed later with directions you choose, crash-resume of the training job, and any quantity you did not think to extract. Everything off the ladder is extract-then-delete.

### 2.1 Atomicity and idempotency

The extractor must never read a half-written checkpoint, and a restart must never re-process or, worse, delete-after-partial.

- Training writes to a temporary name and atomically renames on the same filesystem, **or** drops a zero-byte sentinel (`<ckpt>.done`) only after the checkpoint is fully flushed. The extractor only touches checkpoints whose sentinel exists (or whose rename has completed).
- Before reading, the extractor renames the checkpoint to `<ckpt>.processing` (or takes a lock), so a second worker or a restart cannot collide.
- Ordering is strict: **write the extracted record, fsync, verify it is durable, then delete the checkpoint.** Never delete first. Record processed step indices (in the manifest or by presence in the store) so a restart skips finished work.

### 2.2 Backpressure

With expensive extraction on every checkpoint, a burst of rapid emission (see §3) could let the extractor fall behind. The min-step-gap guardrail in §3 bounds the worst-case emission rate; the retention ladder bounds disk if the backlog grows. If extraction wall-time per checkpoint times the emission rate exceeds the production rate for a sustained period, raise the min-step gap or reduce probe-set size rather than letting disk fill — and see the Escalate section at the end.

### 2.3 Running three conditions under one collector (Plan A)

The chosen layout for the four-GPU budget: three training jobs (one per condition) and a single shared collector, all concurrent, with the intrinsic-dimension sweep (§9.1) handled as a separate batch before and/or after the main block rather than concurrently.

| GPU | Job |
|---|---|
| 1 | train `imagenet` |
| 2 | train `vt` |
| 3 | train `very_naive` |
| 4 | the single collector (serves all three) |

Reserve the faster cards for the three trainers; the collector does not need a B200, since per-checkpoint extraction for ViT-B is light — an L40 is sufficient. Do **not** co-locate the collector on a trainer's GPU: extraction bursts risk OOM against an active ViT-B training process and add step-time jitter. (Loss-vs-step is deterministic, so co-location would not corrupt the measurements, but the budget allows clean separation, which is the default.)

**One collector, three producers, shared queue.** The collector watches all three checkpoint directories and maintains a single work queue of `(run_id, step, path)` entries for checkpoints whose `.done` sentinel exists and are not yet `.processing`. It processes strictly **sequentially** — one checkpoint at a time — so peak GPU memory is one model plus its extraction activations, never three. The three producers write to three separate directories and the single consumer renames-to-`.processing` before reading, so the §2.1 ordering (write record → fsync → verify → delete) already rules out collisions; there is no cross-producer contention to manage.

**Fairness.** Schedule oldest-first by wall-time, but cap how many consecutive checkpoints come from one `run_id` — round-robin across the run_ids that have ready work — so a fast-moving trainer (typically all three early in training) cannot monopolize the collector and let another condition's checkpoints pile past its retention ladder. A workable rule: take the oldest ready checkpoint, but if any run_id has more than two queued while another has had none processed this cycle, serve the starved run first. Records and the retention ladder are per-`run_id` (§7).

**Utilization check — run on a pilot before launching.** Confirm one collector can keep up with three trainers:

```
T_ex         = full extraction wall-time on one ViT-B checkpoint
               (this optimizer, this probe set, this Lanczos budget)
sec_per_step = measured training step time
r_max        = 1 / (min_gap_steps × sec_per_step)   # worst-case emissions/sec per trainer
utilization  = 3 × r_max × T_ex                      # three trainers, one collector
```

Keep `utilization` below ~0.7 for burst headroom, and set `min_gap_steps` (the §3 floor) from this inequality rather than from convenience. The stress window is early training, when all three trainers cross the displacement threshold quickly and emit together; the adaptive sampler front-loads emission, so average utilization over the whole run sits well below that early peak — size for the peak. If `utilization` exceeds ~0.7, raise `min_gap_steps`, cut Lanczos iterations or probe-set size, and re-measure. If it still will not fit, fall back to waves (two trainers per collector) rather than starving the queue — but for ViT-B with a modest extraction budget it should fit comfortably. The retention ladder absorbs transient early bursts; sustained over-subscription is the failure mode to avoid.

*Optional efficiency, off by default:* late in training the collector is mostly idle (sparse emission), so its GPU could backfill a few init-d₉₀ subspace runs. That reintroduces co-location jitter on that one card and complicates scheduling, so leave it off unless compute-starved; clean Plan A runs d₉₀ as a separate batch (§9.1).

---

## 3. The adaptive sampler (when to emit a checkpoint)

A fixed schedule of widening gaps assumes change decelerates monotonically. It does on average, but it would miss a late burst of rapid change — the punctuated case, long stasis interrupted by a fast shift. Sample on **motion**, not on a clock, so emission is dense when the model moves and sparse when it does not, and a late burst triggers extra checkpoints automatically. Drive this on **weight change, not loss change.**

**Signal: net displacement from the last saved checkpoint, made scale-free.**

```
trigger when   sum_l ||theta_l(t) - theta_l(saved)|| / ||theta_l(saved)||   >   delta
```

summed over trainable layers `l` (the frozen embedder contributes zero and is excluded). Use *relative* per-layer displacement, not a single global norm, so the largest layer does not dominate and the threshold means the same thing across the three conditions — the very naive baseline starts with randomly initialized downstream weights at a different scale from ImageNet, and a raw-norm threshold would sample it differently. The training process already tracks per-step displacement (§5), so this test is cheap.

Net displacement (not cumulative path length) is the right trigger: it spaces consecutive checkpoints roughly uniformly in *location*, so each is a genuinely different point, and a model churning in place does not trigger — you do not spend checkpoints on motion that goes nowhere. Distinguishing that churn from real travel is the job of §6.3, not the sampler.

**Two guardrails, both required:**

- **Min-step floor:** never emit more often than every `k` steps. Bounds extractor load during a spike (important now that expensive extraction runs every time).
- **Max-step cap:** force an emission every `K` steps regardless of motion. Guarantees you never go blind, catches slow churn that never crosses `delta`, and provides a regularly spaced backbone for time-series analyses.

Optional: add a path-length OR-trigger (`sum of per-step ||v_t|| since last save > delta_path`) if you want to guarantee sampling during a high-velocity excursion that returns to its start. Default off.

**Caveat to handle downstream:** adaptive emission produces non-uniform time sampling. Store the exact step index and wall-time on every record (§7). Any analysis of rates (`dL/dt`) or autocorrelations must resample onto a uniform grid; the max-step cap gives a uniform-ish backbone for that.

Set `delta`, `k`, `K` from a short pilot: pick `delta` so a typical early-training segment emits at a workable rate, confirm late training falls back to the max-step cap, and confirm the extractor keeps up.

---

## 4. Comparability invariants (gate — set these once, freeze them)

Every cross-condition contrast and every time series depends on these being identical across all three conditions and all checkpoints. This is the same controlled-constant discipline Part I applies to the tokenizer (Part I §4). Record all of them in the manifest (§7).

1. **Fixed probe-image set.** One held-out set of images used for every forward/backward-based metric (attention statistics, per-head gradient rank, CKA, the linear probes). Same images, same order, every checkpoint, every condition. Split it into a probe-train and probe-eval partition for the linear probes (§6.7) and never let probe images leak into the fine-tuning data.
2. **Fixed random projection.** One seeded projection for the dimension-reduced weights (§6.2), shared across conditions. Store only the seed and the output dimension. **Do not materialize a dense `[d × n_params]` matrix** — for ViT-B the trainable parameter count is ~10^7–10^8 and a dense projection is infeasible. Use a sparse random projection (Achlioptas / very-sparse Li et al.) or a hashing-based signed projection (count-sketch / feature hashing) regenerated deterministically from the seed, applied per-layer and concatenated.
3. **Fixed Hessian/Fisher batch.** One fixed data batch (or fixed set of batches) for the curvature estimates of §6.4, so sharpness numbers are comparable over time and across conditions.
4. **Architecture and config sanity.** Confirm ViT variant, patch size, resolution, and `D` match across all three checkpoints (the running example is ViT-B/16, 224, `D=768`, 12 heads, head dim 64 — see Part I §2; verify against the real checkpoints). Record the optimizer config (exponent / `precond_power`, `precondition_frequency`, `max_precond_dim`, damping, shrinkage) in the manifest.

---

## 5. What the training process logs inline (every step, cheap)

These need only parameter and gradient norms, cost almost nothing, and must not be lost if a checkpoint is deleted. They also feed the adaptive sampler and the trajectory geometry.

- **Fitness:** train loss; the phenology task metric; held-out loss if a cheap eval is available. Record the **pre-update loss at step 0** explicitly (ecological fitting).
- **Weight displacement:** `||theta_t - theta_0||` over trainable parameters only, global and per-layer; and the per-layer relative displacement used by the sampler (§3).
- **Per-step velocity quantities** for §6.3: keep a ring buffer of recent velocities `v_t = theta_t - theta_{t-1}` (or their projected form, §6.2) and accumulate running dot products `v_t · v_{t+tau}` for a few small lags `tau` so velocity autocorrelation can be computed without storing full vectors. Accumulate per-step `||v_t||` for path length.
- **Optimizer scalars:** learning rate, the realized exponent / shrinkage if the schedule varies them (§6.1), and global update norm.

Log these to the same store as the extractor (§7), keyed by step, so inline and extracted records share a timeline.

---

## 6. What the extractor collects per checkpoint

Each subsection notes which part of the framework decomposition it measures and whether it is a **direct test** of a framework prediction or **general-purpose context**. Collect everything below on every emitted checkpoint.

### 6.0 Framework mapping (why each block is collected)

| Block | Measures | Direct test or context |
|---|---|---|
| 6.1 Optimizer geometric state | Metric evolution (the M/G analog) | Direct — the headline signal |
| 6.2 Dimension-reduced weights | Position evolution; where adaptation concentrates | Context + direct (displacement-on-curvature) |
| 6.3 Trajectory geometry | Directed vs confined vs oscillatory motion; approach to the OU stationary distribution | Direct |
| 6.4 Curvature / landscape | Short-term evolvability (local geometry); long-term evolvability (basin sharpness) | Direct |
| 6.5 QK / OV circuits | Mechanistic structure of attention; signed patch interactions | Context (mechanistic) |
| 6.6 ViT interpretability | Module specialization / modularity; representational drift | Context + direct (per-head gradient rank) |
| 6.7 Per-patch + per-layer probes | Where semantic/task content lives and how it forms | Context |
| 6.8 Fitness on held-out | Ecological fitting; short- and long-term evolvability | Direct |

### 6.1 Optimizer geometric state (metric evolution) — adaptive to the optimizer variant

The SOAP-family optimizer maintains, per 2D parameter under `max_precond_dim`, a slow Kronecker eigenbasis (`QL`, `QR` — the eigenvectors of `L = E[G Gᵀ]` and `R = E[Gᵀ G]`), a rotated second moment (`exp_avg_sq`, whose per-coordinate values estimate the curvature eigenvalues), and the Kronecker accumulators `L`, `R` themselves. The exponent (`precond_power`: 0.5 = whitening, 1.0 = full inverse) lives in the param group. For ViT-B every attention QKV/proj and MLP weight matrix is 2D and under threshold, so all the interesting layers get full Kronecker treatment; 1D parameters (LayerNorm, biases, CLS token) fall back to Adam and have no eigenbasis. Iterate over param groups, check the per-parameter state for the eigenbasis fields, and treat only the 2D-preconditioned layers here. **Verify the exact field names against the real optimizer class** (§8); the names above are the SOAPFullPower defaults and may differ.

Per preconditioned layer, store (summaries, not full matrices):

- **Top-`k` eigenvectors and eigenvalues** of `QL` and `QR` (the metric's principal directions and their strengths). Full bases are large; `k` of order 16–64 is enough for the alignment statistics below.
- **Rotation rate:** principal angles of the current basis against the previous checkpoint's basis (how fast the metric is moving) and against the pretrained-init basis `Q_0` (how far the metric has drifted from its ancestral state). The init-relative angles, per condition, are the phylogenetic "metric divergence from ancestor" signal.
- **Curvature-eigenvalue spectrum:** the sorted `exp_avg_sq` values (a per-layer Hessian/Fisher eigenvalue estimate, for free). Store the full sorted spectrum or a compact summary (top values, trace, participation ratio, a coarse histogram).
- **Kronecker-factor spectra:** top eigenvalues / trace / participation ratio of `L` and `R`.
- **Realized exponent:** `precond_power`, and if a noise-dependent schedule or shrinkage is active, the shrinkage `rho` and the *realized* preconditioner spectrum exponent (how close to A^{-1/2} whitening vs A^{-1} full inverse). The exponent is the quantity the cross-substrate prediction is about; watching it move per layer is the optimizer-side observation of that law.

**Conditional M block — present only if the optimizer has an evolving-M meta-loop.** Some variants (the Riccati / evolving-M design) carry, per layer, an explicit *target* covariance `M` — the source term in the preconditioner update, the slow third-timescale object that is the direct M-matrix analog. Detect it as in §8.

- **If `M` is present:** store its eigenvalue spectrum and its alignment (principal angles / cosine of eigenbases) with the curvature estimate `C` (from `L`, `R`, `exp_avg_sq`) and with the fitness curvature `A` where available. This is the single most theory-loaded object in the run — it is the moving M-matrix itself. Track its drift from any initial value and its rotation rate, as for the eigenbases above.
- **If `M` is absent:** the optimizer's "target" is a constant isotropic floor set by damping / shrinkage — the canalization floor, not an evolving matrix. Record the scalar damping, `relative_damping`, and shrinkage parameters from the param group instead, and skip the matrix-valued M extraction. The curvature estimate `C` above is then the only metric object; that is the expected, fully valid non-evolving case.

Write the extractor so the M block is gated on a single detected boolean (§8) and both paths produce well-formed records, differing only in whether an `M` group exists in the store.

### 6.2 Dimension-reduced weights

- **Primary representation: the frozen shared random projection** (§4, invariant 2) of the trainable parameters — global and per-layer coordinates. A shared frozen projection gives the three lineages a common low-dimensional coordinate system, so their trajectories can be plotted in the same space; per-checkpoint PCA cannot, because its axes drift. Run trajectory PCA *post hoc* from these coordinates (or from the retained-ladder weights) rather than online.
- **Per-layer change profile:** `||theta_l(t) - theta_l(0)||` per layer (where adaptation concentrates — which modules "evolve"); the singular-value spectrum of each weight matrix (effective rank); and the cosine between each layer's cumulative change direction and its init. A per-module "how much and in what direction has this changed" profile, which feeds the modularity reading.
- **Displacement on the curvature basis (direct test):** project `theta_l(t) - theta_l(0)` onto the pretrained-curvature eigenbasis `Q_0` and report the fraction of displacement in low-curvature (flat) vs high-curvature (steep) directions. This asks whether adaptation moves along productive directions — the geometric signature of short-term evolvability, and the spectral counterpart of the efficiency ratio.

The random projection approximately preserves L2 distances, so the trajectory geometry of §6.3 can be computed on these coordinates cheaply and faithfully.

### 6.3 Trajectory geometry — the *type* of movement

This panel answers a specific question raised by an earlier experiment: a PlantCLEF-pretrained model changed its loss at the same rate as the ImageNet model but traveled nearly three times as far in weight space. Was that directed travel into a more distant basin, or churning in place — moving fast but going nowhere? The four diagnostics below distinguish directed, diffusive, confined, and oscillatory motion. The first three need only the (projected) parameter trajectory and run mostly inline (§5); the fourth reuses the eigenbasis from §6.1.

- **(a) Straightness ratio** `R = net_displacement / path_length` over sliding windows, with `path_length = sum ||v_t||` and `net_displacement = ||theta_end - theta_start||`. `R ≈ 1` is directed (ballistic); `R ≈ 0` is churning. This is the direct PlantCLEF test: ~3× the path length with similar net displacement gives `R_PlantCLEF ≪ R_ImageNet` (churning in a basin); ~3× net displacement too gives similar `R` (genuinely farther travel).
- **(b) Mean-squared-displacement scaling** `MSD(tau) = mean_t ||theta_{t+tau} - theta_t||²` versus lag `tau`; report the log-log slope and whether `MSD` saturates. Slope ≈ 2 is ballistic/directed, ≈ 1 is diffusive/random-walk, and a **plateau** is confined — bounded motion within a basin, with the plateau ≈ basin size². A high step velocity with a saturating `MSD` is the precise signature of "fast but going nowhere," and it yields the confinement *scale*, which `R` alone does not. **Framework reading (solid):** the framework treats training as an Ornstein–Uhlenbeck process, whose `MSD` grows then saturates at the stationary covariance. Directed→confined is therefore the *expected* arc (transient descent, then sampling the stationary distribution), and the plateau estimates `tr(Σ_∞)`. The PlantCLEF observation then splits into two testable hypotheses — a deeper/more distant basin (large net displacement, directed) vs a wider/flatter stationary distribution (more confined diffusion, larger `Σ_∞` in flat directions). (a)+(b) tell them apart.
- **(c) Velocity autocorrelation** `C(tau) = mean_t (v_t · v_{t+tau}) / mean_t ||v_t||²` for a few small lags (accumulated inline, §5). Positive over many steps is persistent directed motion; **negative** at some lag is back-and-forth oscillation; fast decay to ~0 is diffusive/circling. This is the finest discriminator and the one that separates oscillation from circling.
- **(d) Spectral flat/steep split of motion (direct test, reuses §6.1):** project the per-layer displacement-since-last-checkpoint (or the optimizer's rotated first moment `exp_avg`, which is already the smoothed update direction in the eigenbasis) into the curvature basis `Q`, and report the fraction of motion in flat vs steep directions, plus loss change per unit motion resolved by direction. Churning is motion piled into flat directions, where moving costs no loss; progress is motion along loss-reducing directions. **Hypothesis to test, not assert:** persistent late motion in flat directions is what the exploration / natural-gradient stationary regime predicts (large standing variance in flat directions while the iterate sits at the optimum). If PlantCLEF is doing that, the "fast but going nowhere" is a regime signature, not a pathology — flag it as a hypothesis these diagnostics evaluate.

Compute all four **globally and per-layer** — where the churn lives (which modules travel vs oscillate in place) is itself a modularity signal.

### 6.4 Curvature / loss-landscape

Part of this comes free from §6.1; beyond that, the explicit estimates on the fixed Hessian batch (§4, invariant 3):

- **Hessian / Fisher top eigenvalues, trace, and spectral density** via Lanczos / stochastic Lanczos quadrature (e.g. PyHessian, `hessian-eigenthings`). Top eigenvalue is sharpness; trace is mean curvature; the eigenvalue density is the basin's spectral fingerprint (short-term evolvability in the local geometry; long-term evolvability in basin sharpness). These need a forward/backward pass — fine on the GPU node, on every checkpoint.
- **Effective-dimensionality proxy:** participation ratio of the curvature or gradient-covariance spectrum, per layer — a cheap stand-in for the full intrinsic dimension `d₉₀` (which is its own protocol, §9).

### 6.5 QK / OV circuits (mechanistic, weight-only)

Each attention head factors into two circuits (Elhage et al., transformer-circuits framework): the **QK circuit** `W_QK = W_Qᵀ W_K` sets the attention *pattern* (which patches read from which), and the **OV circuit** `W_OV = W_O W_V` sets *what* is written to the destination patch when attention is paid. Both are functions of the weights alone, so the extractor forms them directly from the checkpoint with **no forward pass**.

For a timm-style ViT with a fused `qkv` Linear (weight `[3D, D]`) and a `proj` Linear (`[D, D]`), per head `h` with head dim `d_h = D / n_heads`:

```
Wq_h = qkv.weight[0:D]      [h*d_h:(h+1)*d_h, :]
Wk_h = qkv.weight[D:2D]     [h*d_h:(h+1)*d_h, :]
Wv_h = qkv.weight[2D:3D]    [h*d_h:(h+1)*d_h, :]
Wo_h = proj.weight[:, h*d_h:(h+1)*d_h]
W_QK_h = Wq_h.T @ Wk_h        # D x D, rank <= d_h
W_OV_h = Wo_h @ Wv_h          # D x D, rank <= d_h
```

Per head, per layer, store:

- **Singular / eigenvalue spectra** of `W_OV` and `W_QK` (effective rank; how the read and write transforms reshape).
- **OV eigenvalue sign structure:** positive eigenvalues are copying/reinforcing, negative are anti-copying/suppressing. This operationalizes the signed, competition/mutualism-style patch interactions (one patch suppressing another) that a plain attention-weight heatmap discards — the negative structure lives here and in negative QK routing logits.
- **Circuit drift from init:** `||W_OV(t) - W_OV(0)||` and principal angles per head — which heads' circuits reorganize most under phenology adaptation, per condition.

**Caveats (do not overclaim).** The clean QK/OV factorization is *exact only for attention-only* transformers. A real ViT block interleaves an MLP and LayerNorms, so for ViT-B this is an idealization: the MLP is a third computational element (two linear layers plus a nonlinearity, separately analyzable as key–value memories), and LayerNorm should be folded into the adjacent weights for a faithful circuit (a first pass may use raw weights and note the approximation). Cross-layer composition (induction-head-style path tracing through the residual stream) is a research thread, not a per-checkpoint metric — out of scope here.

### 6.6 ViT interpretability (forward/backward on the probe set)

All on the fixed probe set (§4). Store statistics as the time series; store full maps only sparsely (§6.7).

- **Mean attention distance** per head per layer (the average spatial distance between a query patch and the patches it attends to — local vs global). Tracking it over training is heads specializing or despecializing: the evolution of modularity, and the most framework-relevant ViT metric here.
- **Attention entropy** per head (sharp vs diffuse) and **CLS-token attention concentration** (where the trainable readout query points; watching it move is the readout adapting).
- **Per-head and per-layer gradient-covariance rank (direct test):** the eigenspectrum / effective rank of the gradient covariance per layer, and per-head gradient rank — whether individual modules show dimensionality reduction during fine-tuning. Needs a backward pass on the probe set.
- **CKA** (centered kernel alignment; Kornblith et al.) between consecutive checkpoints, each checkpoint vs init, and across the three conditions at matched training fraction — representational drift, and whether the three lineages converge representationally despite different ancestry.
- **Frozen-embedder assertion (correctness check):** since the patch-embedding and positional embeddings are excluded from the optimizer, `||E(t) - E(0)||` over those parameters must be **exactly zero** every checkpoint. Log it; a nonzero value means a bug in the freezing. (The one-time pre-experiment `||E_VT - E_ImageNet||` check in Part I §6 still gates the design separately.)

### 6.7 Per-patch and per-layer probes

Two related probes on the fixed probe set, with a fixed probe-train/probe-eval split (§4). These answer "what does each patch mean and where does task content live," which the attention heatmap does not.

- **Per-layer image-level linear probe:** the CLS/pooled token at layer `l` to the image class set; report accuracy versus depth and how it migrates and sharpens over training. Fit closed-form ridge or a few SGD epochs on cached features — cheap per checkpoint.
- **Per-patch classification to image-level classes:** each patch token at layer `l` to the class set, giving (a) a per-patch class map = coarse segmentation at patch resolution (14×14 for ViT-B/16 at 224 — coarse, not fine), showing which patches carry the class signal (for phenology, whether flower/fruit patches light up — a biologically meaningful localization readout), and (b) the **emergence depth**: the first layer at which patch tokens become linearly class-predictive, and how that depth shifts over training as the localization computation forms. (Related to emergent ViT segmentation, e.g. DINO/Caron et al., and MaskCLIP-style per-patch classification.)
- **Store:** per-layer image-probe accuracy; per-layer per-patch accuracy and the emergence depth; and a few example per-patch class maps on a **tiny fixed image subset** for qualitative figures (heavy, so sparse).

**Caveats.** Linear decodability is representational *content*, not proof the network causally uses it (content, not mechanism). Patch-resolution segmentation is coarse. State both.

### 6.8 Fitness on held-out data

Held-out loss and the phenology metric on a fixed eval set; and, if auxiliary held-out categories or related tasks exist, a zero-shot transfer probe (ecological fitting). The trajectory of initial → early-slope → final loss is the spine: pre-update loss is ecological fitting, early slope is short-term evolvability, final loss is long-term evolvability / basin depth (existing anchors from prior runs: VT converged in ~4 epochs, ImageNet in ~11).

---

## 7. Storage schema

Do not write thousands of small pickle files. Use two stores plus a manifest, all keyed so inline (§5) and extracted (§6) records share one timeline.

- **Scalars and small vectors → Parquet (or DuckDB/SQLite), tidy long format**, one row per `(condition, run_id, step, wall_time, layer, head, quantity, value)`. Covers losses, displacements, norms, straightness, MSD slope, autocorrelation values, attention distances/entropies, probe accuracies, emergence depth, principal-angle summaries, realized exponent, frozen-embedder check.
- **Arrays → one Zarr (or HDF5) store**, groups indexed on `(step, layer[, head])` and **chunked along the step axis** for efficient time-series reads. Covers eigenvalue/singular-value spectra, `MSD(tau)` curves, random-projection coordinates, CKA matrices, circuit spectra, the conditional `M` eigenstructure, and the sparse example per-patch maps. Zarr lets the extractor append step-slices concurrently while analysis reads lazily.
- **Manifest (JSON, one per run):** condition name; optimizer config (exponent / `precond_power`, `precondition_frequency`, `max_precond_dim`, damping, `relative_damping`, shrinkage); the detected **M-present boolean** (§8); the random-projection seed and output dimension and method; the probe-image IDs and the probe-train/eval split; the Hessian-batch IDs; architecture (variant, patch size, resolution, `D`, `n_heads`, `d_h`); the retention-ladder policy; and the sampler parameters (`delta`, `k`, `K`).

Use one `run_id` per (condition × seed) so replicates and conditions are queryable together.

---

## 8. Detecting the optimizer variant (the M-adaptability instruction)

The extractor must work whether or not the optimizer carries an evolving-M meta-loop. **You have the real optimizer; verify field names against it rather than trusting the defaults below.** Inspect `optimizer.state_dict()` — both the per-parameter `state` and the `param_groups`.

```python
# SOAP-family per-parameter state typically includes (SOAPFullPower defaults):
#   'exp_avg'      first moment in the rotated basis
#   'exp_avg_sq'   second moment in the rotated basis  -> curvature-eigenvalue estimate
#   'QL', 'QR'     Kronecker eigenbases (the slow metric directions)
#   'L', 'R'       Kronecker accumulators E[G Gᵀ], E[Gᵀ G]
# param_groups carry: 'precond_power' (the exponent), 'damping', 'relative_damping',
#   and, if present, a shrinkage knob (e.g. 'shrink'/'rho').

def detect_evolving_M(state_for_one_param, group):
    # An evolving-M meta-loop carries an explicit *target* covariance / source matrix
    # per layer (the Riccati 'M'/source), distinct from the curvature accumulators L/R.
    # Look for a per-parameter buffer that is matrix-valued and is updated on a slow,
    # separate timescale (its own EMA / meta-update), with a name such as:
    candidates = ['M', 'M_target', 'source', 'precond_target', 'G_target']
    return any(k in state_for_one_param for k in candidates)
```

- **If detected:** run the conditional M block in §6.1 — store the target's eigenstructure and its alignment to `C` and `A`. Set `M_present = true` in the manifest.
- **If not detected:** skip the matrix-valued M block, record the scalar damping/shrinkage as the constant isotropic floor (the canalization floor), and set `M_present = false`. This is the expected non-evolving case and is fully valid.

Gate the whole behavior on the single boolean so both paths produce well-formed records that differ only in whether an `M` group exists in the Zarr store. If the candidate names do not match the real optimizer, identify the target buffer by its semantics — matrix-valued, per layer, slow separate update — and wire `detect_evolving_M` to the actual field.

---

## 9. Separate protocols (do not run these inside the extractor)

These need their own runs or post-hoc passes over the retention ladder, not a per-checkpoint readout.

### 9.1 The init-d₉₀ batch

**Purpose.** Per condition, measure the intrinsic dimension of fine-tuning *from that condition's init* — the smallest random-subspace dimension `d` at which fine-tuning reaches 90% of full fine-tuning performance. The framework prediction is that more task-aligned pretraining lowers d₉₀ (expected order VT < ImageNet < very naive). Because it depends only on the three init checkpoints, which exist before any fine-tuning starts, it runs **independently of the main block** — embarrassingly parallel, no data dependency on the live runs. It is a *downstream* d₉₀, since the tokenizer and positions are shared and frozen.

**Reparameterization** (Li et al. 2018; Aghajanyan et al. 2021). `θ = θ₀ + P v`, where `θ₀` is the condition's init over **trainable parameters only** (exclude the frozen embedder and positions, as everywhere else), `P` is a fixed `D_trainable × d` projection drawn once per `(condition, d, replicate)` seed, and `v ∈ R^d` is trained from zero. Train only `v`; gradients flow back through `P`. Hold the rest of the protocol — frozen embedder, uniform LR, augmentation, data, batch size — identical to the main runs, so the only change is confining updates to the subspace.

**Projection.** As in §4, never materialize a dense `D_trainable × d` matrix. Use the same seeded sparse / hashing-based machinery (the Li et al. fastfood transform is the canonical memory-efficient choice; a seeded sparse or hashing projection is equally fine), applied per-layer and concatenated. Implement `P v` and its transpose matrix-free, and regenerate `P` identically from its seed (or cache it on the GPU for the run's duration — it is constant within a run).

**Sweep and replicates.**
- `d`-grid: geometric over roughly `10²`–`10⁵`–`10⁶` (e.g. 100, 300, 1k, 3k, 10k, 30k, 100k, 300k), then refine around wherever the 90% threshold falls once a coarse sweep brackets it.
- Replicates: **5–10 per `(condition, d)`**, different `P` seeds. The per-estimate variance is small by concentration of measure, but the cross-condition *difference* in d₉₀ is what needs confidence intervals, so replicate.
- All three conditions. Total ≈ 8 d-values × 3 conditions × 5–10 replicates ≈ **120–240 independent runs**.

**The 90% criterion — fraction of the gain, not the absolute metric.** Use 90% of the *improvement over the init*, since the absolute score is dominated by the floor:

```
target(condition) = perf_init(condition) + 0.9 × (perf_full(condition) − perf_init(condition))
```

where `perf_full` is the converged held-out metric of that condition's **main** full-rank run and `perf_init` is the metric at the init (pre-fine-tuning). `d₉₀(condition, replicate)` is the smallest `d` on the grid whose run reaches `target`.

**Stopping rule per subspace run.** Stop on whichever comes first: (a) the `target` is reached (record `criterion_met`, the step, the metric); (b) convergence — held-out metric plateau, no improvement beyond `eps` over a `patience` window; or (c) a **step cap matched to the main run's budget**, so a subspace run never costs more than a full fine-tuning run. If you only need the binary "did this `d` reach 90%," early-stop at (a) to save compute; if you want smooth `perf(d)` curves for the figure, run each grid point to convergence or the cap.

**Optimizer for `v`.** Use one optimizer across **all** d₉₀ runs and all conditions — that invariant matters more than the choice. Adam is the literature default for this method and is fine here (`v` is a low-dimensional generic vector, where the curvature-adaptive machinery of the main optimizer buys little); matching the main SOAP-family optimizer is also defensible if you prefer consistency. Pick one, hold it fixed.

**Scheduling under Plan A.** Run the 120–240 runs as a batch across all four GPUs (a simple job array, one subspace run per GPU at a time), **before** the main block — which hands you the ecological-fitting / short-term-evolvability baseline up front — and/or after. Not concurrent with the main fine-tuning block.

**Logging.** Write each run's `(condition, d, replicate_seed, perf, steps, criterion_met)` to a separate `d90` table in the same store, plus the per-`d` performance curve. Derive `d₉₀` per `(condition, replicate)` and report mean ± CI per condition.

**Runner spec (one independent subspace run; launch many across GPUs).**

```python
def run_subspace(condition, d, seed, init_ckpt, data, perf_full, perf_init,
                 step_cap, patience, eps, eval_every):
    model = load(init_ckpt)                       # this condition's init
    freeze_embedder_and_positions(model)          # same exclusions as the main runs
    theta0 = trainable_params_flat(model).detach()
    P = SeededSparseOrHashProjection(D=theta0.numel(), d=d, seed=seed)  # never dense; matvec + rmatvec matrix-free
    v   = zeros(d, requires_grad=True)
    opt = adam([v])                               # ONE optimizer for all d90 runs; LR/protocol comparable to main
    target = perf_init + 0.9 * (perf_full - perf_init)
    best, since = perf_init, 0
    for step in range(step_cap):
        set_trainable_params(model, theta0 + P.matvec(v))    # theta = theta0 + P v
        loss = task_loss(model, next(data)); loss.backward()  # grad reaches v through P
        opt.step(); opt.zero_grad()
        if step % eval_every == 0:
            p = held_out_metric(model)
            if p >= target:        return dict(criterion_met=True,  steps=step, perf=p)
            if p > best + eps:     best, since = p, 0
            else:                  since += 1
            if since >= patience:  break          # converged below the criterion
    return dict(criterion_met=(best >= target), steps=step, perf=best)
# d90(condition, replicate) = smallest d on the grid with criterion_met=True.
```

### 9.2 Other post-hoc protocols (over the retention ladder)

- **Mode connectivity** between the three endpoints and along trajectories: post-hoc over retained-ladder checkpoints (another reason the ladder must exist) — does a low-loss path connect them, the operational form of the holey-landscape / reachable-basin question.
- **Filter-normalized loss-landscape slices** (Li et al.): many forward passes per slice; ladder-only or post-hoc.

---

# Do not (whole document)

From the experiment setup (Part I):

- Do **not** cache post-tokenizer activations to disk; store images at training resolution as WebP/AVIF and run the frozen tokenizer on the fly (Part I §5).
- Do **not** label the very naive baseline "from scratch," "random init," "neutral," or a "zero-history null" in any logged metadata — it carries a pretrained tokenizer (Part I §3.3).
- Do **not** merely set the frozen tokenizer and positional embeddings to LR 0 — exclude them from the optimizer param groups entirely, so no weight decay or optimizer state touches them (Part I §4.1).
- Do **not** fine-tune any condition at a resolution different from the one the positional embeddings were learned at while positions are frozen (Part I §4.4).

From the extractor and orchestration (Part II):

- Do **not** delete every checkpoint — keep the logarithmic retention ladder (Part II §2). A deleted checkpoint is a quantity you can never recompute and a post-hoc analysis you can never run.
- Do **not** delete a checkpoint before its extracted record is durably written and verified (Part II §2.1).
- Do **not** materialize a dense `[d × n_params]` random projection (Part II §4) — infeasible at ViT scale; use a seeded sparse or hashing-based projection.
- Do **not** change the probe set, the random-projection seed, or the Hessian batch between conditions or across checkpoints (Part II §4).
- Do **not** include the frozen patch-embedding / positional-embedding parameters in any displacement, projection, or rank statistic — they contribute zero and dilute every per-layer signal.
- Do **not** trigger the sampler on raw (unnormalized) global weight norm, or on loss (Part II §3).
- Do **not** present the QK/OV decomposition or the linear probes as more than they are (Part II §6.5, §6.7 caveats).
- Do **not** assert the flat-direction-churn reading; it is a hypothesis these diagnostics test (Part II §6.3d).

---

# Escalate if (whole document)

Experiment setup (Part I):

- Tokenizer or positional-embedding drift between VT and ImageNet is large — it breaks the controlled-constant input stage (Part I §6).
- VT is self-supervised and the uniform-LR setting visibly disadvantages it (Part I §4.3).
- The very naive baseline fails to train to a meaningful point even with the frozen tokenizer — the reduced-data budget may be too small even for the downstream-only problem.

Extractor and orchestration (Part II):

- Optimizer state is **not** saved in the checkpoints — the metric-evolution signal (Part II §6.1), the project's headline, cannot be recovered. Stop and get optimizer-state checkpointing added.
- The extractor cannot keep up after raising the min-step gap and reducing probe-set size — disk is at risk. Throttle emission at the producer, or move to a faster store, before letting disk fill.
- The frozen-embedder assertion (Part II §6.6) is nonzero — the freezing is broken and every cross-condition contrast is confounded.
- The optimizer's target/`M` structure cannot be cleanly classified as evolving vs constant-floor (Part II §8) — flag it rather than guessing.

---

# Combined checklist

**Experiment setup (Part I):**

- [ ] Confirmed ViT variant / patch size / resolution / `D` match across all three checkpoints
- [ ] Measured input-stage drift `‖E_VT − E_ImageNet‖` **and** `‖pos_VT − pos_ImageNet‖`; small → use one identical frozen ImageNet tokenizer + positions for all three; large → escalate
- [ ] Confirmed VT pretraining type (supervised vs MIM/contrastive)
- [ ] Built the very naive baseline = ImageNet tokenizer + positions + reinitialized everything else
- [ ] Tokenizer **and** positions frozen (`requires_grad=False`, excluded from optimizer) in all three; CLS trainable in all three
- [ ] Uniform LR (no layer-wise decay); MIM caveat checked
- [ ] Images stored as WebP/AVIF at training resolution; augment in pixel space; tokenizer on the fly; **no token caching**

**Extractor, orchestration, and intrinsic dimension (Part II):**

- [ ] Confirmed the training job saves **optimizer state** in every checkpoint
- [ ] Extractor runs on a GPU node with access to the checkpoint storage
- [ ] Atomic write + sentinel; `.processing` lock; write→fsync→verify→delete ordering; processed-step idempotency
- [ ] Retention ladder (logarithmic) implemented, decoupled from extraction cadence
- [ ] Adaptive sampler: scale-free per-layer net-displacement trigger, min-step floor, max-step cap; `delta/k/K` set from a pilot
- [ ] Comparability invariants frozen and in the manifest: probe set (+split), random-projection seed/dim/method, Hessian batch, architecture, optimizer config
- [ ] Inline per-step logging in place (fitness, displacement, velocity dot-products, optimizer scalars)
- [ ] Optimizer state with the **M block gated on `detect_evolving_M`** (Part II §6.1, §8); both paths produce valid records
- [ ] Dimension-reduced weights (shared frozen projection, never dense), per-layer change profile, displacement-on-`Q_0` (Part II §6.2)
- [ ] Trajectory geometry: straightness, MSD(+plateau), velocity autocorrelation, spectral flat/steep split — global and per-layer (Part II §6.3)
- [ ] Hessian/Fisher spectra on the fixed batch; participation ratio (Part II §6.4)
- [ ] QK/OV circuit spectra + OV sign structure + circuit drift, per head (Part II §6.5)
- [ ] Attention distance/entropy, CLS concentration, per-head/per-layer gradient rank, CKA, frozen-embedder assertion (Part II §6.6)
- [ ] Per-layer image probe + per-patch classification + emergence depth + sparse example maps (Part II §6.7)
- [ ] Held-out fitness; initial/slope/final captured; zero-shot if available (Part II §6.8)
- [ ] Parquet + Zarr + manifest schema wired; inline and extracted records share `run_id` and `step`
- [ ] Plan A layout: 3 trainers (one per condition) + 1 shared collector on 4 GPUs; collector off the trainers' GPUs (Part II §2.3)
- [ ] Shared `run_id`-tagged queue, sequential processing, oldest-first with per-run fairness; pilot utilization check < ~0.7 (Part II §2.3)
- [ ] Init-d₉₀ batch: θ₀+Pv, never-dense projection, geometric d-grid, 5–10 replicates/condition, 90%-of-gain criterion, step cap matched to budget, one fixed optimizer; run before/after on all 4 GPUs (Part II §9.1)
- [ ] Mode connectivity + landscape slices left to post-hoc over the ladder (Part II §9.2)

---

*Single handoff document for the PhenoVision experiment (Part I) and its checkpoint-extraction and orchestration pipeline (Part II). Part I §6 (verification) gates the validity of the experimental design and must be run first; Part II §8 (optimizer-variant detection) is the one place that must be reconciled against the actual optimizer class on the training machine.*
