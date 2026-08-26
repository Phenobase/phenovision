# xAI Analysis Log

## 2026-03-16: Initial Fitness Landscape Analysis

### Session Goal
Generate presentation-quality figures comparing PlantCLEF ("Virtual Taxonomist") and MAE ("ImageNet Generalist") pretraining strategies, framed through evolutionary biology concepts for a talk.

### Fitness Conversion Choice: exp(-loss)

We convert BCE training loss to a Wrightian relative fitness via W = exp(-L):
- Range (0, 1) like biological relative fitness
- Random baseline (loss = ln(2)) maps to exactly 0.5 — intuitive starting point
- Converged models approach but never reach 1.0 — "no organism is perfectly adapted"
- Connection to Malthusian fitness: loss as maladaptation rate, exp(-loss) as survival probability

### The Random Head Problem

Both models start at identical loss (0.6931) because the classification head is randomly initialized. This is important theoretically:

**What we can claim (simplified story):** The random head deliberately removes the "ecological fitting" component (Component 1 from the notes), cleanly isolating the "evolvability" signal (Component 2a). Both lineages start at the same fitness and we watch which one adapts faster.

**What's actually happening (more subtle):** The random head *masks* ecological fitting — the backbone representations may already be closer to a good solution (PlantCLEF) or further away (MAE), but we can't see this through the random head's noise. To truly measure ecological fitting, we would need to:
1. Freeze the backbone
2. Train only the classification head to convergence
3. Compare converged head-only performance (= quality of pretrained representations = ecological fitting)
4. Then unfreeze and track adaptation dynamics (= evolvability)

This two-phase experiment is future work. For now, the simplified story is defensible for the talk audience.

### Observations from Figures

#### Figure 1: Fitness Trajectory
- Both models start near 0.8 in the smoothed view (the 0.5→0.8 jump happens within the first ~75 smoothing-window batches)
- Virtual Taxonomist is consistently above ImageNet Generalist at every point
- The gap is largest in the first ~10 epochs, then narrows but never closes
- Both approach an asymptote but Virtual Taxonomist's is higher (~0.975 vs ~0.951)
- Interpretation: The "preadapted" lineage both adapts faster AND reaches a higher fitness ceiling

#### Figure 2: Rate of Adaptation (Gradient)
Surprising finding: **ImageNet Generalist has a higher initial gradient spike (~0.00015) than Virtual Taxonomist (~0.00010).**

This is counterintuitive but interpretable: the ImageNet model starts at lower fitness, so it has more room to improve. The steeper initial gradient doesn't mean better evolvability — it means the model is further from any optimum and the landscape is steeper (higher loss = higher gradient of exp(-loss) because d/dt[exp(-L)] = -exp(-L) * dL/dt, and exp(-L) is smaller when L is larger).

**The real evolvability signal is in the sustained rate:** After epoch ~3, Virtual Taxonomist maintains a higher adaptation rate than ImageNet Generalist (visible in the zoomed early plot). This is where the pretraining advantage manifests — not as a bigger initial burst, but as more efficient continued improvement.

This maps beautifully to the theoretical framework:
- The initial spike is mostly about landscape steepness at the starting position
- The sustained rate reflects the quality of the optimization geometry — smoother, better-conditioned loss surface around the pretrained initialization
- This is Component 2a (short-term evolvability) in the notes

#### Figure 3: J-index (True Skill Statistic)
- Confirms the published result: PlantCLEF peaks at epoch 3 (mean J-index 0.864), MAE at epoch 11 (0.835)
- PlantCLEF is higher for both flower and fruit components
- The fruit component shows the largest absolute gap — consistent with the idea that PlantCLEF pretraining is particularly helpful for the harder classification task (fruits are more ambiguous than flowers)
- After peak, both models show gradual decline (overfitting to training data at the expense of generalization)

#### Figure 4: J-index Gradient
- Dominated by the massive epoch 1→2 jump (both go from ~0 to ~0.85 mean J-index)
- Virtual Taxonomist's jump is slightly larger
- After epoch 3, both hover near zero with small fluctuations
- Less informative than the batch-level gradient plot — epoch granularity is too coarse to see the adaptation dynamics clearly

### Figures Generated
- `xAI/figures/fitness_vs_training_step.png` — Figure 1
- `xAI/figures/rate_of_adaptation.png` — Figure 2 (full)
- `xAI/figures/rate_of_adaptation_early.png` — Figure 2b (zoomed first 10 epochs)
- `xAI/figures/jindex_fitness_by_epoch.png` — Figure 3
- `xAI/figures/jindex_gradient_by_epoch.png` — Figure 4

### Intermediate Data
- `xAI/output/batch_fitness_combined.csv` — batch-level fitness with smoothing and gradients
- `xAI/output/epoch_validation_combined.csv` — per-epoch validation metrics

### Code
- `xAI/R/pretraining_fitness_figures.R` — complete analysis script

### Revision 2: Validation Fitness Reveals Conceptual Complexity (2026-03-16)

Switched to validation loss as the fitness measure (exp(-val_loss)). Key finding:

**Validation loss and j-index peak at completely different epochs:**
- J-index (TSS) peaks at epoch 3 (PlantCLEF) and 11 (MAE) — this is the published result
- Validation loss peaks at epoch ~38-42 for both models — much later

**Why they diverge:** Loss measures probability calibration (how well-calibrated are the predicted probabilities?). J-index measures discrimination (can the model separate positive from negative at a threshold?). A model can continue improving its calibration long after its discrimination starts declining due to overconfidence.

**Implications for the fitness analogy:**
- Neither training loss nor validation loss is a perfect fitness analog
- J-index (discrimination) is closest to biological fitness (can the organism do the thing?)
- But we only have j-index per epoch, not per batch
- Training loss gives batch-level resolution but measures "lab fitness" not "field fitness"

**Resolution options:**
1. **For the talk (pragmatic):** Use training fitness for the dramatic batch-level trajectory, j-index briefly for the validation story. Acknowledge the distinction.
2. **For the paper (rigorous):** Retrain with validation evaluation every ~50 batches, logging both loss and j-index. This gives "field fitness" at batch resolution.
3. **For the ecological fitting decomposition:** Two-phase training (freeze backbone → unfreeze) with frequent validation logging.

### Figures Generated (Revision 2)
- `xAI/figures/val_fitness_vs_epoch.png` — Validation fitness, full 47 epochs
- `xAI/figures/val_fitness_vs_epoch_zoomed.png` — Same, first 20 epochs
- `xAI/figures/val_fitness_gradient.png` — Validation fitness gradient per epoch
- `xAI/figures/train_fitness_vs_step.png` — Training fitness, batch-level (relabeled)
- `xAI/figures/train_fitness_gradient_early.png` — Training fitness gradient, first 10 epochs
- (Older figures still in directory: fitness_vs_training_step.png, rate_of_adaptation*.png, jindex*.png)

### xAI Retraining Runs Designed (2026-03-16)

**Two-phase training scripts created** in `xAI/R/xai_train.R` + `xAI/py/xai_engine.py`:
- Phase 1 (Equalization): Frozen backbone, train head only → reveals ecological fitting
- Phase 2 (Evolution): Full model unfrozen → measures evolvability
- Validation every 100 batches on fixed 10K subset (AUC-ROC, AUC-PR, val_loss)
- Representation snapshots every 500 batches (1K images, 1024-dim features)
- Gradient norms per batch (total + per-layer)
- Hybrid checkpoint schedule: frequent early, per-epoch later (~235 GB total for both models)

**SLURM scripts**: `xAI/scripts/submit_xai_{plantclef,mae}.sh` (B200 GPU, 24h walltime)

**Blocker**: PlantCLEF base model file (`models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth`) is missing. Russell is recovering it. MAE run can proceed immediately.

**Data decision**: Using v1.1.0 data (original March 2025 training data also gone).

### Two-Phase Training Results (2026-03-17)

Both runs completed successfully (~9 hours each on B200 GPUs, 214K training images).

**Phase 1 (Equalization) — Ecological Fitting:**
- PlantCLEF: AUC-ROC 0.923, converged in 1,400 steps (5 epochs)
- MAE: AUC-ROC 0.818, converged in 2,000 steps (7 epochs)
- Gap: 0.105 — PlantCLEF representations immediately more useful for phenology

**Phase 2 (Evolution) — Evolvability:**
- PlantCLEF: peaked at ~0.968 (asymptote at p2_batch 1,200)
- MAE: peaked at ~0.948 (asymptote at p2_batch 3,000)
- Final gap: 0.020 — ecological fitting accounts for 5x more of the advantage than long-term evolvability

**Three-component decomposition:**
1. Ecological fitting (where you start): PlantCLEF 0.923 vs MAE 0.818 (gap = 0.105)
2. Long-term evolvability (where you end): PlantCLEF 0.968 vs MAE 0.948 (gap = 0.020)
3. Short-term evolvability (rate): PlantCLEF 0.038 vs MAE 0.043 AUC/1000 batches (nearly identical)

**Learning rate finding:** Both models had effectively zero base learning rate during Phase 2 (cosine schedule decayed by epoch 6-8). Adaptation driven entirely by AdamW momentum. The parallel rates of fitness gain are genuine, not an LR artifact.

### Weight Displacement Analysis (2026-03-17)

Computed L2 displacement in parameter space from Phase 2 start to peak checkpoints (`xAI/py/weight_displacement.py`). Full trajectories saved in `xAI/output/{plantclef,mae}_weight_displacement.csv`.

**Terminology convention:** We use biological convention where "deep layers" = near the input (foundational, general features like edges/textures/shapes — the body plan), "shallow layers" = near the output (task-specific, adaptive features — the phenotype). This is opposite to the common ML convention where "deep" means further from input.

**Total displacement (L2 distance start→peak):**
- PlantCLEF: 53.35 — moved further despite needing less fitness gain
- MAE: 33.00 — moved less but also gained less

**Per-layer displacement (biological convention):**

| Layer | Depth | PlantCLEF | MAE | Interpretation |
|-------|-------|-----------|-----|----------------|
| blocks.0 (deepest) | Deep | 0.003 | 0.005 | Conserved core — barely changed |
| blocks.1 | Deep | 0.005 | 0.007 | Conserved core |
| blocks.2-22 (middle) | Middle | 29.08 | 17.73 | Moderate restructuring |
| blocks.22 | Shallow | 34.97 | 17.05 | Heavy adaptation |
| blocks.23 (shallowest) | Shallow | 27.84 | 21.97 | Heavy adaptation |
| head | Output | 1.51 | 1.19 | Small (only 2050 params) |

**Key observations:**
- Deep layers (near input) are essentially conserved for both models — the fundamental visual feature extraction is shared and doesn't need adaptation. This is the "body plan" — whether pretrained on plant species or ImageNet, the low-level visual features are similar.
- Shallow layers (near output) show the most change — this is where task-specific adaptation happens. These are the "adaptive traits."
- PlantCLEF moved MORE in the shallow layers despite needing less fitness gain. This suggests the PlantCLEF backbone was actively restructuring its shallow representations from species-classification features toward phenology-classification features — a genuine repurposing of existing structure (exaptation).
- MAE moved less overall, possibly because its generic representations required less restructuring — they weren't specialized for anything, so there was less to "unlearn."
- The conservation of deep layers across both pretraining strategies echoes Kirschner & Gerhart's "facilitated variation" — conserved core processes (deep visual features) combined with evolvable regulatory interfaces (shallow task-specific layers).

**Per-parameter displacement** (controls for layer size):
- PlantCLEF shallow (blocks.22): 0.0099/param; MAE: 0.0048/param — PlantCLEF changed 2x more per parameter
- This reinforces the exaptation interpretation: PlantCLEF had more to restructure because its shallow layers were specifically organized for species classification

**Displacement at asymptote (start → fitness peak):**
- PlantCLEF at p2_batch ~1200 (step ~2600): displacement ≈ **30**
- MAE at p2_batch ~3000 (step ~5000): displacement ≈ **33**
- Nearly identical distance! Both traverse roughly the same distance to reach their peak.

**Post-peak wandering:** PlantCLEF continued evolving past its fitness peak — by epoch 14, total displacement reached 56.3 (vs 30 at peak). MAE also continued but wandered less (33 at peak → 33 at end). This suggests PlantCLEF's loss landscape allows more exploration without fitness penalty (flatter plateau), while MAE sits in a tighter basin.

**Key insight:** The preadapted model doesn't take a shortcut through weight space — it takes an equally long path but arrives at a better destination. Ecological fitting is about the DIRECTION of available evolutionary trajectories, not the distance traveled. Same evolutionary distance, different basins of attraction.

### Figures Generated (2026-03-17)
- `xai_01_hero_trajectory.png` — Complete fitness trajectory with phase transition markers
- `xai_02_phase1_ecological_fitting.png` — Phase 1 zoom with ecological fitting gap
- `xai_03_phase2_evolution.png` — Phase 2 aligned with rate shading and asymptote detection
- `xai_04_decomposition.png` — Stacked bar: ecological fitting + evolutionary gain
- `xai_05_summary_metrics.png` — Three-panel: start/peak fitness, time to peak, rate
- `xai_06_per_class_auc.png` — Flower vs fruit AUC-ROC (Phase 2 only)
- `xai_07_gradient_norms.png` — Per-layer gradient magnitude over Phase 2
- `xai_08_weight_distance.png` — Cumulative gradient path length
- `xai_09_rate_of_adaptation.png` — d(AUC-ROC)/d(step) during Phase 2
- `xai_10_val_loss_trajectory.png` — Validation loss (shows overfitting dynamics)
- `xai_11_val_fitness.png` — exp(-val_loss) fitness measure

### Data Files (2026-03-17)
- `xAI/output/xai_validation_combined.csv` — All validation metrics both models
- `xAI/output/xai_summary_metrics.csv` — Summary statistics
- `xAI/output/xai_weight_distance.csv` — Gradient-based path length
- `xAI/output/plantclef_weight_displacement.csv` — Per-checkpoint L2 displacement from Phase 2 start
- `xAI/output/mae_weight_displacement.csv` — Same for MAE

### Future Steps
1. **Weight displacement trajectory figures** — plot displacement vs training step
2. **Representation analysis** — UMAP/PCA on the saved feature snapshots to visualize representation evolution
3. **Intrinsic dimensionality (d_90)** — future work requiring specialized training
4. **LR schedule investigation** — the near-zero LR in Phase 2 merits exploration; restarting the cosine schedule at Phase 2 onset could reveal different dynamics
