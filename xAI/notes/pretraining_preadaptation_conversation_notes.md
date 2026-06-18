# Pretraining as Preadaptation: Extended Conversation Notes

## Summary of Discussion Following Initial Research Reports

These notes capture the theoretical development that emerged through extended conversation, building on two prior research documents: (1) "Preadaptation, Exaptation, and Evolvability: A Conceptual Map for the Pretraining Analogy" and (2) "Plasticity, Deep History, and the Terminology of Evolutionary Legacy in AI Pretraining." The material here goes substantially beyond those documents into new theoretical territory.

---

## 1. Decomposing Ecological Fitting from Evolvability in PhenoVision

### The identification problem

If a pretrained model converges to better final performance than a randomly initialized model, the improvement could stem from two distinct mechanisms that the framework treats as fundamentally different:

- **Ecological fitting (Component 1):** The pretrained weights are already near a good local optimum. The model starts in the right neighborhood — proximity, not speed.
- **Short-term evolvability (Component 2a):** The pretrained model traverses the loss landscape more efficiently per gradient step. The geometry of the loss surface around the pretrained initialization is more favorable (smoother, better-conditioned Hessian, aligned with useful directions).

### Proposed measurements to separate the components

- **Loss trajectory analysis:** Plot loss per iteration in early training. If ecological fitting dominates, expect lower initial loss but similar rate of decrease. If evolvability dominates, expect comparable initial loss but steeper descent.
- **Weight displacement analysis:** Track cumulative parameter distance from initialization (||θ_t − θ_0||). If ecological fitting dominates, the pretrained model converges with less total displacement.
- **Efficiency ratio:** Δloss / Δ||θ|| gives "fitness gain per unit of evolutionary change," conceptually close to Hansen's conditional evolvability. Higher ratio for pretrained models = evidence for Component 2a.

---

## 2. Intrinsic Dimensionality as a Measure of Pretraining's Effect

### The Aghajanyan et al. (2021) method

Based on Li et al. (2018, "Measuring the Intrinsic Dimension of Objective Landscapes"):

- Reparameterize fine-tuning: θ = θ₀ + Pv, where P is a random D × d projection matrix (drawn once, frozen), and v is a d-dimensional vector initialized at zero.
- Train only v — all updates confined to a random d-dimensional subspace.
- Sweep d across values, measure performance at convergence.
- **d₉₀** = smallest d achieving 90% of full fine-tuning performance.
- Key finding: more pretraining decreases d₉₀. Pretrained models are fine-tunable in surprisingly low-dimensional subspaces.

### Practical considerations

- Concentration of measure in high-dimensional spaces means variance across random projections is low — random subspaces are surprisingly consistent. The Johnson-Lindenstrauss phenomenon.
- However, for comparing d₉₀ between pretrained and random-init models, the variance negligible for single estimates could still be large relative to the difference being detected. **5–10 replicates per d value per condition** recommended for proper confidence intervals.
- Random projections treat all parameters as exchangeable, but network parameters are structurally non-exchangeable. LoRA is more biologically interesting: it constrains adaptation to low-rank perturbations *within each weight matrix*, preserving modular architecture — analogous to developmental constraints channeling variation through specific regulatory interfaces.

### Connection to biological evolution

The speculation that biological evolution tends to decrease intrinsic dimensionality of organismal traits connects to multiple established concepts:

- **Canalization** (Waddington) is literally dimensionality reduction — the epigenetic landscape channels development along fewer trajectories.
- **G-matrix eigenvalue spectra** are empirically skewed: few principal components capture most heritable variation (Schluter's "genetic lines of least resistance").
- **Watson & Szathmáry's Hebbian learning mechanism:** Gene regulatory networks encode correlations from past selective environments, creating modular structure equivalent to reducing the rank of the variational structure.
- **Facilitated variation** (Kirschner & Gerhart 2007): Conserved core processes with weak regulatory linkage mean evolution operates through a small number of regulatory "knobs."

**Theoretical statement:** Pretraining (biological or artificial) reduces intrinsic dimensionality by encoding distributional structure from the training environment into the model's parameter geometry, such that future adaptation requires changes along fewer effective dimensions.

---

## 3. Three Distinct Sources of Low Dimensionality in Adaptation

A key conceptual clarification emerged: there are three separable mechanisms producing "low-dimensional" adaptive dynamics, and the biological and ML cases differ in which ones operate.

### (1) Restricted variation (the G-matrix effect)

The organism can only explore directions where heritable variation exists. The fitness landscape might be high-dimensional locally, but the organism only "sees" the projection onto the subspace spanned by the G-matrix's major eigenvectors. A filter on *input to the process*.

### (2) Local landscape geometry (the Hessian effect)

At a given point in parameter/genotype space, the fitness surface may have curvature in only a few directions. Most directions are flat. A property of the landscape *at that location*. Even an omnidirectional optimizer would find only a few directions matter.

### (3) Position on the landscape

Where you are determines which local Hessian you encounter. Pretraining navigates the model to a position where the Hessian for downstream tasks has few large eigenvalues.

**Key insight:** In gradient-based ML training, mechanism (1) is essentially absent — SGD can move in any direction. So when pretraining reduces intrinsic dimensionality, it must be doing so through (3): steering to a position where (2) holds. In biological evolution, all three operate simultaneously and are deeply confounded. The G-matrix evolves to *align with* the directions of fitness curvature (Watson & Szathmáry), so (1) and (2) become correlated.

**Novel observation:** From the perspective of evolutionary outcome, mechanisms (1) and (2) are degenerate — they produce the same observable signature. The ML case allows clean disentanglement because you can hold position fixed while varying dimensionality of available variation (random subspace method), or hold variation unconstrained while varying position (pretrained vs. random init with full-rank fine-tuning).

**Proposed concept:** "Evolution reduces the effective dimensionality of the adaptive landscape *as experienced by the lineage*" — restating the facilitated variation thesis in geometric language.

---

## 4. The G(x) = C(x) + V(x) Decomposition

### Position-dependent constraints as Riemannian geometry

A fixed G-matrix is a linearization valid only locally. Making G position-dependent — **G(x)** — turns parameter space into a Riemannian manifold, where G(x) is the metric tensor defining local geometry of available variation.

The update rule θ_{t+1} = θ_t − ηG(θ_t)∇L(θ_t) is Riemannian gradient descent. Formally equivalent to natural gradient descent (Amari 1998), but the conceptual move is different: imposing a biologically-inspired G to simulate constrained variation.

### The two-component decomposition of constraints

**G(x) = C(x) + V(x)**

- **C(x):** Extrinsic constraints — biomechanical limits, thermodynamic boundaries, physical laws, body plan features effectively frozen on relevant timescales. The floor and walls of the room.
- **V(x):** Intrinsic variation-generating architecture — the developmental-genetic machinery determining which mutations produce which phenotypic effects. Subject to selection and drift. Where the furniture goes.

### Why this decomposition matters

It resolves a persistent confusion in the constraint-vs-adaptation debate about G-matrix orientation. When the G-matrix's leading eigenvectors align with evolutionary divergence (Schluter 1996; Bégin & Roff 2003), two interpretations get conflated:

1. Alignment from immutable constraint (C) — evolution goes that direction because it has no choice.
2. Alignment from evolved architecture (V) — the variation-generating system has been shaped by selection to produce variation along historically useful directions.

These have completely different implications for evolvability. Under (1), the organism is trapped. Under (2), it's well-tuned and could retune.

### Coupled two-timescale dynamics

- **Fast:** Δz̄ = [C(x) + V(x)]β — phenotypic evolution under current constraints
- **Slow:** ΔV = f(β, landscape curvature, current V) — evolution of variation-generating architecture to match the adaptive landscape

V(x) should tend to align with high-curvature fitness directions *not already covered by C(x)* — an economy-of-variation argument.

### Manifold curvature and the evolvability paradox

Low-rank G(x) with principal eigenvectors that rotate smoothly as x varies means the organism is confined to a low-dimensional manifold, but if the manifold is curved, the total accessible phenotype space over evolutionary time can be much larger than what's locally accessible. This resolves how organisms can be simultaneously constrained (low local dimensionality) and evolvable (capable of reaching distant innovations). The manifold is low-dimensional but highly curved and well-connected — Gavrilets' neutral network insight in differential-geometric language.

### Evolutionary dead-ends as geometric features

An evolutionary dead-end: a point where manifold curvature goes to zero in relevant directions — the manifold becomes locally flat, the lineage is stuck on a low-dimensional linear subspace. Specialist evolution freezes the rotation of principal directions. Measurable as the rate of rotation of principal eigenvectors of gradient covariance across training steps.

---

## 5. Refining the C(x) Component: Architecture as Constraint

### Three levels of architectural constraint (hierarchical)

1. **Hard representational limits** — functions the architecture literally cannot compute at a given size. Analogous to absolute physical constraints (thermodynamics, etc.). Strongest but least interesting.
2. **Soft dynamical constraints** — functions the architecture can represent but SGD is unlikely to find. Spectral bias, simplicity bias, implicit regularization. Analogous to developmental bias. Strong enough to channel but potentially overcomeable.
3. **Inductive biases** — preferences, not barriers. The ViT's attention mechanism biases toward spatial relationships but doesn't prevent learning local features. Analogous to genetic lines of least resistance.

### Universal approximation doesn't undermine the constraint story

Universal approximation is an existence theorem, not a reachability theorem. A genome is also in principle a universal function approximator. What matters is what's reachable from where you are, given the variation available, in the time you have. The architecture constrains not what the model could theoretically represent but the topology and geometry of the path through parameter space that training actually follows.

Valle-Pérez et al. (2019): SGD on neural networks is biased toward low-complexity functions through a mechanism entirely architectural — the parameter-to-function map assigns exponentially more parameter configurations to simple functions.

### Refined decomposition

- **C_hard:** Representational limits of the architecture. Truly fixed.
- **C_soft(x):** Architecture's implicit dynamical biases at a given parameter configuration. Position-dependent. Partially "evolvable" in that pretraining changes position, which changes C_soft.
- **V(x):** Learnable preconditioner explicitly modifying optimization dynamics.

**Key insight:** Standard pretraining operates on C_soft, not V. It modifies parameters (position), which changes the position-dependent dynamical constraints. This is ecological fitting: the pretrained model starts where existing architectural biases happen to be useful. Adding a learnable V on top is genuinely novel — Component 2 layered on Component 1.

---

## 6. Vision Transformer Specifics

For ViT-based experiments, the attention blocks are where the most interesting constraint-variation interplay lives:

- **QKV projections:** Fixed dimensionality per head (d_k = d_model / n_heads) is a hard architectural constraint. Which features each head learns to attend to is V.
- **Attention head specialization as evolved modularity:** Heads specialize during pretraining (local texture, global structure, etc.). This is evolution of V. Measurable: per-head gradient covariance rank during fine-tuning, pretrained vs. random init.
- **Attention pattern as reaction norm:** Fixed weights, variable attention maps across inputs = fixed genotype, variable phenotype across environments. The diversity of attention patterns measures plasticity of each module.
- **Residual stream as shared developmental field:** Dimensionality of residual stream representations at each layer (via PCA of activations) tracks effective degrees of freedom.

### Concrete experimental design for PhenoVision

During fine-tuning of pretrained vs. random-init ViTs on phenology task, at regular intervals compute:

1. **Per-layer gradient covariance** eigenspectrum → effective rank per layer
2. **Per-head gradient rank** → whether individual modules show dimensionality reduction
3. **Gradient alignment with pretrained representation structure** → project fine-tuning gradients onto PCs of pretrained activations; angle between gradient subspace and pretrained feature subspace quantifies alignment of V with current task demands

---

## 7. Simplicity Bias: The Mathematical Bridge Between ML and Evo-Devo

### The Dingle, Camargo & Louis (2018) result

For any compressive structured map from high-dimensional input space to lower-dimensional output space, the probability of producing an output x decays approximately exponentially with its Kolmogorov complexity: P(x) ≲ 2^{−aK̃(x) − b}.

This is not about neural networks specifically. It applies to **any compressive generative map with structure** — and the genotype-to-phenotype map is exactly such a map.

**Reference:** Dingle, K., Camargo, C.Q. & Louis, A.A. (2018). Input–output maps are strongly biased towards simple outputs. *Nature Communications* 9, 761. https://doi.org/10.1038/s41467-018-03101-6

### Connection to biology

Development has a simplicity bias for the same mathematical reason as neural networks: both are compressive structured maps, and Levin's coding theorem guarantees outputs biased toward low algorithmic complexity.

Existing related ideas in evo-devo not previously connected to AIT:

- **Wagner's "Arrival of the Fittest"** — the GP map determines which innovations are accessible; highly accessible phenotypes have large neutral networks = large preimages = low complexity outputs
- **Kauffman's "order for free"** — self-organization produces ordered behavior because ordered states occupy more state space
- **Facilitated variation** — conserved core processes generate variation biased toward functional outcomes

**What the ML theory adds:** A quantitative prediction (P(x) ≈ 2^{−aK−b}) that could be tested empirically against phenotypic complexity distributions.

### What the complexity measure captures and what it misses

Dingle et al. use **Lempel-Ziv complexity** (compression ratio) as a practical approximation of Kolmogorov complexity. This captures:

- **Structural/morphological complexity** (McShea's "non-colloquial" complexity) — number of distinct parts, degree of differentiation. A segmented body with identical segments = low complexity. Every segment unique = high complexity.

It misses:

- **Functional/organizational complexity** — Adami's (2002) "physical complexity" = mutual information between genome and environment. A random string has maximum Lempel-Ziv complexity but zero functional complexity. Lempel-Ziv can't distinguish "incompressible because random" from "incompressible because informationally rich."
- The dissociation between regularity and functional elaboration: the most functionally complex organisms are often the most structurally regular (symmetry, modularity, repeated parts).

Key distinction: **Kolmogorov complexity** (total information content) vs. **effective complexity** (Gell-Mann & Lloyd 2004 — length of description of regularities only, stripping noise). Biology occupies the intermediate zone (moderate Kolmogorov complexity, high effective complexity).

### The input distribution problem

The Dingle et al. result assumes **uniform random sampling of inputs**. This is biologically unrealistic:

1. **Mutation is local:** Real organisms sample the neighborhood of their current genotype, not genotype space uniformly. The preimage of a simple phenotype might be enormous but distributed in scattered pockets unreachable from the current position.
2. **Selection structures the input distribution:** The population's location in genotype space is non-random — shaped by generations of selection and drift.
3. **Recombination creates structured input sampling:** New genotypes arise by mixing existing parental genomes, not random sampling.

Dingle et al. characterize the **prior** (what the map favors before any history). The G(x) framework characterizes the **posterior** (given specific evolutionary history, what's locally favored). These are complementary.

### Drift toward simplification

Under pure drift (unbiased random walk on the neutral network), populations should gravitate toward simpler phenotypes — not because simplicity is selected for, but because there are more genotypes encoding simple phenotypes (more doors leading in, fewer leading out).

**This is a second law of thermodynamics for phenotypic evolution under drift.**

Empirical support: Schaper & Louis (2014, *PLOS ONE*) — "the arrival of the frequent" in RNA secondary structure; Taverna & Goldstein (2002) — drift biases toward highly designable (regular, simple) protein folds.

Complications: neutral network connectivity matters (not just preimage size); drift also increases evolvability (van Nimwegen et al. 1999 — "survival of the flattest"); mutation bias interacts with preimage-size effect; even weak selection can overwhelm the drift-toward-simplicity effect.

### Free energy of phenotypic complexity

Observed organismal complexity reflects a balance between:

- **Entropic simplification pressure** from GP map geometry (drift pushes toward large-preimage, low-complexity phenotypes)
- **Selection pressure** for adaptive complexity (when complex phenotypes are fitter)

Formally analogous to free energy F = E − TS:

- "Energy" = negative fitness
- "Entropy" = log-preimage-size
- "Temperature" ∝ 1/Nₑ (strength of drift relative to selection)

High temperature (small populations) → simplification dominates (genome reduction in parasites, endosymbionts, cave organisms). Low temperature (large populations) → complex adaptations maintained. Related to Sella & Hirsh (2005, *PNAS*) "free fitness" framework.

**ML translation:** Models trained with strong signal maintain complex representations; models with weak signal drift toward simpler internal representations. Testable: track effective complexity of representations during training on progressively noisier data.

---

## 8. Key References Identified in Discussion

- **Dingle, K., Camargo, C.Q. & Louis, A.A.** (2018). Input–output maps are strongly biased towards simple outputs. *Nature Communications* 9, 761. https://doi.org/10.1038/s41467-018-03101-6
- **Watson, R.A. & Szathmáry, E.** (2016). How can evolution learn? *Trends in Ecology & Evolution* 31(2), 147–157. https://doi.org/10.1016/j.tree.2015.11.009
- **Kouvaris, K., Clune, J., Kounios, L., Brede, M. & Watson, R.A.** (2017). How evolution learns to generalise. *PLOS Computational Biology* 13(4), e1005358. https://doi.org/10.1371/journal.pcbi.1005358
- **Valle-Pérez, G., Camargo, C.Q. & Louis, A.A.** (2019). Deep learning generalizes because the parameter-function map is biased towards simple functions. *ICLR 2019*. https://arxiv.org/abs/1805.08522
- **Aghajanyan, A. et al.** (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL-IJCNLP 2021*. https://arxiv.org/abs/2012.13255
- **Li, C. et al.** (2018). Measuring the Intrinsic Dimension of Objective Landscapes. *ICLR 2018*. https://arxiv.org/abs/1804.08838
- **Gell-Mann, M. & Lloyd, S.** (2004). Information measures, effective complexity, and total information. *Complexity* 9(1), 37–49.
- **Adami, C.** (2002). What is complexity? *BioEssays* 24(12), 1085–1094.
- **Schaper, S. & Louis, A.A.** (2014). The arrival of the frequent. *PLOS ONE* 9, e86635.
- **Sella, G. & Hirsh, A.E.** (2005). The application of statistical physics to evolutionary biology. *PNAS* 102(27), 9541–9546.
- **González-Forero, M.** (2023). A mathematical framework for evo-devo dynamics. *Theoretical Population Biology*.
- **van Nimwegen, E., Crutchfield, J.P. & Huynen, M.** (1999). Neutral evolution of mutational robustness. *PNAS* 96(17), 9716–9720.
- **Amari, S.** (1998). Natural gradient works efficiently in learning. *Neural Computation* 10(2), 251–276.
- **Martin, G. & Lenormand, T.** (2006). A general multivariate extension of Fisher's geometrical model. *Evolution* 60(4), 751–767.

---

## 9. Potential Paper Ideas Emerging from This Discussion

### Paper 1: Empirical decomposition in PhenoVision
Ground the theoretical framework in measurable training dynamics — loss trajectories, weight displacement, intrinsic dimensionality, gradient alignment. Evidence for Components 1 vs. 2a at the mechanistic level. Target: methods/ecology journal.

### Paper 2: G(x) = C(x) + V(x) on Riemannian manifolds
The full theoretical treatment: position-dependent constraints, Riemannian structure of constrained evolutionary spaces, manifold curvature as the mechanism connecting constraint to evolvability, evolutionary dead-ends as geometric features. ML experimental platform for testing predictions untestable in biology. Export quantitative genetics formalism into ML optimization. Target: interdisciplinary theory venue (Upper Bound talk, *PNAS*, *Interface*).

### Paper 3: Simplicity bias bridges ML and evo-devo
Connect Dingle-Camargo-Louis simplicity bias (static, about GP map structure) to Watson-Szathmáry evolution-as-learning (dynamic, about how evolutionary trajectories encode past environments) through the shared mathematics of compressive maps and algorithmic probability. The free-energy-of-complexity framework. Neural network experiments testing predictions about drift, selection, and equilibrium complexity. Target: *Nature Communications*, *PLOS Computational Biology*.

### Paper 4: Intrinsic dimensionality as a unifying measure
The biological conjecture that evolution reduces effective dimensionality of adaptive variation, connected to canalization, G-matrix concentration, facilitated variation, and modularity. ML experiments comparing intrinsic dimensionality across pretraining regimes. The distinction between dimensionality of variation (G-matrix rank) and dimensionality of the landscape (Hessian rank) and how they interact. Target: evolutionary theory journal or cross-disciplinary venue.
