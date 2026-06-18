# Bibliography: SGD ↔ Stochastic Breeder's Equation Bridge

A working bibliography for the formal correspondence between stochastic gradient descent and quantitative genetic dynamics, organized by topic. DOI links provided where available.

---

## 1. Lande's Quantitative Genetics Framework

The foundational papers establishing the multivariate breeder's equation, the selection gradient β, and stochastic dynamics under combined selection and drift.

- **Lande, R.** (1976). Natural selection and random genetic drift in phenotypic evolution. *Evolution* 30(2): 314–334. <https://doi.org/10.1111/j.1558-5646.1976.tb00911.x>
  *The single-trait stochastic version. Fokker-Planck (PDE) treatment of mean phenotype under selection + drift; diffusion coefficient G/N_e.*

- **Lande, R.** (1979). Quantitative genetic analysis of multivariate evolution, applied to brain:body size allometry. *Evolution* 33(1): 402–416. <https://doi.org/10.1111/j.1558-5646.1979.tb04694.x>
  *The multivariate deterministic Δz̄ = Gβ form, with β = ∇ ln W̄.*

- **Lande, R.** (1980). The genetic covariance between characters maintained by pleiotropic mutations. *Genetics* 94(1): 203–215. <https://doi.org/10.1093/genetics/94.1.203>
  *G-matrix dynamics under mutation–selection–drift balance.*

- **Lande, R. & Arnold, S.J.** (1983). The measurement of selection on correlated characters. *Evolution* 37(6): 1210–1226. <https://doi.org/10.1111/j.1558-5646.1983.tb00236.x>
  *Empirical operationalization of s = Pβ; partition of selection differential.*

- **Veller, C. & Muralidhar, P.** (2026). [Title forthcoming]. *PNAS*.
  *Notes Lande (1976) used Fokker-Planck PDEs rather than SDEs — useful framing for the novelty of the SGD bridge.*

---

## 2. SGD as a Stochastic Process and Bayesian Inference

The continuous-time SDE treatment of SGD, the OU stationary distribution, and the natural-gradient / Fisher-scoring connection.

- **Mandt, S., Hoffman, M.D. & Blei, D.M.** (2017). Stochastic gradient descent as approximate Bayesian inference. *Journal of Machine Learning Research* 18(134): 1–35. <https://jmlr.org/papers/v18/17-214.html>
  *The core paper. OU stationary distribution, optimal learning rate, preconditioning, and SGFS as natural-gradient Langevin sampling.*

- **Mandt, S., Hoffman, M.D. & Blei, D.M.** (2016). A variational analysis of stochastic gradient algorithms. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*. <https://proceedings.mlr.press/v48/mandt16.html>
  *Earlier conference version with the variational framing.*

- **Welling, M. & Teh, Y.W.** (2011). Bayesian learning via stochastic gradient Langevin dynamics. *Proceedings of the 28th International Conference on Machine Learning (ICML)*. <https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf>
  *SGLD: adding noise to SGD to sample from the Bayesian posterior.*

- **Ahn, S., Korattikara, A. & Welling, M.** (2012). Bayesian posterior sampling via stochastic gradient Fisher scoring. *Proceedings of the 29th International Conference on Machine Learning (ICML)*. <https://arxiv.org/abs/1206.6380>
  *Stochastic Gradient Fisher Scoring — preconditioning by inverse gradient covariance.*

- **Li, Q., Tai, C. & E, W.** (2017). Stochastic modified equations and adaptive stochastic gradient algorithms. *Proceedings of the 34th International Conference on Machine Learning (ICML)*. <https://arxiv.org/abs/1511.06251>
  *Mathematically rigorous justification of the continuous-time limit.*

- **Li, C., Farkhoor, H., Liu, R. & Yosinski, J.** (2018). Measuring the intrinsic dimension of objective landscapes. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/1804.08838>
  *Random subspace fine-tuning method.*

- **Aghajanyan, A., Zettlemoyer, L. & Gupta, S.** (2021). Intrinsic dimensionality explains the effectiveness of language model fine-tuning. *ACL-IJCNLP 2021*. <https://arxiv.org/abs/2012.13255>
  *More pretraining decreases intrinsic dimensionality of fine-tuning.*

---

## 3. G-Matrix Evolution and Stability

Theoretical and simulation work on how the G-matrix itself evolves, especially under stabilizing and correlational selection. Critical for the G(x) = C(x) + V(x) decomposition.

- **Jones, A.G., Arnold, S.J. & Bürger, R.** (2003). Stability of the G-matrix in a population experiencing pleiotropic mutation, stabilizing selection, and genetic drift. *Evolution* 57(8): 1747–1760. <https://doi.org/10.1111/j.0014-3820.2003.tb00583.x>
  *First major individual-based simulation of G-matrix evolution.*

- **Jones, A.G., Arnold, S.J. & Bürger, R.** (2007). The mutation matrix and the evolution of evolvability. *Evolution* 61(4): 727–745. <https://doi.org/10.1111/j.1558-5646.2007.00071.x>
  *M-matrix evolves toward alignment with the adaptive landscape — directly relevant to V(x) evolving toward task-aligned variation.*

- **Arnold, S.J., Bürger, R., Hohenlohe, P.A., Ajie, B.C. & Jones, A.G.** (2008). Understanding the evolution and stability of the G-matrix. *Evolution* 62(10): 2451–2461. <https://doi.org/10.1111/j.1558-5646.2008.00472.x>
  *Comprehensive review. Key claim: G aligns with the adaptive landscape under stabilizing + correlational selection.*

- **Jones, A.G., Bürger, R., Arnold, S.J., Hohenlohe, P.A. & Uyeda, J.C.** (2012). The effects of stochastic and episodic movement of the optimum on the evolution of the G-matrix and the response of the trait mean to selection. *Journal of Evolutionary Biology* 25(11): 2210–2231. <https://doi.org/10.1111/j.1420-9101.2012.02598.x>

- **Jones, A.G., Bürger, R. & Arnold, S.J.** (2014). Epistasis and natural selection shape the mutational architecture of complex traits. *Nature Communications* 5: 3709. <https://doi.org/10.1038/ncomms4709>

- **Schluter, D.** (1996). Adaptive radiation along genetic lines of least resistance. *Evolution* 50(5): 1766–1774. <https://doi.org/10.1111/j.1558-5646.1996.tb03563.x>
  *Original "lines of least resistance" concept — divergence aligned with G-matrix leading eigenvector.*

- **Bégin, M. & Roff, D.A.** (2003). The constancy of the G matrix through species divergence and the effects of quantitative genetic constraints on phenotypic evolution: a case study in crickets. *Evolution* 57(5): 1107–1120. <https://doi.org/10.1111/j.0014-3820.2003.tb00320.x>
  *Empirical evidence for cross-species G-matrix constancy.*

- **Hansen, T.F.** (2003). Is modularity necessary for evolvability? Remarks on the relationship between pleiotropy and evolvability. *Biosystems* 69(2-3): 83–94. <https://doi.org/10.1016/S0303-2647(02)00132-6>

- **Hansen, T.F. & Houle, D.** (2008). Measuring and comparing evolvability and constraint in multivariate characters. *Journal of Evolutionary Biology* 21(5): 1201–1219. <https://doi.org/10.1111/j.1420-9101.2008.01573.x>
  *Conditional evolvability and the geometry of constraint.*

- **Houle, D.** (1992). Comparing evolvability and variability of quantitative traits. *Genetics* 130(1): 195–204. <https://doi.org/10.1093/genetics/130.1.195>
  *Original formal definition of evolvability.*

- **Pavlicev, M. & Cheverud, J.M.** (2015). Constraints evolve: context dependency of gene effects allows evolution of pleiotropy. *Annual Review of Ecology, Evolution, and Systematics* 46: 413–434. <https://doi.org/10.1146/annurev-ecolsys-120213-091721>

- **Melo, D. & Marroig, G.** (2015). Directional selection can drive the evolution of modularity in complex systems. *PNAS* 112(2): 470–475. <https://doi.org/10.1073/pnas.1322632112>

- **Milocco, L. & Salazar-Ciudad, I.** (2022). Evolution of the G matrix under nonlinear genotype-phenotype maps. *American Naturalist* 199(3): 420–435. <https://doi.org/10.1086/717814>
  *Important counterpoint: nonlinear GP maps can prevent simple alignment results.*

---

## 4. Information Geometry, Natural Gradient, and Evolution Strategies

The CS literature on natural gradient descent and CMA-ES, which provides the formal machinery for the Fisher-scoring case of the SGD-Lande bridge.

- **Amari, S.** (1998). Natural gradient works efficiently in learning. *Neural Computation* 10(2): 251–276. <https://doi.org/10.1162/089976698300017746>
  *Foundational paper for natural gradient descent.*

- **Akimoto, Y., Nagata, Y., Ono, I. & Kobayashi, S.** (2010). Bidirectional relation between CMA evolution strategies and natural evolution strategies. *Parallel Problem Solving from Nature (PPSN XI)*: 154–163. <https://doi.org/10.1007/978-3-642-15844-5_16>

- **Akimoto, Y., Nagata, Y., Ono, I. & Kobayashi, S.** (2012). Theoretical foundation for CMA-ES from information geometry perspective. *Algorithmica* 64: 698–716. <https://doi.org/10.1007/s00453-011-9564-8>; <https://arxiv.org/abs/1206.0730>
  *Establishes CMA-ES updates as natural gradient ascent on the manifold of Gaussian distributions.*

- **Ollivier, Y., Arnold, L., Auger, A. & Hansen, N.** (2017). Information-geometric optimization algorithms: a unifying picture via invariance principles. *Journal of Machine Learning Research* 18(18): 1–65. <https://jmlr.org/papers/v18/14-467.html>
  *The IGO framework. Unifies CMA-ES, natural evolution strategies, and other algorithms under information geometry.*

- **Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. & Schmidhuber, J.** (2014). Natural evolution strategies. *Journal of Machine Learning Research* 15: 949–980. <https://jmlr.org/papers/v15/wierstra14a.html>

- **Hansen, N.** (2016). The CMA evolution strategy: A tutorial. arXiv:1604.00772. <https://arxiv.org/abs/1604.00772>

- **Shir, O.M. & Yehudayoff, A.** (2020). On the covariance-Hessian relation in evolution strategies. *Theoretical Computer Science* 801: 157–174. <https://doi.org/10.1016/j.tcs.2019.08.024>; <https://arxiv.org/abs/1806.03674>
  *Proves the covariance matrix in (1,λ)-ES converges to inverse Hessian — direct analog of the Fisher-scoring case.*

- **Martens, J.** (2020). New insights and perspectives on the natural gradient method. *Journal of Machine Learning Research* 21(146): 1–76. <https://jmlr.org/papers/v21/17-678.html>
  *Modern review of natural gradient methods.*

---

## 4b. Practical Approximations of Natural Gradient: From Adam to SOAP

The applied machine learning lineage of natural-gradient-like methods. Directly relevant for the two-timescale framing in Part E of the derivation document and for any empirical work that needs to choose an optimizer matched to the theoretical regime.

### Diagonal preconditioners (the historical default)

- **Kingma, D.P. & Ba, J.** (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/1412.6980>
  *The dominant deep-learning optimizer of the last decade. Diagonal preconditioning by exponential moving average of squared gradients, with inverse square root scaling — not natural gradient, but close enough to work.*

- **Duchi, J., Hazan, E. & Singer, Y.** (2011). Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research* 12: 2121–2159. <https://jmlr.org/papers/v12/duchi11a.html>
  *AdaGrad — the historical origin of the inverse-square-root form that Adam inherits.*

- **Kunstner, F., Hennig, P. & Balles, L.** (2019). Limitations of the empirical Fisher approximation for natural gradient descent. *Advances in Neural Information Processing Systems (NeurIPS) 32*. <https://arxiv.org/abs/1905.12558>
  *Shows that the empirical Fisher (which Adam's v approximates diagonally) is meaningfully different from the true Fisher, and that this difference matters for convergence behavior. Important sanity check for any claim that Adam approximates natural gradient.*

### Kronecker-factored preconditioners (block-diagonal natural gradient)

- **Martens, J. & Grosse, R.** (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*. <https://arxiv.org/abs/1503.05671>
  *K-FAC — block-diagonal Kronecker-factored approximation of the Fisher matrix. The breakthrough that made natural-gradient-style preconditioning tractable for deep networks.*

- **Grosse, R. & Martens, J.** (2016). A Kronecker-factored approximate Fisher matrix for convolution layers. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*. <https://arxiv.org/abs/1602.01407>

- **George, T., Laurent, C., Bouthillier, X., Ballas, N. & Vincent, P.** (2018). Fast approximate natural gradient descent in a Kronecker-factored eigenbasis. *Advances in Neural Information Processing Systems (NeurIPS) 31*. <https://arxiv.org/abs/1806.03884>
  *E-KFAC — adds a diagonal preconditioner in K-FAC's eigenbasis. Conceptual precursor to SOAP.*

- **Gupta, V., Koren, T. & Singer, Y.** (2018). Shampoo: Preconditioned stochastic tensor optimization. *Proceedings of the 35th International Conference on Machine Learning (ICML)*. <https://arxiv.org/abs/1802.09568>
  *Shampoo — Kronecker factorization across tensor dimensions of weight matrices. Alternative to K-FAC's block structure.*

- **Anil, R., Gupta, V., Koren, T., Regan, K. & Singer, Y.** (2021). Scalable second order optimization for deep learning. arXiv:2002.09018. <https://arxiv.org/abs/2002.09018>
  *Distributed Shampoo — the engineering work that made Shampoo practical at frontier scale. The 2025 AlgoPerf-winning entry was a descendant of this implementation.*

### Two-timescale methods (SOAP and successors)

- **Vyas, N., Morwani, D., Zhao, R., Shapira, I., Brandfonbrener, D., Janson, L. & Kakade, S.** (2025). SOAP: Improving and stabilizing Shampoo using Adam. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/2409.11321>
  *SOAP — Adam in Shampoo's eigenbasis. The leading non-diagonal optimizer for LLM training as of 2025. The "diagonal updates in a slowly-rotated latent space" paper.*

- **Morwani, D., Shapira, I., Vyas, N., Malach, E., Kakade, S. & Janson, L.** (2024). A new perspective on Shampoo's preconditioner. arXiv:2406.17748. <https://arxiv.org/abs/2406.17748>
  *Theoretical analysis showing Shampoo's preconditioner approximates the optimal Kronecker factorization of the Adagrad/empirical-Fisher matrix.*

- **Liu, Y., Cao, S., Chen, K. & Kingma, D.P.** (2025). Understanding SOAP from the perspective of gradient whitening. arXiv:2509.22938. <https://arxiv.org/abs/2509.22938>
  *Reframes SOAP as gradient whitening. Cleaner theoretical interpretation of why the two-timescale structure works.*

- **Eschenhagen, R., Wang, Y., Bartoldson, B.R., Kailkhura, B. & Hennig, P.** (2025). Purifying Shampoo: Investigating Shampoo's heuristics by decomposing its preconditioner. arXiv:2506.03595. <https://arxiv.org/abs/2506.03595>
  *Disentangles which of Shampoo's design choices are doing the real work.*

- **An, K., Liu, Y., Pan, R., Ren, Y., Ma, S., Goldfarb, D. & Zhang, T.** (2025). KL-Shampoo: Understanding and improving the Shampoo optimizer via Kullback-Leibler minimization. arXiv:2509.03378. <https://arxiv.org/abs/2509.03378>
  *Interprets Shampoo's preconditioner estimation as KL minimization. Directly relevant to the Bayesian / free-fitness thread — KL minimization to a posterior is structurally Iwasa-Sella-Hirsh free-fitness construction.*

- **Author(s)** (2025). DyKAF: Dynamical Kronecker approximation of the Fisher information matrix for gradient preconditioning. arXiv:2511.06477. <https://arxiv.org/abs/2511.06477>
  *Uses projector-splitting integrators to maintain better Kronecker approximations of the Fisher over training.*

- **Yang, X. et al.** (2024). An improved empirical Fisher approximation for natural gradient descent (iEF). arXiv:2406.06420. <https://arxiv.org/abs/2406.06420>
  *Adds diagonal rescaling to the empirical Fisher preconditioner to address the inversely-scaled projection issue.*

### Other relevant work

- **Frantar, E., Kurtic, E. & Alistarh, D.** (2021). M-FAC: Efficient matrix-free approximations of second-order information. *NeurIPS 2021*. <https://arxiv.org/abs/2107.03356>
  *Matrix-free full-Fisher approximations.*

- **Agarwal, N., Bullins, B., Chen, X., Hazan, E., Singh, K., Zhang, C. & Zhang, Y.** (2019). Efficient full-matrix adaptive regularization (GGT). *ICML 2019*. <https://arxiv.org/abs/1806.02958>
  *Full-matrix AdaGrad via sliding-window gradient history.*

- **Bernstein, J. & Newhouse, L.** (2024). Old optimizer, new norm: An anthology. arXiv:2409.20325. <https://arxiv.org/abs/2409.20325>
  *Frames modern optimizers (Adam, Shampoo, Muon) as steepest descent under different norms — a useful unifying perspective on the natural-gradient family.*

---

## 5. Fitness Landscapes and Topology

- **Wright, S.** (1932). The roles of mutation, inbreeding, crossbreeding, and selection in evolution. *Proceedings of the Sixth International Congress of Genetics* 1: 356–366. <http://www.esp.org/books/6th-congress/facsimile/contents/6th-cong-p356-wright.pdf>

- **Gavrilets, S.** (1997). Evolution and speciation on holey adaptive landscapes. *Trends in Ecology & Evolution* 12(8): 307–312. <https://doi.org/10.1016/S0169-5347(97)01098-7>

- **Gavrilets, S.** (2004). *Fitness Landscapes and the Origin of Species*. Princeton University Press. <https://press.princeton.edu/books/paperback/9780691119830/fitness-landscapes-and-the-origin-of-species>

- **Martin, G. & Lenormand, T.** (2006). A general multivariate extension of Fisher's geometrical model and the distribution of mutation fitness effects across species. *Evolution* 60(4): 751–767. <https://doi.org/10.1111/j.0014-3820.2006.tb01156.x>

- **van Nimwegen, E., Crutchfield, J.P. & Huynen, M.** (1999). Neutral evolution of mutational robustness. *PNAS* 96(17): 9716–9720. <https://doi.org/10.1073/pnas.96.17.9716>
  *"Survival of the flattest" — drift biases populations toward neutral-network density maxima.*

---

## 6. Evolution as Learning, Facilitated Variation

- **Watson, R.A. & Szathmáry, E.** (2016). How can evolution learn? *Trends in Ecology & Evolution* 31(2): 147–157. <https://doi.org/10.1016/j.tree.2015.11.009>
  *GRNs as Hebbian-learning networks encoding past selective environments.*

- **Kouvaris, K., Clune, J., Kounios, L., Brede, M. & Watson, R.A.** (2017). How evolution learns to generalise: using the principles of learning theory to understand the evolution of developmental organisation. *PLOS Computational Biology* 13(4): e1005358. <https://doi.org/10.1371/journal.pcbi.1005358>

- **Kirschner, M. & Gerhart, J.** (1998). Evolvability. *PNAS* 95(15): 8420–8427. <https://doi.org/10.1073/pnas.95.15.8420>

- **Kirschner, M.W. & Gerhart, J.C.** (2005). *The Plausibility of Life: Resolving Darwin's Dilemma*. Yale University Press.
  *Facilitated variation framework.*

- **González-Forero, M.** (2023). A mathematical framework for evo-devo dynamics. *Theoretical Population Biology* 155: 1–23. <https://doi.org/10.1016/j.tpb.2023.10.004>

---

## 7. Statistical Physics, Free Fitness, and Drift

- **Iwasa, Y.** (1988). Free fitness that always increases in evolution. *Journal of Theoretical Biology* 135(3): 265–281. <https://doi.org/10.1016/S0022-5193(88)80243-1>
  *Original free-fitness function. Lyapunov function for evolutionary dynamics with mutation, selection, and drift.*

- **Sella, G. & Hirsh, A.E.** (2005). The application of statistical physics to evolutionary biology. *PNAS* 102(27): 9541–9546. <https://doi.org/10.1073/pnas.0501865102>
  *Free-fitness analogy with thermodynamics; stationary distribution proportional to W̄^{2N_e}.*

- **Mustonen, V. & Lässig, M.** (2010). Fitness flux and ubiquity of adaptive evolution. *PNAS* 107(9): 4248–4253. <https://doi.org/10.1073/pnas.0907953107>

- **Barton, N.H. & Coe, J.B.** (2009). On the application of statistical physics to evolutionary biology. *Journal of Theoretical Biology* 259(2): 317–324. <https://doi.org/10.1016/j.jtbi.2009.03.019>

- **Khatri, B.S. & Goldstein, R.A.** (2015). Simple biophysical model predicts faster accumulation of hybrid incompatibilities in small populations under stabilizing selection. *Genetics* 201(4): 1525–1537. <https://doi.org/10.1534/genetics.115.181685>

---

## 8. Simplicity Bias, GP Maps, Algorithmic Information Theory

- **Dingle, K., Camargo, C.Q. & Louis, A.A.** (2018). Input–output maps are strongly biased towards simple outputs. *Nature Communications* 9: 761. <https://doi.org/10.1038/s41467-018-03101-6>

- **Valle-Pérez, G., Camargo, C.Q. & Louis, A.A.** (2019). Deep learning generalizes because the parameter-function map is biased towards simple functions. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/1805.08522>

- **Schaper, S. & Louis, A.A.** (2014). The arrival of the frequent: how bias in genotype-phenotype maps can steer populations to local optima. *PLOS ONE* 9(2): e86635. <https://doi.org/10.1371/journal.pone.0086635>

- **Taverna, D.M. & Goldstein, R.A.** (2002). Why are proteins so robust to site mutations? *Journal of Molecular Biology* 315(3): 479–484. <https://doi.org/10.1006/jmbi.2001.5226>

- **Adami, C.** (2002). What is complexity? *BioEssays* 24(12): 1085–1094. <https://doi.org/10.1002/bies.10192>

- **Gell-Mann, M. & Lloyd, S.** (2004). Effective complexity. In *Nonextensive Entropy: Interdisciplinary Applications* (M. Gell-Mann & C. Tsallis, eds.). Oxford University Press, pp. 387–398.

- **Wagner, A.** (2014). *Arrival of the Fittest: How Nature Innovates*. Current.

- **Gould, S.J.** (1996). *Full House: The Spread of Excellence from Plato to Darwin*. Harmony Books.

---

## 9. Adaptive Dynamics and Individual-Based Diffusions

The branch of theory most directly compatible with "individual-as-gradient-sample" intuition.

- **Champagnat, N., Ferrière, R. & Méléard, S.** (2006). Unifying evolutionary dynamics: from individual stochastic processes to macroscopic models. *Theoretical Population Biology* 69(3): 297–321. <https://doi.org/10.1016/j.tpb.2005.10.004>

- **Dieckmann, U. & Law, R.** (1996). The dynamical theory of coevolution: a derivation from stochastic ecological processes. *Journal of Mathematical Biology* 34: 579–612. <https://doi.org/10.1007/BF02409751>

- **Bürger, R. & Lynch, M.** (1995). Evolution and extinction in a changing environment: a quantitative-genetic analysis. *Evolution* 49(1): 151–163. <https://doi.org/10.1111/j.1558-5646.1995.tb05967.x>

- **Walsh, B. & Lynch, M.** (2018). *Evolution and Selection of Quantitative Traits*. Oxford University Press. <https://doi.org/10.1093/oso/9780198830870.001.0001>
  *Comprehensive modern reference on stochastic quantitative genetics.*

- **Bürger, R.** (2000). *The Mathematical Theory of Selection, Recombination, and Mutation*. Wiley.

---

## 10. Foundation Models, Pretraining, Transfer Learning

- **Bommasani, R. et al.** (2021). On the opportunities and risks of foundation models. arXiv:2108.07258. <https://arxiv.org/abs/2108.07258>

- **Hu, E.J. et al.** (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685. <https://arxiv.org/abs/2106.09685>

- **Dosovitskiy, A. et al.** (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*. <https://arxiv.org/abs/2010.11929>
  *Original Vision Transformer paper — relevant for PhenoVision empirical work.*

---

## 11. Langevin Dynamics, Fluctuation-Dissipation, and Diffusion Generative Models

The physical-statistical-mechanics underpinning that unifies Lande's stochastic breeder's equation, SGLD, score-based generative models, and natural-gradient optimization. Directly relevant for Part E of the derivation document.

### Foundational physics

- **Einstein, A.** (1905). Über die von der molekularkinetischen Theorie der Wärme geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten Teilchen [On the movement of small particles suspended in stationary liquids required by the molecular-kinetic theory of heat]. *Annalen der Physik* 17: 549–560. <https://doi.org/10.1002/andp.19053220806>
  *The original Brownian motion paper. First derivation of the fluctuation-dissipation relation D = k_B T / γ.*

- **Kubo, R.** (1966). The fluctuation-dissipation theorem. *Reports on Progress in Physics* 29(1): 255–284. <https://doi.org/10.1088/0034-4885/29/1/306>
  *Classic review formalizing the general fluctuation-dissipation theorem. Standard reference for the linkage between dissipation matrices and equilibrium noise structure.*

- **Risken, H.** (1996). *The Fokker-Planck Equation: Methods of Solution and Applications* (2nd ed.). Springer. <https://doi.org/10.1007/978-3-642-61544-3>
  *Standard reference textbook for Fokker-Planck and Langevin dynamics. Contains the OU process, Lyapunov equations, and stationary distribution machinery used in Part D of the derivation.*

- **Gardiner, C.** (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences* (4th ed.). Springer.
  *Another canonical SDE/Fokker-Planck reference. The MHB paper cites this for SDE methodology.*

### Score-based generative modeling

- **Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S.** (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*. <https://arxiv.org/abs/1503.03585>
  *Original diffusion model paper. Explicitly inspired by nonequilibrium statistical mechanics — the connection to physics has been there from the start of this line of work.*

- **Song, Y. & Ermon, S.** (2019). Generative modeling by estimating gradients of the data distribution. *Advances in Neural Information Processing Systems (NeurIPS) 32*. <https://arxiv.org/abs/1907.05600>
  *Score-based generative modeling. Introduces the score-matching + Langevin-sampling architecture.*

- **Ho, J., Jain, A. & Abbeel, P.** (2020). Denoising diffusion probabilistic models (DDPM). *Advances in Neural Information Processing Systems (NeurIPS) 33*. <https://arxiv.org/abs/2006.11239>
  *DDPM — the formulation that triggered the diffusion-model wave in image generation. Stable Diffusion, DALL-E 2, and most current systems descend from this.*

- **Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. & Poole, B.** (2021). Score-based generative modeling through stochastic differential equations. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/2011.13456>
  *Unifies score-based and DDPM approaches under a common SDE framework. The cleanest reference for "diffusion models are Langevin sampling from a learned score function." Makes the connection to Part E of the derivation document most explicit.*

- **Anderson, B.D.O.** (1982). Reverse-time diffusion equation models. *Stochastic Processes and their Applications* 12(3): 313–326. <https://doi.org/10.1016/0304-4149(82)90051-5>
  *Original derivation of the time-reversal of an SDE. The technical foundation that makes denoising-as-sampling work.*

### Statistical physics connections to evolutionary biology

(Cross-references — these are also listed in Section 7, included here for the FDT thematic grouping.)

- **Iwasa, Y.** (1988). Free fitness that always increases in evolution. *Journal of Theoretical Biology* 135(3): 265–281. <https://doi.org/10.1016/S0022-5193(88)80243-1>

- **Sella, G. & Hirsh, A.E.** (2005). The application of statistical physics to evolutionary biology. *PNAS* 102(27): 9541–9546. <https://doi.org/10.1073/pnas.0501865102>

---

*Last updated: November 2026. Working bibliography for the SGD–quantitative genetics formal bridge and the broader "deep past in the present" framework.*
