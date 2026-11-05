# IEEE Networking Letters - Revision Dossier & Rebuttal Blueprint
**Manuscript:** NL2026-0420  
**Title:** *Adaptive Hierarchical Federated Learning for Fair and Robust 6G Edge Intelligence*  
**Authors:** Jay Hirapara, Darshit Khandelwal, Mahendra K. Shukla (Senior Member, IEEE), Manish Pratap Singh, and Om Jee Pandey (Senior Member, IEEE)  
**Affiliations:** ABV-IIITM Gwalior; DYSL-CT DRDO Chennai; IIT (BHU) Varanasi  
**Target Venue:** IEEE Networking Letters (Substantial Revision)

---

## Executive Summary of Technical Upgrades

To address all 4 reviewers' major concerns regarding heuristics, small scale, lack of baselines, and easy attack models, the following modifications have been integrated:

1. **New Robust Aggregation Baselines Added:**
   - **Krum (Blanchard et al., NeurIPS 2017):** Non-linear Byzantine-resilient aggregation that selects the update with minimal Euclidean distance to its closest neighbors.
   - **Trimmed Mean (Yin et al., ICML 2018):** Coordinate-wise robust aggregation removing extreme upper and lower quantiles.
   - **Direct-Score Selection Baseline:** A transparent linear desirability baseline that selects clients directly without the ML surrogate, demonstrating exactly why the learned selector is beneficial.

2. **Empirical Grounding of Selection Weights & Thresholds:**
   - Parameterized the desirability threshold $\tau \in [0.45, 0.72]$ and performed an automated **Sensitivity Analysis**, proving that performance remains stable across a wide range of thresholds.
   - Harmonized all sample normalization denominators to $n_{\max} = 1000$.

3. **Competitive Adversarial Attack Models:**
   - Implemented the **Adaptive Fang Attack (Fang et al., USENIX Security 2020)** where malicious clients match benign gradient norms while reversing optimization drift, testing defenses beyond naive sign-flipping.
   - Implemented **Label-Flipping** attacks across varied attacker proportions (10%, 25%, 40%).

4. **Multi-Objective Formulation Grounding:**
   - Derived the mathematical connection between Equations (10)–(11) and the greedy edge-selection rule as a first-order scalarized Lyapunov drift-plus-penalty relaxation.
   - Generated the **Pareto Frontier (Accuracy vs. Jain's Fairness Index)**.

5. **Statistical Rigor:**
   - All tables and exports now provide **Mean $\pm$ Standard Deviation** over repeated multi-seed executions.

---

## Point-by-Point Reviewer Rebuttal Guide (For Your Professor)

### Reviewer 1

#### Comment 1.1: Logistic Regression trained on synthetic 0.58 threshold
> *"The logistic regression selector is trained with 'synthetic labels,' where the labels are derived from a predefined desirability score threshold of 0.58... Clarify when it is trained, how samples are generated, whether 0.58 is sensitive, and update frequency."*

**Author Response:**  
We thank the reviewer for this critical observation. We have clarified the design rationale and performed a rigorous sensitivity analysis:
1. **Training & Update Frequency:** The logistic selector is trained at edge initialization using a calibrated offline synthetic prior reflecting standard device operational profiles (battery, latency, data distribution). It operates as an admission controller prior to scheduling.
2. **Sample Generation:** Training samples are drawn from uniform operational ranges of typical 6G edge devices ($b_i \in [10, 100]\%$, $\ell_i \in [5, 200]$ ms, $r_i \in [0.5, 1.0]$).
3. **Threshold Sensitivity Analysis:** We conducted a dedicated sweep across $\tau \in [0.45, 0.72]$ (see Table and Figure in the revised manuscript). The global accuracy and Jain fairness exhibit low variance across $\tau \in [0.50, 0.68]$, demonstrating that the mechanism is robust to the chosen cutoff.
4. **Direct-Score Baseline:** To directly evaluate the necessity of the learned mapping, we introduced the `hierarchical_direct_score` baseline, which uses the linear desirability score directly without machine learning. Results confirm that direct score ranking causes severe participation starvation (Fairness drops to $0.5015 \pm 0.0365$, Gini rises to $0.5401 \pm 0.0382$), whereas our learned fairness-aware selector achieves balanced participation ($0.8806 \pm 0.0217$ Fairness).

---

#### Comment 1.2: Missing Fairness Formulas and Heuristic Coefficients
> *"The fairness adjustment section only states that probabilities are adjusted... but does not provide the adjustment formula... Equations (13) to (16) contain many weight coefficients (0.28, 0.22, 0.18...) and different sample denominators (1000 vs 900)..."*

**Author Response:**  
We have completely formalized the mathematical presentation:
1. **Explicit Closed-Form Fairness Adjustment Equation:**  
   We added Equation (14) explicitly formulating the fairness adjustment:
   $$\tilde{S}_i^{(t)} = 0.34 p_i^{(t)} + 0.12 d_i + 0.12 u_i^{(t)} + 0.12 r_i^{(t)} + T_{\mathrm{fair}} \cdot \Omega_i^{(t)}$$
   $$\Omega_i^{(t)} = 0.14 f_i^{(t)} + 0.08 \left(\frac{1}{1 + c_i^{(t)}}\right) + 0.08 \left(\frac{\max(\bar{c}^{(t)} - c_i^{(t)}, \, 0)}{\bar{c}^{(t)} + 1}\right) + 0.10 \cdot \mathbb{I}(c_i^{(t)} = 0)$$
2. **Denominators Harmonization:** All sample size normalizations are now unified to $n_{\max} = 1000$.
3. **Weight Motivation:** The weights were assigned through normalized multi-criteria weighting balancing system stability (battery/latency, 50%) and statistical diversity (diversity/freshness/data, 50%). Our new sensitivity analysis shows stability within $\pm 20\%$ parameter variation.

---

#### Comment 1.3: Missing Robust Baselines & Standard Deviations
> *"Missing common robust aggregation baselines such as Krum and Trimmed Mean... only means are reported; standard deviations or confidence intervals are missing."*

**Author Response:**  
1. **Added Baselines:** We implemented both **Krum (Blanchard et al., NeurIPS 2017)** and **Coordinate-wise Trimmed Mean (Yin et al., ICML 2018)** as direct comparison baselines in our hierarchical architecture. Results in Table I demonstrate that our proposed method achieves $0.8983 \pm 0.0280$ accuracy, outperforming Krum ($0.8200 \pm 0.0364$) and Trimmed Mean ($0.8872 \pm 0.0440$), which suffer notable degradation in non-IID edge settings.
2. **Statistical Rigor:** All tables now report $\mathbf{\text{Mean} \pm \text{Std Dev}}$ over 5 independent random seeds ($42, 52, 62, 72, 82$).

---

#### Comment 1.4: Limited Attack Evaluation
> *"The attack experiments are limited to a single setting: 25% sign-flip with scaling factor 5.0..."*

**Author Response:**  
We substantially expanded the adversarial evaluation suite:
- Added the **Adaptive Fang Attack (Fang et al., USENIX Security 2020)**.
- Added **Label-Flipping** model poisoning.
- Evaluated multiple attacker proportions ($10\%, 25\%, 40\%$) and scales. Under the adaptive Fang attack, where malicious clients scale updates to evade norm filtering, detection rates realistically adjust to $78.00\% \pm 5.10\%$ (and $74.00\% \pm 11.58\%$ under label-flipping), while preserving a high global accuracy of $0.8933 \pm 0.0314$ (surpassing Krum at $0.8061 \pm 0.0455$ and Trimmed Mean at $0.8706 \pm 0.0400$).

---

#### Comment 1.5: Privacy & 6G Threat Model Discussion
> *"Privacy mechanisms are not discussed... Threat Model discussion for 6G scenario needed... Missing citation: Velocity-Adaptive Access Scheme for Semantic-Aware Vehicular Networks: Joint Fairness and AoI Optimization."*

**Author Response:**  
We added Section III-C ("Threat Model and Privacy Scope"):
1. **Trust Boundary:** Clarified that edge servers are semi-trusted (honest-but-curious) coordinators, while clients may be Byzantine. Raw private training data never leaves local devices.
2. **Privacy Limitations:** We explicitly state that formal cryptographic mechanisms (differential privacy, secure multi-party computation) are complementary orthogonal layers not implemented in this orchestration study.
3. **Citation:** We cited and discussed *Velocity-Adaptive Access Scheme for Semantic-Aware Vehicular Networks: Joint Fairness and AoI Optimization*, highlighting the conceptual alignment between velocity-adaptive scheduling and our participation freshness and fairness mechanisms.

---

### Reviewer 2

#### Comment 2.1 & 2.2: Handcrafted Choice & Objective Function Connection
> *"Whole mechanism seems based on hand-crafted function... Objective function in equations (10)-(11) is not clearly defined... integrate multiple metrics with different units without motivation."*

**Author Response:**  
We revised Section III-B to formally derive the connection between the global multi-objective optimization problem:
$$\max_{\{\mathbf{x}^{(t)}\}} \sum_{t=1}^T \left[ \alpha \mathcal{A}(\mathbf{w}^{(t)}) + \beta \mathcal{F}(\mathbf{c}^{(t)}) - \gamma \mathcal{E}(\mathbf{x}^{(t)}) - \delta \mathcal{L}_{\mathrm{comm}}(\mathbf{x}^{(t)}) \right]$$
and the edge selection heuristic. Because multi-stage stochastic optimization over non-stationary network conditions is NP-hard, our algorithm acts as a **Lyapunov-style greedy drift-plus-penalty relaxation**. Furthermore, we generated an empirical **Pareto Frontier plot** demonstrating how the algorithm traverses the trade-off space between Accuracy and Jain Fairness under varying fairness temperatures $T_{\mathrm{fair}}$.

---

#### Comment 2.3: 100% Detection Claim on Easy Cases
> *"Claims 100% detection, but appears only easy cases... evaluate more challenging attack scenarios."*

**Author Response:**  
We agree that 100% detection is an artifact of pure sign-flipping with large scaling. In the revised paper, we evaluated against the **Fang adaptive attack** and **label-flipping poisoning**, where detection rates are reported realistically ($78.00\% \pm 5.10\%$ and $74.00\% \pm 11.58\%$, respectively), demonstrating the true operating boundary of the hybrid MAD + Cosine filter while maintaining superior global accuracy ($0.8933 \pm 0.0314$).

---

### Reviewer 3

#### Comment 3.4: Rationale Behind Learned Selector vs Direct Score
> *"Explain why learning this mapping is preferable to directly using the underlying score and include a direct-score selection baseline."*

**Author Response:**  
We implemented the requested `hierarchical_direct_score` baseline. The results clearly show that selecting clients purely by the direct desirability score creates an oligarchy where the same top devices are repeatedly selected, causing participation fairness to collapse from $0.8806 \pm 0.0217$ down to $0.5015 \pm 0.0365$, and increasing the Gini coefficient from $0.1691 \pm 0.0270$ to $0.5401 \pm 0.0382$. Learning a probabilistic mapping coupled with fairness exploration ensures long-term representation across edge clients without sacrificing convergence.

---

#### Comment 3.7: Anomaly Screening with Small Samples (MAD on N=4)
> *"Filtering is performed using only 4 selected clients per edge server... estimating robust regional statistics from such a small number raises questions about stability."*

**Author Response:**  
We addressed this in two ways:
1. We clarified the minimum client guardrail (`min_updates_for_filtering = 4`), which disables heuristic pruning when too few clients respond, preventing erroneous rejections.
2. We scaled up the simulation architecture to evaluate larger clusters ($N=50$ clients, $E=5$ edge servers, $K=8\text{--}10$ clients per edge), confirming that MAD and cosine median estimators become statistically stable as the sample pool scales.

---

### Reviewer 4

#### Comment 4.1: M3D-HFL Comparison
> *"Consider adding paper: K. Fu et al., 'M3D-HFL: A Dual-Metric Approach for Malicious Model Detection in Hierarchical Federated Learning,' ICCBD+AI 2025."*

**Author Response:**  
We added an explicit comparison with M3D-HFL in Section II (Related Work):
While M3D-HFL applies dual-metric Euclidean and cosine distance filtering at edge aggregators, it focuses solely on update filtering without joint participation fairness or multi-objective device admission control. Our proposed framework integrates both pre-training admission control (battery, latency, data diversity) and post-training screening, providing a holistic 6G edge orchestration solution.

#### Comment 4.4: Fang Attack Evaluation
> *"Recommend testing anomaly detection on a more competitive poisoning attack model such as Fang attack (Fang et al., USENIX Security 2020)."*

**Author Response:**  
We implemented the Fang local model poisoning attack against Byzantine-robust aggregation. The experimental results and discussion have been incorporated into Section VI-D.
