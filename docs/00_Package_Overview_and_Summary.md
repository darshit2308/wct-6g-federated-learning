# Package Overview & Experimental Deliverables Summary

**Manuscript ID:** NL2026-0420  
**Title:** *Adaptive Hierarchical Federated Learning for Fair and Robust 6G Edge Intelligence*  
**Authors:** Jay Hirapara, Darshit Khandelwal, Mahendra K. Shukla (Senior Member, IEEE), Manish Pratap Singh, and Om Jee Pandey (Senior Member, IEEE)  
**Target Journal:** *IEEE Networking Letters*  

---

## 1. Overview of Experimental Work Completed

This package contains all artifacts, experimental data, theoretical derivations, and publication-ready tables and figures prepared to address the review feedback for manuscript NL2026-0420. All benchmarks have been rigorously executed across **5 independent random seeds (42, 52, 62, 72, 82)**.

### Summary of Additions & Upgrades:
1. **Classical Robust Baselines Evaluated:**
   - **Krum (Blanchard et al., NeurIPS 2017):** Euclidean distance-based Byzantine-resilient aggregation.
   - **Coordinate-wise Trimmed Mean (Yin et al., ICML 2018):** Statistical aggregation trimming extreme gradient dimensions.
2. **Fairness Selector Validation (Direct-Score Baseline):**
   - Implemented `hierarchical_direct_score` that directly selects devices via linear desirability scores without the learned ML selector.
   - Demonstrates that omitting the learned fairness regularization causes device participation fairness to collapse from **0.8806 down to 0.5015**, providing clear empirical evidence for the necessity of the proposed selector.
3. **Competitive Adversarial Attack Models:**
   - **Adaptive Fang Attack (Fang et al., USENIX Security 2020):** Manipulates update direction while matching benign gradient norms to test evasion of threshold defenses.
   - **Label-Flipping & Sign-Flipping Attacks** evaluated across varying attacker proportions.
4. **Mathematical Derivation for Equations (10)–(11):**
   - Formally derived the connection between the multi-objective selection formulation and the greedy edge selection rule as a first-order **scalarized Lyapunov drift-plus-penalty relaxation**.
5. **Sensitivity Analysis & Pareto Frontier:**
   - Evaluated threshold sensitivity across $\tau \in [0.45, 0.72]$ and generated the Pareto-optimal frontier for Accuracy vs. Jain Fairness vs. Communication overhead.

---

## 2. Inventory and Description of Package Files

| File Name | Format | Content & Purpose |
| :--- | :--- | :--- |
| **`00_Package_Overview_and_Summary.md`** | Markdown | This document; high-level summary of all package contents and experimental results. |
| **`IEEE_Networking_Letters_Revision_Dossier.md`** | Markdown | Comprehensive point-by-point rebuttal draft addressing the Associate Editor and Reviewers 1, 2, 3, and 4. |
| **`mathematical_formulations_and_reviewer_rebuttal.md`** | Markdown | Detailed mathematical derivations, Lyapunov drift-plus-penalty proof for Equations (10)–(11), and Byzantine threat models. |
| **`research_paper.tex`** | LaTeX (`IEEEtran`) | Full manuscript source code updated with exact author affiliations, revised narrative, and updated 5-seed benchmark tables. |
| **`table_clean_publication.tex`** | LaTeX (`table*`) | Standalone publication table comparing all 10 methods on clean data across 5 seeds ($\text{Mean} \pm \text{Std Dev}$). |
| **`table_attack_publication.tex`** | LaTeX (`table*`) | Standalone publication table evaluating robustness under Adaptive Fang, Label-flip, and Sign-flip attacks. |
| **`pareto_frontier.png`** | High-Res PNG | Pareto-optimal frontier showing Accuracy vs. Jain Fairness Index across hierarchical configurations. |
| **`threshold_sensitivity.png`** | High-Res PNG | Sensitivity curve demonstrating performance stability across selection thresholds $\tau \in [0.45, 0.72]$. |
| **`fig_convergence_rounds.png`** | High-Res PNG | Accuracy convergence curves across communication rounds comparing all key baselines. |
| **`fig_fairness_comparison.png`** | High-Res PNG | Bar chart comparing Jain's fairness index across all 10 methods, highlighting the Direct-Score collapse ($0.5015$) vs. Proposed ($0.8806$). |
| **`fig_attack_detection.png`** | High-Res PNG | Two-panel publication figure: (a) Byzantine detection rates (%) and (b) final model accuracy under Adaptive Fang, Label-flip, and Sign-flip attacks. |
| **`fig_cloud_uploads.png`** | High-Res PNG | Cloud communication load showing a 75% reduction in cloud synchronization uploads (20 vs. 80 uploads). |

---

## 3. Summary of Key Experimental Results

### Clean Non-IID Benchmark (5 Seeds: 42, 52, 62, 72, 82)
- **Proposed Method:** Accuracy $= \mathbf{0.8983 \pm 0.0280}$, Macro-F1 $= \mathbf{0.8965 \pm 0.0307}$, Jain Fairness $= \mathbf{0.8806 \pm 0.0217}$, Latency $= 120.0$ ms.
- **Direct-Score Selection (No ML):** Accuracy $= 0.9050 \pm 0.0127$, Jain Fairness $= \mathbf{0.5015 \pm 0.0365}$ (*Fairness collapses by 43%*).
- **Krum Baseline (NeurIPS 2017):** Accuracy $= 0.8200 \pm 0.0364$, Jain Fairness $= 0.8544 \pm 0.0222$.
- **Trimmed Mean Baseline (ICML 2018):** Accuracy $= 0.8872 \pm 0.0440$, Jain Fairness $= 0.8544 \pm 0.0222$.
- **Flat FedAvg:** Accuracy $= 0.9061 \pm 0.0205$, Latency $= 199.4$ ms (*66% higher latency, 400% higher cloud communication fan-in*).

### Adversarial Security Benchmark (25% Attackers, 5 Seeds)
- **Adaptive Fang Attack (USENIX Security 2020):**
  - Proposed Method: Accuracy $= \mathbf{0.8933 \pm 0.0314}$, Detection Rate $= \mathbf{78.00\% \pm 5.10\%}$.
  - Krum: Accuracy $= 0.8061 \pm 0.0455$.
  - Trimmed Mean: Accuracy $= 0.8706 \pm 0.0400$.
  - Unshielded FedAvg: Accuracy $= 0.8622 \pm 0.0512$.
- **Label-Flipping Attack:**
  - Proposed Method: Accuracy $= \mathbf{0.8861 \pm 0.0191}$, Detection Rate $= \mathbf{74.00\% \pm 11.58\%}$.
  - Krum: Accuracy $= 0.8061 \pm 0.0455$.
  - Trimmed Mean: Accuracy $= 0.8444 \pm 0.0600$.
- **Sign-Flipping Attack:**
  - Proposed Method: Accuracy $= \mathbf{0.8950 \pm 0.0211}$, Detection Rate $= \mathbf{100.0\% \pm 0.0\%}$.
  - Flat FedAvg / Unshielded methods collapse to **$0.0978\text{--}0.1411$** (random guess).

---

## 4. Cross-Reference Mapping to Reviewer Comments

- **Reviewer 1 (Heuristics & Theory):**
  - Mathematical justification of selection formulation $\to$ `mathematical_formulations_and_reviewer_rebuttal.md` (Section 1).
  - Empirical validation of learned fairness vs. direct score $\to$ `table_clean_publication.tex` (`hierarchical_direct_score` vs `proposed`).
- **Reviewer 2 (Threat Model & Attacks):**
  - Formal Byzantine threat model and Fang attack derivation $\to$ `mathematical_formulations_and_reviewer_rebuttal.md` (Section 2).
  - Empirical results under stealthy attacks $\to$ `table_attack_publication.tex`.
- **Reviewer 3 (Baselines & Sensitivity):**
  - Traditional robust baselines (Krum, Trimmed Mean) $\to$ `table_clean_publication.tex` and `table_attack_publication.tex`.
  - Sensitivity analysis over threshold $\tau \in [0.45, 0.72]$ $\to$ `threshold_sensitivity.png`.
  - Pareto trade-off analysis $\to$ `pareto_frontier.png`.
- **Reviewer 4 (Statistical Rigor):**
  - 5-seed evaluation with $\text{Mean} \pm \text{Std Dev}$ across all metrics $\to$ `table_clean_publication.tex` and `table_attack_publication.tex`.
