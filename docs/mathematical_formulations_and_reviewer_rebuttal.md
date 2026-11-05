# Mathematical Formulations and System Mechanics Guide
**Manuscript Revision Companion for:** *Adaptive Hierarchical Federated Learning for Fair and Robust 6G Edge Intelligence* (NL2026-0420)

---

## 1. Explicit Mathematical Formulations (Answers Rev 1.2, Rev 3.6, Rev 4.2)

### 1.1 Device State Features & Evolution
Each client $i \in \mathcal{C}$ at communication round $t$ is characterized by the state vector:
$$\mathbf{s}_i^{(t)} = \left[ b_i^{(t)}, \ell_i^{(t)}, n_i, d_i, r_i^{(t)}, f_i^{(t)} \right]^\top$$

1. **Battery Level ($b_i^{(t)} \in [15, 100]$%):**
   $$b_i^{(t+1)} = \max\left(15, \, b_i^{(t)} - \delta_i^{(t)}\right), \quad \delta_i^{(t)} \sim \mathcal{U}\{1, 4\}$$

2. **Network Latency ($\ell_i^{(t)} \in [5, 200]$ ms):**
   $$\ell_i^{(t+1)} = \mathrm{clip}\left(\ell_i^{(t)} + \xi_i^{(t)}, \, 5, \, 200\right), \quad \xi_i^{(t)} \sim \mathcal{U}\{-8, 8\}$$

3. **Data Diversity Score ($d_i \in [0, 1]$):**
   Computed once from local dataset class distribution and feature dispersion:
   $$d_i = 0.70 \cdot \frac{H(\mathbf{p}_i)}{\ln(C)} + 0.30 \cdot \tanh\left(\mathrm{Var}(\mathbf{X}_i)\right)$$
   where $H(\mathbf{p}_i) = -\sum_{c=1}^C p_{ic} \ln(p_{ic})$ is the Shannon entropy over class empirical frequencies $p_{ic} = \frac{n_{ic}}{n_i}$, $C$ is the total number of classes, and $\mathrm{Var}(\mathbf{X}_i)$ is the local feature variance.

4. **Participation Freshness Score ($f_i^{(t)} \in [0, 1]$):**
   Tracks rounds elapsed since last participation $\Delta t_i$:
   $$f_i^{(t)} = \min\left( \frac{\Delta t_i}{4}, \, 1.0 \right)$$
   where $\Delta t_i$ increments by 1 if unselected, and resets to 0 upon selection.

5. **Reliability Score Update ($r_i^{(t)} \in [0.15, 1.0]$):**
   Evolves dynamically based on validation acceptance and historical quality:
   $$r_i^{(t)} = \begin{cases}
   \mathrm{clip}\left( 0.80 r_i^{(t-1)} + 0.20 (0.55 + 0.45 a_i^{(t)}), \, 0.20, \, 1.0 \right), & \text{if update accepted} \\
   \mathrm{clip}\left( 0.82 r_i^{(t-1)}, \, 0.15, \, 1.0 \right), & \text{if update filtered/rejected}
   \end{cases}$$
   where $a_i^{(t)}$ is the client's local evaluation accuracy.

---

### 1.2 System & Learning Quality Scores (Harmonized Denominators)
Harmonizing all normalization constants to $n_{\max} = 1000$:

1. **System Condition Score ($u_i^{(t)} \in [0, 1]$):**
   $$u_i^{(t)} = 0.35 \left(\frac{b_i^{(t)}}{100}\right) + 0.20 \left(1 - \frac{\ell_i^{(t)}}{200}\right) + 0.30 \min\left(\frac{n_i}{n_{\max}}, 1.0\right) + 0.15 r_i^{(t)}$$

2. **Immediate Learning-Quality Score ($\lambda_i^{(t)} \in [0, 1]$):**
   $$\lambda_i^{(t)} = 0.45 a_i^{(t)} + 0.25 e^{-L_i^{(t)}} + 0.20 d_i + 0.10 h_i^{(t)}$$
   where $L_i^{(t)}$ is local cross-entropy loss and $h_i^{(t)}$ is historical utility.

3. **Overall Quality Score ($q_i^{(t)}$):**
   $$q_i^{(t)} = 0.30 u_i^{(t)} + 0.45 \lambda_i^{(t)} + 0.15 d_i + 0.10 r_i^{(t)}$$

---

### 1.3 Fairness-Adjusted Client Selection Probability (Answers Rev 1.2, Rev 3.6)
Let $p_i^{(t)} = \sigma(\boldsymbol{\theta}^\top \tilde{\mathbf{s}}_i^{(t)})$ be the base probability from the logistic classifier.
The fairness-adjusted selection score $\tilde{S}_i^{(t)}$ is explicitly formulated as:
$$\tilde{S}_i^{(t)} = 0.34 p_i^{(t)} + 0.12 d_i + 0.12 u_i^{(t)} + 0.12 r_i^{(t)} + T_{\mathrm{fair}} \cdot \Omega_i^{(t)}$$
where $T_{\mathrm{fair}} \ge 0$ is the fairness temperature, and the fairness regularization term $\Omega_i^{(t)}$ is:
$$\Omega_i^{(t)} = 0.14 f_i^{(t)} + 0.08 \left(\frac{1}{1 + c_i^{(t)}}\right) + 0.08 \left(\frac{\max(\bar{c}^{(t)} - c_i^{(t)}, \, 0)}{\bar{c}^{(t)} + 1}\right) + 0.10 \cdot \mathbb{I}\left(c_i^{(t)} = 0\right)$$
Here:
- $c_i^{(t)}$ is the cumulative selection count of client $i$ up to round $t$,
- $\bar{c}^{(t)} = \frac{1}{|\mathcal{C}_e|}\sum_{j \in \mathcal{C}_e} c_j^{(t)}$ is the regional mean selection count,
- $\mathbb{I}(\cdot)$ is the indicator function granting a first-time coverage bonus.

Clients are sampled without replacement from categorical distribution $\pi_i^{(t)} = \frac{\tilde{S}_i^{(t)}}{\sum_{j \in \mathcal{C}_e} \tilde{S}_j^{(t)}}$.

---

### 1.4 Hybrid Anomaly Decision Rule (Answers Rev 3.7)
Let $\Delta_i^{(t)} = \mathbf{w}_i^{(t)} - \mathbf{w}^{(t-1)}$ be the flattened local update vector.
The edge reference consensus is the coordinate-wise median:
$$\Delta_{\mathrm{ref}}^{(t)} = \mathrm{median}_{j \in \mathcal{S}_e^{(t)}} \left(\Delta_j^{(t)}\right)$$
For each client $i \in \mathcal{S}_e^{(t)}$, compute:
1. Norm: $\|\Delta_i^{(t)}\|_2$
2. Directional Cosine: $\rho_i^{(t)} = \frac{\Delta_i^{(t)} \cdot \Delta_{\mathrm{ref}}^{(t)}}{\max\left(\|\Delta_i^{(t)}\|_2 \|\Delta_{\mathrm{ref}}^{(t)}\|_2, \, 10^{-12}\right)}$

Decision Rule: Client $i$ update is rejected if:
$$\left( \frac{\left| \|\Delta_i^{(t)}\|_2 - \mathrm{median}(\mathcal{N}) \right|}{1.4826 \cdot \mathrm{MAD}(\mathcal{N})} > \tau_{\mathrm{mad}} \right) \quad \bigvee \quad \left( \rho_i^{(t)} < \rho_{\mathrm{floor}} \right) \quad \bigvee \quad \left( \frac{\left| \rho_i^{(t)} - \mathrm{median}(\boldsymbol{\rho}) \right|}{1.4826 \cdot \mathrm{MAD}(\boldsymbol{\rho})} > \tau_{\cos} \right)$$
where $\tau_{\mathrm{mad}} = 3.5$, $\tau_{\cos} = 3.0$, and $\rho_{\mathrm{floor}} = 0.0$.

---

## 2. Theoretical Connection: Multi-Objective Optimization to Greedy Selection (Answers Rev 2.2, Rev 3.3)

In Equations (10)–(11), the global system objective is formulated as:
$$\max_{\{\mathbf{x}^{(t)}\}} \sum_{t=1}^T \left[ \alpha \mathcal{A}(\mathbf{w}^{(t)}) + \beta \mathcal{F}(\mathbf{c}^{(t)}) - \gamma \mathcal{E}(\mathbf{x}^{(t)}) - \delta \mathcal{L}_{\mathrm{comm}}(\mathbf{x}^{(t)}) \right]$$
subject to per-round client budget $\sum_{i \in \mathcal{C}_e} x_i^{(t)} \le K_e$.

**Greedy Decomposition:**
Because the global loss is non-convex and device dynamics are non-stationary, exact dynamic programming over $T$ rounds is intractable. Our edge-native heuristic is a **first-order Lyapunov/scalarized greedy relaxation**:
1. Accuracy $\mathcal{A}$ is approximated by client learning potential $\lambda_i^{(t)}$ and diversity $d_i$.
2. Fairness $\mathcal{F}$ is enforced through the virtual queue penalty $\Omega_i^{(t)}$ (drift-plus-penalty framework on participation disparity).
3. Resource overhead ($\mathcal{E}, \mathcal{L}_{\mathrm{comm}}$) is captured by system state $u_i^{(t)}$ (battery and network latency).

Thus, the ranking score $\tilde{S}_i^{(t)}$ directly mirrors the scalarized marginal gain of client $i$ at round $t$.

---

## 3. Threat Model & Trust Boundaries (Answers Rev 1.5, Rev 3.9)

| Metric | Source | Verification Mechanism at Edge | Can Attacker Spoof? |
| :--- | :--- | :--- | :--- |
| Model Delta $\Delta_i^{(t)}$ | Transmitted weights | Verified via Norm MAD + Directional Cosine against batch median. | **No.** Outlier deltas are rejected. |
| Reported Loss $L_i$ / Acc $a_i$ | Client metadata | Edge evaluates update on a small regional validation proxy if needed. | Attacker can lie, but delta geometry exposes poisoning. |
| Battery $b_i$ / Latency $\ell_i$ | Client telemetry | Latency measured directly by edge RTT (Round Trip Time). | Battery can be spoofed; Latency is network-enforced. |
| Reliability $r_i$ / Counts $c_i$ | Server-side state | Maintained **strictly on the edge server**. | **Impossible.** Client has no write access. |

**Trust Boundary:**
Edge servers are honest-but-curious or semi-trusted aggregators. Device clients may be malicious (Byzantine). Private raw data $\mathcal{D}_i$ never leaves the client, preventing direct data leakage.

---

## 4. Literature Comparison & Citations (Answers Rev 1.5, Rev 3.11, Rev 4.1)

1. **M3D-HFL (Fu et al., ICCBD+AI 2025):**
   * *Difference:* M3D-HFL uses a dual-metric detection relying strictly on cosine and euclidean distances on the edge without joint participation fairness or multi-objective device resource admission. Our work integrates both pre-training admission control and post-training filtering.
2. **Fang Attack (Fang et al., USENIX Security 2020):**
   * *Significance:* Demonstrates that coordinate-wise trim/median defenses fail against crafted perturbations that manipulate the variance while keeping mean shifts inside the boundary. We incorporate this adaptive attack to demonstrate realistic resilience.
3. **Joint Fairness & AoI in Networks (IEEE TVT / JSAC):**
   * References velocity-adaptive access and Age-of-Information (AoI), directly analogous to our selection freshness $f_i^{(t)}$ and fairness temperature $T_{\mathrm{fair}}$.
