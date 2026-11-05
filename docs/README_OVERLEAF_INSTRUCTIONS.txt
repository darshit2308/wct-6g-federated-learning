Overleaf Update Package for Dr. Mahendra Shukla & Author Team
Manuscript: NL2026-0420 (IEEE Networking Letters)
Title: Adaptive Hierarchical Federated Learning for Fair and Robust 6G Edge Intelligence

Dear Sir,

As requested, here are the original Figures (Fig. 1, Fig. 2, and Fig. 3) from our earlier submitted paper (Jay_WCT_2026_05_9_26 (7)), along with the updated 5-seed figures and the updated LaTeX source code:

1. INCLUDED FIGURE FILES:
   - fig1_architecture.png: Fig. 1 Architecture Diagram (3-Tier Device-Edge-Cloud)
   - fig2_energy.png: Fig. 2 Cumulative Normalized Energy Consumption (FedAvg vs. Proposed)
   - fig3_convergence.png: Fig. 3 Model Convergence over Rounds (from original manuscript)
   - fig3_convergence_updated_5seed.png: Updated Fig. 3 generated from our new 5-seed benchmark comparing Proposed, Direct-Score, Krum, and Trimmed Mean
   - pareto_frontier.png: Pareto Optimal Frontier (Accuracy vs. Jain Fairness)
   - threshold_sensitivity.png: Admission Threshold Sensitivity Analysis (tau in [0.45, 0.72])
   - fig_attack_detection.png: Byzantine Robustness under Adaptive Fang, Label-flip, and Sign-flip

2. HOW TO UPDATE OVERLEAF:
   Option A (Fastest - Drag & Drop):
     - Upload 'fig1_architecture.png', 'fig2_energy.png', and 'fig3_convergence.png' directly into your Overleaf project file tree on the left.
     - Replace your Overleaf 'main.tex' with the attached 'research_paper.tex'.
     - Click "Recompile". Everything will compile cleanly with all figures, authors, and tables in place.

   Option B (Copy LaTeX Snippets):
     - For Fig. 1 (Section III):
         \begin{figure}[!t]
         \centering
         \includegraphics[width=\columnwidth]{fig1_architecture.png}
         \caption{Adaptive hierarchical FL architecture with fairness-aware selection, anomaly screening, quality-aware aggregation, and cloud synchronization.}
         \label{fig:architecture}
         \end{figure}

     - For Fig. 2 & Fig. 3 (Section V-A):
         \begin{figure}[!t]
         \centering
         \includegraphics[width=\columnwidth]{fig2_energy.png}
         \caption{Cumulative normalized energy consumption of traditional FL and the proposed adaptive FL framework over communication rounds.}
         \label{fig:energy}
         \end{figure}

         \begin{figure}[!t]
         \centering
         \includegraphics[width=\columnwidth]{fig3_convergence.png}
         \caption{Global model convergence of traditional FL (FedAvg) and the proposed adaptive FL framework over communication rounds.}
         \label{fig:convergence}
         \end{figure}
