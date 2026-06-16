#!/usr/bin/env python3
"""Generate Figure 4 (baseline comparison) from the locked CSVs.

Data sources (deterministic outputs of 08-multidimensional-eval.py):
  RESULTS/EVALUATION/statistical_tests.csv   (WCS, baseline mean, p-values)
  RESULTS/EVALUATION/baseline_comparison.csv (per-run WCS per task)
  RESULTS/EVALUATION/multidimensional_scores.csv (precision/recall)

No numbers are hard-coded; the baseline distribution is the per-run mean across
the four tasks, matching run_statistical_tests() in the pipeline.

Outputs Fig4.{png,tif,pdf} into RESULTS/EVALUATION/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL = PROJECT_ROOT / "RESULTS" / "EVALUATION"

models = ["Gemini 3 Pro", "Claude Sonnet 4.5", "Claude Opus 4.5",
          "Mistral Large 2", "DeepSeek V3", "GPT-5.2"]

# ── Load locked results ──
stats = pd.read_csv(EVAL / "statistical_tests.csv").set_index("Model").reindex(models)
if stats["WCS"].isna().any():
    raise SystemExit(f"statistical_tests.csv missing models: "
                     f"{stats[stats['WCS'].isna()].index.tolist()}")
llm_performances = stats["WCS"].values
p_values = np.maximum(stats["p_value"].values, 1e-300)

baseline_df = pd.read_csv(EVAL / "baseline_comparison.csv")
baseline_mean_per_run = baseline_df.mean(axis=1).values   # mean across 4 tasks per run
baseline_mean = baseline_mean_per_run.mean()

mm = (pd.read_csv(EVAL / "multidimensional_scores.csv")
      .groupby("Model")[["FactualCorrectness", "SemanticAccuracy"]].mean()
      .reindex(models))
precision_scores = mm["FactualCorrectness"].values   # precision
recall_scores = mm["SemanticAccuracy"].values         # recall

# ── Build Figure 4 ──
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# (A) Performance vs Baseline
ax1.hist(baseline_mean_per_run, bins=30, alpha=0.7, label="Random Baseline",
         color="red", density=True)
ax1.axvline(llm_performances.mean(), color="blue", linestyle="--", linewidth=2,
            label=f"LLM Mean: {llm_performances.mean():.3f}")
ax1.axvline(llm_performances.min(), color="green", linestyle="--", linewidth=2,
            label=f"LLM Min: {llm_performances.min():.3f}")
ax1.set_xlabel("Weighted Coverage Score", fontsize=10)
ax1.set_ylabel("Density", fontsize=10)
ax1.set_title("LLM Performance vs Random Baseline", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# (B) Improvement factors
improvement_factors = llm_performances / baseline_mean
colors_b = plt.cm.viridis(np.linspace(0, 1, len(models)))
bars = ax2.bar(range(len(models)), improvement_factors, color=colors_b)
ax2.set_xlabel("Models", fontsize=10)
ax2.set_ylabel("Improvement Factor (×)", fontsize=10)
ax2.set_title("Improvement Over Random Baseline", fontsize=11, fontweight="bold")
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, fontsize=9, ha="right")
ax2.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, improvement_factors):
    ax2.text(bar.get_x() + bar.get_width() / 2., val + max(improvement_factors) * 0.02,
             f"{val:.1f}×", ha="center", va="bottom", fontweight="bold", fontsize=8)

# (C) Statistical significance (Mann-Whitney U)
significance_colors = ["darkgreen" if p < 0.001 else "orange" if p < 0.05 else "red"
                       for p in p_values]
ax3.bar(range(len(models)), [-np.log10(p) for p in p_values], color=significance_colors)
ax3.axhline(-np.log10(0.001), color="red", linestyle="--", label="p < 0.001", alpha=0.7)
ax3.axhline(-np.log10(0.05), color="orange", linestyle="--", label="p < 0.05", alpha=0.7)
ax3.set_xlabel("Models", fontsize=10)
ax3.set_ylabel("-log₁₀(p-value)", fontsize=10)
ax3.set_title("Statistical Significance vs Baseline (Mann-Whitney U)",
              fontsize=11, fontweight="bold")
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, rotation=45, fontsize=9, ha="right")
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)

# (D) Precision-Recall analysis
scatter = ax4.scatter(precision_scores, recall_scores, s=120, c=llm_performances,
                      cmap="viridis", alpha=0.8, edgecolors="black")
for i, model in enumerate(models):
    ax4.annotate(model, (precision_scores[i], recall_scores[i]),
                 xytext=(8, 8), textcoords="offset points", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
precision_range = np.linspace(0.01, 1, 100)
for f1 in [0.3, 0.5, 0.7]:
    recall_curve = f1 * precision_range / (2 * precision_range - f1)
    recall_curve = np.clip(recall_curve, 0, 1)
    valid = recall_curve > 0
    ax4.plot(precision_range[valid], recall_curve[valid], "--", alpha=0.5, label=f"F1={f1}")
ax4.set_xlabel("Factual Correctness (Precision)", fontsize=10)
ax4.set_ylabel("Semantic Accuracy (Recall)", fontsize=10)
ax4.set_title("Precision-Recall Analysis", fontsize=11, fontweight="bold")
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
ax4.grid(alpha=0.3); ax4.legend(fontsize=8)
cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
cbar.set_label("Overall WCS", fontsize=9)

plt.tight_layout()
for ext in ["png", "tif", "pdf"]:
    fig.savefig(EVAL / f"Fig4.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved Fig4.png/tif/pdf to {EVAL}")
plt.close()
