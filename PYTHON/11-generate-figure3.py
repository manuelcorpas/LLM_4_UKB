#!/usr/bin/env python3
"""Generate Figure 3 (multidimensional performance) from the locked CSVs.

Data source: RESULTS/EVALUATION/multidimensional_scores.csv, the deterministic
output of 08-multidimensional-eval.py. No scores are hard-coded here; the
per-model per-dimension matrix is computed by averaging the four task rows per
model, exactly as the pipeline does.

Outputs Fig3.{png,tif,pdf} into RESULTS/EVALUATION/.
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

CSV_DIMS = ["SemanticAccuracy", "FactualCorrectness", "DomainKnowledge",
            "ReasoningQuality", "ResponseDepth", "BiobankSpecificity"]
dimensions = ["Semantic Accuracy", "Factual Correctness", "Domain Knowledge",
              "Reasoning Quality", "Response Depth", "Biobank Specificity"]
models = ["Gemini 3 Pro", "Claude Sonnet 4.5", "Claude Opus 4.5",
          "Mistral Large 2", "DeepSeek V3", "GPT-5.2"]

# ── Load real computed scores from CSV (mean across the 4 tasks per model) ──
raw = pd.read_csv(EVAL / "multidimensional_scores.csv")
model_means = raw.groupby("Model")[CSV_DIMS].mean().reindex(models)
missing = model_means[model_means.isna().any(axis=1)].index.tolist()
if missing:
    raise SystemExit(f"Models missing from multidimensional_scores.csv: {missing}")

scores_df = pd.DataFrame(model_means.values, index=models, columns=dimensions)
scores_matrix = scores_df.values

overall_performance = scores_df.mean(axis=1)
std_scores = scores_df.std(axis=1)
consistency_scores = np.clip(1 - (std_scores / 0.5), 0, 1)

precision_scores = scores_df["Factual Correctness"].values
recall_scores = scores_df["Semantic Accuracy"].values
f1_scores = np.where(
    (precision_scores + recall_scores) > 0,
    2 * (precision_scores * recall_scores) / (precision_scores + recall_scores),
    0.0,
)
rankings = scores_df.rank(ascending=False, method="min").astype(int)

# ── Build Figure 3 ──
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1, 1],
                      width_ratios=[1, 1, 1, 1, 1, 1],
                      hspace=0.55, wspace=0.35,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)

# (A) Radar plot
ax_radar = fig.add_subplot(gs[0, 0:2], projection="polar")
N = len(dimensions)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
for i, (model, color) in enumerate(zip(models, colors)):
    values = scores_matrix[i].tolist() + [scores_matrix[i][0]]
    linestyle = "-" if i < 4 else "--"
    linewidth = 2.5 if i < 4 else 2.0
    alpha = 0.8 if i < 4 else 0.6
    ax_radar.plot(angles, values, linestyle, linewidth=linewidth,
                  label=model, color=color, marker="o", markersize=3, alpha=alpha)
    ax_radar.fill(angles, values, alpha=0.05, color=color)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels([d.replace(" ", "\n") for d in dimensions], fontsize=9)
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax_radar.grid(True, alpha=0.3)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=8, ncol=1)
ax_radar.set_title("Multidimensional Performance (All Models)", fontsize=12,
                   fontweight="bold", pad=20)

# (B) Key Performance Dimensions bar chart
ax_bar = fig.add_subplot(gs[0, 2:6])
key_dims = ["Semantic Accuracy", "Reasoning Quality", "Domain Knowledge"]
key_data = scores_df[key_dims]
x = np.arange(len(models))
width = 0.25
bar_colors = ["#ff9999", "#66b3ff", "#99ff99"]
for i, (dim, color) in enumerate(zip(key_dims, bar_colors)):
    offset = (i - 1) * width
    bars = ax_bar.bar(x + offset, key_data[dim], width, label=dim, color=color, alpha=0.8)
    for bar in bars:
        h = bar.get_height()
        if h > 0.3:
            ax_bar.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
ax_bar.set_xlabel("Models", fontsize=11)
ax_bar.set_ylabel("Performance Score", fontsize=11)
ax_bar.set_title("Key Performance Dimensions", fontsize=12, fontweight="bold")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(models, fontsize=9, rotation=0)
ax_bar.legend(fontsize=10, loc="upper right")
ax_bar.set_ylim(0, 1.1)
ax_bar.grid(axis="y", alpha=0.3)

# (C) Ranking heatmap
ax_hm = fig.add_subplot(gs[1, :])
im = ax_hm.imshow(rankings.T.values, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=6)
for i in range(len(dimensions)):
    for j in range(len(models)):
        ax_hm.text(j, i, int(rankings.iloc[j, i]),
                   ha="center", va="center", color="black", fontweight="bold", fontsize=10)
ax_hm.set_xticks(range(len(models)))
ax_hm.set_xticklabels(models, fontsize=10)
ax_hm.set_yticks(range(len(dimensions)))
ax_hm.set_yticklabels(dimensions, fontsize=10)
ax_hm.set_title("Model Rankings by Dimension (1=Best, 6=Worst)", fontsize=12, fontweight="bold")
cbar = plt.colorbar(im, ax=ax_hm, shrink=0.8)
cbar.set_label("Rank", rotation=270, labelpad=15, fontsize=10)

# (D) Performance summary table
ax_tbl = fig.add_subplot(gs[2, :4])
summary_data = {
    "Model": models,
    "Mean Score": [f"{s:.3f}" for s in overall_performance],
    "Std Dev": [f"{s:.3f}" for s in scores_df.std(axis=1)],
    "Min-Max Range": [f"{row.min():.2f}-{row.max():.2f}" for _, row in scores_df.iterrows()],
    "Consistency": [f"{s:.3f}" for s in consistency_scores],
    "F1 Score": [f"{s:.3f}" for s in f1_scores],
}
df_table = pd.DataFrame(summary_data).sort_values("Mean Score", ascending=False).reset_index(drop=True)
ax_tbl.axis("tight"); ax_tbl.axis("off")
table = ax_tbl.table(cellText=df_table.values, colLabels=df_table.columns,
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.3)
for i in range(len(df_table.columns)):
    table[(0, i)].set_facecolor("#4CAF50")
    table[(0, i)].set_text_props(weight="bold", color="white")
for i in range(1, len(df_table) + 1):
    if i <= 3:
        for j in range(len(df_table.columns)):
            table[(i, j)].set_facecolor("#E8F5E8")
    elif i >= len(df_table) - 1:
        for j in range(len(df_table.columns)):
            table[(i, j)].set_facecolor("#FFE8E8")
ax_tbl.set_title("Performance Summary Statistics (Sorted by Mean Score)",
                 fontsize=12, fontweight="bold", pad=15)

# (E) Distribution plots
ax_d1 = fig.add_subplot(gs[2, 4])
ax_d2 = fig.add_subplot(gs[2, 5])
sem_acc = scores_df["Semantic Accuracy"].sort_values(ascending=False)
short_sorted = [m.replace("Claude ", "C.").replace("Gemini 3 Pro", "Gemini 3\nPro")
                for m in sem_acc.index]
bars1 = ax_d1.bar(range(len(sem_acc)), sem_acc.values,
                  color=plt.cm.viridis(np.linspace(0, 1, len(sem_acc))), alpha=0.8)
ax_d1.set_xlabel("Models (Ranked)", fontsize=10)
ax_d1.set_ylabel("Semantic Accuracy", fontsize=10)
ax_d1.set_title("Semantic Accuracy\nDistribution", fontsize=11, fontweight="bold")
ax_d1.set_xticks(range(len(sem_acc)))
ax_d1.set_xticklabels(short_sorted, rotation=45, fontsize=7, ha="right")
ax_d1.grid(axis="y", alpha=0.3)
for i, (bar, val) in enumerate(zip(bars1, sem_acc.values)):
    if i < 3:
        ax_d1.text(bar.get_x() + bar.get_width() / 2., val + 0.02,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
cons_sorted = consistency_scores.sort_values(ascending=False)
short_cons = [m.replace("Claude ", "C.").replace("Gemini 3 Pro", "Gemini 3\nPro")
              for m in cons_sorted.index]
bars2 = ax_d2.bar(range(len(cons_sorted)), cons_sorted.values,
                  color=plt.cm.plasma(np.linspace(0, 1, len(cons_sorted))), alpha=0.8)
ax_d2.set_xlabel("Models (Ranked)", fontsize=10)
ax_d2.set_ylabel("Consistency Score", fontsize=10)
ax_d2.set_title("Performance Consistency\nDistribution", fontsize=11, fontweight="bold")
ax_d2.set_xticks(range(len(cons_sorted)))
ax_d2.set_xticklabels(short_cons, rotation=45, fontsize=7, ha="right")
ax_d2.grid(axis="y", alpha=0.3)
ax_d2.set_ylim(0, 1.0)
for i, (bar, val) in enumerate(zip(bars2, cons_sorted.values)):
    if i < 3:
        ax_d2.text(bar.get_x() + bar.get_width() / 2., val + 0.02,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

panels = [(ax_radar, "A", (-0.2, 1.1)), (ax_bar, "B", (-0.05, 1.05)),
          (ax_hm, "C", (-0.02, 1.05)), (ax_tbl, "D", (-0.02, 1.05)),
          (ax_d1, "E", (-0.1, 1.05))]
for ax, label, pos in panels:
    ax.text(pos[0], pos[1], label, fontsize=16, fontweight="bold",
            transform=ax.transAxes, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

for ext in ["png", "tif", "pdf"]:
    fig.savefig(EVAL / f"Fig3.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved Fig3.png/tif/pdf to {EVAL}")
plt.close()
