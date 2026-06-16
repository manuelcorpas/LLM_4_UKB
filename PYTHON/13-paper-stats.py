#!/usr/bin/env python3
"""Emit every manuscript-facing number from the locked evaluation CSVs.

Single source of truth for the Results/Discussion text and figure captions.
Reads the deterministic outputs of 08-multidimensional-eval.py:
  RESULTS/EVALUATION/multidimensional_scores.csv
  RESULTS/EVALUATION/statistical_tests.csv

Replicates the pipeline's own definitions:
  mean_score  = mean of the 6 per-dimension means (averaged across the 4 tasks)
  consistency = clip(1 - std_across_dimensions / 0.5, 0, 1)   (pandas std, ddof=1)

No numbers are hard-coded here; everything is computed from the CSVs.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL = PROJECT_ROOT / "RESULTS" / "EVALUATION"

DIMS = [
    "SemanticAccuracy", "FactualCorrectness", "DomainKnowledge",
    "ReasoningQuality", "ResponseDepth", "BiobankSpecificity",
]
DIM_LABEL = {
    "SemanticAccuracy": "semantic accuracy",
    "FactualCorrectness": "factual correctness",
    "DomainKnowledge": "domain knowledge",
    "ReasoningQuality": "reasoning quality",
    "ResponseDepth": "response depth",
    "BiobankSpecificity": "biobank specificity",
}


def main():
    scores = pd.read_csv(EVAL / "multidimensional_scores.csv")
    stats = pd.read_csv(EVAL / "statistical_tests.csv")

    # Per-model per-dimension mean across the 4 tasks
    model_means = scores.groupby("Model")[DIMS].mean()
    mean_score = model_means.mean(axis=1)
    consistency = (1 - model_means.std(axis=1) / 0.5).clip(0, 1)

    print("=" * 70)
    print("LOCKED MANUSCRIPT NUMBERS  (source: RESULTS/EVALUATION/*.csv)")
    print("=" * 70)

    baseline = stats["BaselineMean"].iloc[0]
    print(f"\nRandom baseline mean WCS: {baseline:.4f}")
    print("\n[Table 1 / Figure 4]  Weighted Coverage Score and baseline comparison")
    print(f"{'Model':<20}{'WCS':>7}{'Improv':>8}{'p_value':>12}")
    for _, r in stats.iterrows():
        print(f"{r['Model']:<20}{r['WCS']:>7.3f}{r['ImprovementFactor']:>7.1f}x{r['p_value']:>12.2e}")
    print(f"All significant (p<0.001): {bool(stats['significant'].all())}")
    print(f"Improvement range: {stats['ImprovementFactor'].min():.1f}x to {stats['ImprovementFactor'].max():.1f}x")

    print("\n[Figure 3]  Multidimensional mean score and consistency (ranked)")
    summary = pd.DataFrame({"MeanScore": mean_score, "Consistency": consistency})
    summary = summary.sort_values("MeanScore", ascending=False)
    print(f"{'Model':<20}{'Mean':>8}{'Consistency':>13}")
    for m, r in summary.iterrows():
        print(f"{m:<20}{r['MeanScore']:>8.3f}{r['Consistency']:>13.3f}")

    print("\nPer-dimension means (averaged across tasks):")
    print(model_means.round(3).to_string())

    print("\nPer-dimension LEADER (for the Results text):")
    for d in DIMS:
        leader = model_means[d].idxmax()
        val = model_means[d].max()
        runner = model_means[d].drop(leader).idxmax()
        rval = model_means[d].drop(leader).max()
        print(f"  {DIM_LABEL[d]:<22}: {leader} ({val:.2f}); then {runner} ({rval:.2f})")

    print("\nHeadline claims:")
    print(f"  Top multidimensional mean : {mean_score.idxmax()} ({mean_score.max():.3f})")
    print(f"  Most consistent           : {consistency.idxmax()} ({consistency.max():.3f})")
    print(f"  Top semantic accuracy     : {model_means['SemanticAccuracy'].idxmax()} "
          f"({model_means['SemanticAccuracy'].max():.3f})")

    # Persist a tidy summary for the record
    out = summary.join(model_means.round(4)).round(4)
    out.insert(0, "WCS", stats.set_index("Model")["WCS"])
    out.insert(1, "ImprovementFactor", stats.set_index("Model")["ImprovementFactor"])
    out.to_csv(EVAL / "paper_stats_summary.csv")
    print(f"\nWrote {EVAL / 'paper_stats_summary.csv'}")


if __name__ == "__main__":
    main()
