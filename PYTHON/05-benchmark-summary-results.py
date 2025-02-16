import csv
import os
import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# 1) FILE PATHS FOR THE FOUR CSVs
##############################################################################
AUTHORS_CSV = "RESULTS/BENCHMARK/authors_coverage.csv"          # Model,CoverageScore,WeightedCoverageScore,AuthorsFound
INSTITUTIONS_CSV = "RESULTS/BENCHMARK/institutions_coverage.csv" # Model,CoverageScore,WeightedCoverageScore,InstitutionsFound
SUBJECTS_CSV = "RESULTS/BENCHMARK/top-cited-paper-coverage.csv"        # Model,CoverageScore,WeightedCoverageScore,SubjectsFound
KEYWORDS_CSV = "RESULTS/BENCHMARK/keywords_results.csv"        # Model,CoverageScore,WeightedCoverageScore,KeywordsFound

OUTPUT_CSV = "RESULTS/BENCHMARK/final_overall_ranking.csv"
    main()

##############################################################################
# 2) HELPER FUNCTION: LOAD A CSV INTO A DICT
##############################################################################
def load_results(csv_path):
    """
    Returns a dict: {model_name: {"CoverageScore": float, "WeightedCoverageScore": float}, ...}
    """
    results_dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["Model"].strip()
            coverage = float(row["CoverageScore"])
            weighted = float(row["WeightedCoverageScore"])
            results_dict[model] = {
                "CoverageScore": coverage,
                "WeightedCoverageScore": weighted
            }
    return results_dict

##############################################################################
# 3) MAIN MERGING & RANKING
##############################################################################

def main():
    # 1) Load each CSV as a dict
    authors_data = load_results(AUTHORS_CSV)
    institutions_data = load_results(INSTITUTIONS_CSV)
    subjects_data = load_results(SUBJECTS_CSV)
    keywords_data = load_results(KEYWORDS_CSV)

    # 2) Combine into a single structure
    # We'll gather each model's WeightedCoverage from each of the 4 tasks
    # Then compute an overall average WeightedCoverage
    # You can adapt to sum them if you prefer

    all_models = set(authors_data.keys()) | set(institutions_data.keys()) \
                 | set(subjects_data.keys()) | set(keywords_data.keys())

    combined = []
    for model in all_models:
        # fetch WeightedCoverage from each, default 0 if missing
        auth_weighted = authors_data.get(model, {}).get("WeightedCoverageScore", 0.0)
        inst_weighted = institutions_data.get(model, {}).get("WeightedCoverageScore", 0.0)
        subj_weighted = subjects_data.get(model, {}).get("WeightedCoverageScore", 0.0)
        keyw_weighted = keywords_data.get(model, {}).get("WeightedCoverageScore", 0.0)

        # average across 4 tasks
        overall_weighted = (auth_weighted + inst_weighted + subj_weighted + keyw_weighted) / 4.0

        # similarly for coverage if you want (not mandatory)
        auth_cov = authors_data.get(model, {}).get("CoverageScore", 0.0)
        inst_cov = institutions_data.get(model, {}).get("CoverageScore", 0.0)
        subj_cov = subjects_data.get(model, {}).get("CoverageScore", 0.0)
        keyw_cov = keywords_data.get(model, {}).get("CoverageScore", 0.0)
        overall_cov = (auth_cov + inst_cov + subj_cov + keyw_cov) / 4.0

        combined.append({
            "Model": model,
            "AuthorsWeighted": auth_weighted,
            "InstitutionsWeighted": inst_weighted,
            "SubjectsWeighted": subj_weighted,
            "KeywordsWeighted": keyw_weighted,
            "OverallWeighted": overall_weighted,
            "AuthorsCoverage": auth_cov,
            "InstitutionsCoverage": inst_cov,
            "SubjectsCoverage": subj_cov,
            "KeywordsCoverage": keyw_cov,
            "OverallCoverage": overall_cov
        })

    # 3) Sort by OverallWeighted desc
    combined_sorted = sorted(combined, key=lambda x: x["OverallWeighted"], reverse=True)

    # 4) Write final CSV
    fieldnames = [
        "Model",
        "AuthorsWeighted","InstitutionsWeighted","SubjectsWeighted","KeywordsWeighted","OverallWeighted",
        "AuthorsCoverage","InstitutionsCoverage","SubjectsCoverage","KeywordsCoverage","OverallCoverage"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=fieldnames)
        w.writeheader()
        for row in combined_sorted:
            w.writerow(row)

    print(f"Final overall ranking saved to {OUTPUT_CSV}\n")
    print("=== Overall Ranking by Average Weighted Coverage ===")
    for idx, row in enumerate(combined_sorted, start=1):
        print(f"{idx}. {row['Model']}: OverallWeighted={row['OverallWeighted']:.3f}, Coverage={row['OverallCoverage']:.3f}")

    # 5) (Optional) Plot final overall WeightedCoverage
    plot_final_ranking(combined_sorted)

def plot_final_ranking(data):
    # Bar chart of OverallWeighted
    models = [r["Model"] for r in data]
    overall = [r["OverallWeighted"] for r in data]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(x, overall, color="cornflowerblue")

    ax.set_title("Overall Accuracy Ranking (Average Weighted Coverage)")
    ax.set_ylabel("Average Weighted Coverage (Across 4 Tasks)")
    ax.set_xticks(x, models, rotation=45, ha='right')

    # annotate
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x()+bar.get_width()/2,h),
                    xytext=(0,3), textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

