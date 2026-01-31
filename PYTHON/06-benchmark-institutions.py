import csv
import re
import math
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util

##############################################################################
# 1) HARDCODED TOP 10 INSTITUTIONS & APPLICATION COUNTS (From Your Chart)
##############################################################################
TOP_INSTITUTIONS = [
    {"institution": "University of Oxford", "count": 186},
    {"institution": "University of Cambridge", "count": 74},
    {"institution": "Imperial College London", "count": 69},
    {"institution": "University College London", "count": 69},
    {"institution": "University of Edinburgh", "count": 62},
    {"institution": "University of Manchester", "count": 61},
    {"institution": "UK Biobank Ltd", "count": 60},
    {"institution": "King's College London", "count": 55},
    {"institution": "University of Bristol", "count": 48},
    {"institution": "Sun Yat-Sen University", "count": 39}
]

##############################################################################
# 2) CSV & OUTPUT
##############################################################################
# The CSV should have columns: Model,Response
LLM_RESPONSES_CSV = "DATA/04-top-applicant-institutions.csv"
OUTPUT_CSV        = "RESULTS/BENCHMARK/institutions_coverage.csv"

##############################################################################
# 3) OPTIONAL SYNONYMS
##############################################################################
# If an LLM refers to “Oxford University” instead of “University of Oxford,”
# add a synonyms entry. Use lowercase keys.

SYNONYMS = {
    "university of oxford": [
        "oxford university"
    ],
    "university of cambridge": [
        "cambridge university"
    ],
    "imperial college london": [
        "imperial college"
    ],
    "university college london": [
        "ucl"
    ],
    "university of edinburgh": [
        "edinburgh university"
    ],
    "university of manchester": [
        "manchester university"
    ],
    "uk biobank ltd": [
        "uk biobank"
    ],
    "king's college london": [
        "kings college london",
        "king’s college london"
    ],
    "university of bristol": [
        "bristol university"
    ],
    "sun yat-sen university": [
        "sysu"
    ]
}

##############################################################################
# 4) BERT SEMANTIC MATCHING PARAMS
##############################################################################
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.20  # fairly lenient
DEBUG = True

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def chunk_text(text):
    """
    Splits text into naive sentence-level chunks.
    """
    parts = re.split(r"[.!?\n]+", text)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks

def gather_variants(inst_name):
    """
    Return the main institution name + synonyms, in a list.
    """
    base = inst_name.lower()
    variants = [inst_name]  # original (mixed-case)
    if base in SYNONYMS:
        variants.extend(SYNONYMS[base])
    return list(set(variants))

##############################################################################
# MAIN SCRIPT
##############################################################################

def main():
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(MODEL_NAME)

    # total sum of application counts
    total_apps = sum(i["count"] for i in TOP_INSTITUTIONS)

    # read CSV lines
    with open(LLM_RESPONSES_CSV, "r", encoding="utf-8") as f:
        lines = f.readlines()

    first_line = lines[0].lower()
    if "model" in first_line and "response" in first_line:
        reader = csv.DictReader(lines)
        rows = list(reader)
    else:
        new_lines = lines[1:]
        fieldnames = ["Model","Response"]
        reader = csv.DictReader(new_lines, fieldnames=fieldnames)
        rows = list(reader)

    results = []

    for row in rows:
        model_name = row.get("Model","Unknown").strip()
        text_str   = row.get("Response","").strip()

        # chunk into sentences
        segments = chunk_text(text_str)
        if not segments:
            # no text => coverage = 0
            results.append({
                "Model": model_name,
                "CoverageScore": 0.0,
                "WeightedCoverageScore": 0.0,
                "InstitutionsFound": 0
            })
            continue

        # embed each chunk
        seg_embs = model.encode(segments, convert_to_tensor=True)

        found_count=0
        weighted_sum=0.0

        if DEBUG:
            print(f"\nModel: {model_name}, #Segments={len(segments)}")

        for inst in TOP_INSTITUTIONS:
            inst_name  = inst["institution"]
            inst_count = inst["count"]
            best_sim= 0.0

            # gather synonyms
            variants = gather_variants(inst_name)
            for var_text in variants:
                var_emb = model.encode(var_text.lower(), convert_to_tensor=True)
                sim_scores = util.cos_sim(var_emb, seg_embs)
                max_sim= float(sim_scores.max())
                if max_sim> best_sim:
                    best_sim= max_sim

            if best_sim>= SIMILARITY_THRESHOLD:
                found_count+=1
                weighted_sum+= inst_count
                if DEBUG:
                    print(f" => MATCH {inst_name}, sim={best_sim:.2f}")

        coverage = found_count / len(TOP_INSTITUTIONS)
        weighted= weighted_sum / total_apps

        results.append({
            "Model": model_name,
            "CoverageScore": coverage,
            "WeightedCoverageScore": weighted,
            "InstitutionsFound": found_count
        })

    # sort by WeightedCoverageScore desc
    sorted_data= sorted(results, key=lambda x: x["WeightedCoverageScore"], reverse=True)

    # write CSV
    out_fields= ["Model","CoverageScore","WeightedCoverageScore","InstitutionsFound"]
    with open( "RESULTS/BENCHMARK/institutions_coverage.csv","w", newline="",encoding="utf-8") as outf:
        w= csv.DictWriter(outf, fieldnames=out_fields)
        w.writeheader()
        for rec in sorted_data:
            w.writerow(rec)

    print("\nFinal sorted by WeightedCoverageScore (desc):")
    for rec in sorted_data:
        print(f"{rec['Model']}: coverage={rec['CoverageScore']:.2f}, Weighted={rec['WeightedCoverageScore']:.2f}, found={rec['InstitutionsFound']}")

    plot_results(sorted_data)

def plot_results(data):
    models = [r["Model"] for r in data]
    coverage = [r["CoverageScore"] for r in data]
    weighted = [r["WeightedCoverageScore"] for r in data]

    x= np.arange(len(models))
    width=0.35

    fig, ax= plt.subplots(figsize=(10,6))
    b1= ax.bar(x - width/2, coverage, width, label="CoverageScore", color="skyblue")
    b2= ax.bar(x + width/2, weighted, width, label="WeightedCoverageScore", color="salmon")

    ax.set_ylabel("Scores")
    ax.set_title(f"Top Institutions by Applications: Weighted Coverage (threshold={SIMILARITY_THRESHOLD:.2f})")
    ax.set_xticks(x, models, rotation=45, ha='right')
    ax.legend()

    for bar in b1 + b2:
        val= bar.get_height()
        ax.annotate(f"{val:.2f}",
                    xy=(bar.get_x()+bar.get_width()/2, val),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    main()

