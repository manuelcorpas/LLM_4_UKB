import csv
import re
import math
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util

##############################################################################
# 1) HARDCODED SUBJECTS & CITATION COUNTS
##############################################################################
TOP_SUBJECTS = [
    {"subject": "Sociodemographic and Health-Related Characteristics", "count": 2548},
    {"subject": "Genome-wide association analyses identify risk variants", "count": 2419},
    {"subject": "Genome-wide polygenic scores for common diseases", "count": 2291},
    {"subject": "Body-mass index and all-cause mortality", "count": 1965},
    {"subject": "Gene discovery and polygenic prediction", "count": 1958},
    {"subject": "Genome-wide meta-analysis of depression", "count": 1843},
    {"subject": "Genome-wide meta-analysis identifies new loci", "count": 1777},
    {"subject": "Meta-analysis of genome-wide association studies for height", "count": 1710},
    {"subject": "Multimodal population brain imaging", "count": 1577},
    {"subject": "Identification of novel risk loci and causal insights", "count": 1548}
]

##############################################################################
# 2) PATHS
##############################################################################
LLM_RESPONSES_CSV = "DATA/02-subject-most-cited.csv"
OUTPUT_CSV        = "RESULTS/BENCHMARK/top-cited-paper-coverage.csv"

##############################################################################
# 3) SYNONYMS FOR EACH SUBJECT
##############################################################################
# Lowercase keys; each list has variations
SYNONYMS = {
    "sociodemographic and health-related characteristics": [
        "demographic factors and health",
        "sociodemographic analysis",
        "health-related traits"
    ],
    "genome-wide association analyses identify risk variants": [
        "gwas identifies risk variants",
        "genetic risk variants found by gwas"
    ],
    "genome-wide polygenic scores for common diseases": [
        "polygenic risk scores for diseases",
        "genome wide polygenic score"
    ],
    "body-mass index and all-cause mortality": [
        "bmi and mortality",
        "body mass index mortality"
    ],
    "gene discovery and polygenic prediction": [
        "genetic discovery polygenic",
        "gene discovery, polygenic risk"
    ],
    "genome-wide meta-analysis of depression": [
        "depression meta analysis",
        "gwas meta-analysis depression"
    ],
    "genome-wide meta-analysis identifies new loci": [
        "meta analysis new gwas loci",
        "gwas meta analysis new associations"
    ],
    "meta-analysis of genome-wide association studies for height": [
        "meta analysis height gwas"
    ],
    "multimodal population brain imaging": [
        "population-level imaging",
        "brain imaging multi-modal"
    ],
    "identification of novel risk loci and causal insights": [
        "novel genetic risk loci",
        "risk loci identification",
        "causal genetic insights"
    ]
}

##############################################################################
# 4) BERT SEMANTIC MATCHING PARAMS
##############################################################################
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.20   # quite lenient
DEBUG = True

##############################################################################
# HELPER
##############################################################################

def chunk_text(text):
    """
    Splits text into rough sentence-like chunks.
    """
    parts = re.split(r"[.!?\n]+", text)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks

def gather_variants(subject):
    """
    Return main subject + synonyms in a set. 
    """
    base = subject.lower()
    variants = [subject]  # original
    if base in SYNONYMS:
        variants.extend(SYNONYMS[base])
    return list(set(variants))

##############################################################################
# MAIN
##############################################################################

def main():
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(MODEL_NAME)

    # sum up all citation counts
    total_citations = sum(item["count"] for item in TOP_SUBJECTS)

    # read CSV
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

        segments   = chunk_text(text_str)
        if not segments:
            # no text => coverage = 0
            results.append({
                "Model": model_name,
                "CoverageScore": 0.0,
                "WeightedCoverageScore": 0.0,
                "SubjectsFound": 0
            })
            continue

        # embed the chunks
        chunk_embs = model.encode(segments, convert_to_tensor=True)

        found_count = 0
        weighted_sum= 0.0

        if DEBUG:
            print(f"\nModel: {model_name}, #Segments={len(segments)}")

        for subj_info in TOP_SUBJECTS:
            subj_title = subj_info["subject"]
            subj_cites = subj_info["count"]

            # gather synonyms
            variants = gather_variants(subj_title.lower())
            best_sim = 0.0
            for var_text in variants:
                var_emb = model.encode(var_text, convert_to_tensor=True)
                sim_scores = util.cos_sim(var_emb, chunk_embs)
                max_sim = float(sim_scores.max())
                if max_sim> best_sim:
                    best_sim = max_sim

            if best_sim>= SIMILARITY_THRESHOLD:
                found_count += 1
                weighted_sum += subj_cites
                if DEBUG:
                    print(f" => MATCH {subj_title}, sim={best_sim:.2f}")

        coverage = found_count/len(TOP_SUBJECTS)
        weighted= weighted_sum/total_citations

        results.append({
            "Model": model_name,
            "CoverageScore": coverage,
            "WeightedCoverageScore": weighted,
            "SubjectsFound": found_count
        })

    # sort by WeightedCoverageScore desc
    sorted_data = sorted(results, key=lambda x: x["WeightedCoverageScore"], reverse=True)

    # output CSV
    out_fields=["Model","CoverageScore","WeightedCoverageScore","SubjectsFound"]
    with open(OUTPUT_CSV,"w",newline="",encoding="utf-8") as outf:
        w= csv.DictWriter(outf, fieldnames=out_fields)
        w.writeheader()
        for rec in sorted_data:
            w.writerow(rec)

    print("\nFinal Sorted by WeightedCoverageScore (desc):")
    for rec in sorted_data:
        print(f"{rec['Model']}: coverage={rec['CoverageScore']:.2f}, Weighted={rec['WeightedCoverageScore']:.2f}, found={rec['SubjectsFound']}")

    plot_results(sorted_data)

def plot_results(data):
    models = [r["Model"] for r in data]
    coverage= [r["CoverageScore"] for r in data]
    weighted= [r["WeightedCoverageScore"] for r in data]

    x= np.arange(len(models))
    width=0.35

    fig, ax= plt.subplots(figsize=(10,6))
    c_bars= ax.bar(x-width/2, coverage, width, label="CoverageScore", color="skyblue")
    w_bars= ax.bar(x+width/2, weighted, width, label="WeightedCoverageScore", color="salmon")

    ax.set_ylabel("Scores")
    ax.set_title(f"Top Cited Papers: Weighted Coverage (Threshold={SIMILARITY_THRESHOLD:.2f})")
    ax.set_xticks(x, models, rotation=45, ha='right')
    ax.legend()

    # annotate
    for bar in c_bars + w_bars:
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

