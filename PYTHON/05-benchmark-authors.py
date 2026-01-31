import csv
import re
import math
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util

##############################################################################
# 1) HARDCODED TOP 20 AUTHORS & PUBLICATION COUNTS (From Your Chart)
##############################################################################
TOP_AUTHORS = [
    {"author": "George Davey Smith", "count": 122},
    {"author": "Naveed Sattar", "count": 119},
    {"author": "Kari Stefansson", "count": 105},
    {"author": "Caroline Hayward", "count": 94},
    {"author": "Stephen Burgess", "count": 93},
    {"author": "Wei Cheng", "count": 92},
    {"author": "Carlos Celis-Morales", "count": 79},
    {"author": "Pradeep Natarajan", "count": 79},
    {"author": "Jill P. Pell", "count": 78},
    {"author": "Ian J. Deary", "count": 74},
    {"author": "Lu Qi", "count": 73},
    {"author": "Jin-Tai Yu", "count": 72},
    {"author": "Claudia Langenberg", "count": 71},
    {"author": "Jian Yang", "count": 69},
    {"author": "Ole A. Andreassen", "count": 68},
    {"author": "Gudmar Thorleifsson", "count": 67},
    {"author": "Feng Zhang", "count": 66},
    {"author": "Wei Wang", "count": 64},
    {"author": "Jerome I. Rotter", "count": 63},
    {"author": "Nicholas G. Martin", "count": 63}
]

##############################################################################
# 2) PATHS
##############################################################################
LLM_RESPONSES_CSV = "DATA/03-most-prolific-authors.csv"
OUTPUT_CSV        = "RESULTS/BENCHMARK/authors_coverage.csv"

##############################################################################
# 3) OPTIONAL SYNONYMS
##############################################################################
# If the LLM calls them by a short name or variant, add synonyms here in lowercase:
SYNONYMS = {
    # "george davey smith": ["g. d. smith", "dr. george davey smith"], etc.
    # "naveed sattar": ["dr. sattar"], ...
    # You can expand if you want more robust matching
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

def gather_variants(author_name):
    """Collect synonyms for a given author, if any."""
    base = author_name.lower()
    variants = [author_name]  # original
    if base in SYNONYMS:
        variants.extend(SYNONYMS[base])
    return list(set(variants))

##############################################################################
# MAIN
##############################################################################

def main():
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(MODEL_NAME)

    # sum up all publication counts for weighting
    total_pubs = sum(a["count"] for a in TOP_AUTHORS)

    # read the CSV
    with open(LLM_RESPONSES_CSV,"r",encoding="utf-8") as f:
        lines = f.readlines()

    first_line = lines[0].lower()
    import csv
    if "model" in first_line and "response" in first_line:
        reader = csv.DictReader(lines)
        rows = list(reader)
    else:
        # skip header
        new_lines = lines[1:]
        fieldnames = ["Model","Response"]
        reader = csv.DictReader(new_lines, fieldnames=fieldnames)
        rows = list(reader)

    results = []

    for row in rows:
        model_name = row.get("Model","Unknown").strip()
        text_str   = row.get("Response","").strip()

        segments = chunk_text(text_str)
        if not segments:
            # no text => coverage=0
            results.append({
                "Model": model_name,
                "CoverageScore": 0.0,
                "WeightedCoverageScore": 0.0,
                "AuthorsFound": 0
            })
            continue

        # embed the chunks
        chunk_embs = model.encode(segments, convert_to_tensor=True)

        found_count=0
        weighted_sum=0.0

        if DEBUG:
            print(f"\nModel: {model_name}, #Segments={len(segments)}")

        for author_info in TOP_AUTHORS:
            auth_name = author_info["author"]
            auth_count= author_info["count"]  # weighting
            best_sim= 0.0

            # gather synonyms
            variants = gather_variants(auth_name.lower())
            for var_text in variants:
                var_emb = model.encode(var_text, convert_to_tensor=True)
                sim_scores = util.cos_sim(var_emb, chunk_embs)
                max_sim = float(sim_scores.max())
                if max_sim> best_sim:
                    best_sim= max_sim

            if best_sim>= SIMILARITY_THRESHOLD:
                found_count +=1
                weighted_sum+= auth_count
                if DEBUG:
                    print(f" => MATCH {auth_name}, sim={best_sim:.2f}")

        coverage_score = found_count / len(TOP_AUTHORS)
        weighted_score = weighted_sum / total_pubs

        results.append({
            "Model": model_name,
            "CoverageScore": coverage_score,
            "WeightedCoverageScore": weighted_score,
            "AuthorsFound": found_count
        })

    # sort results by WeightedCoverageScore descending
    sorted_data= sorted(results, key=lambda x: x["WeightedCoverageScore"], reverse=True)

    # write CSV
    out_fields= ["Model","CoverageScore","WeightedCoverageScore","AuthorsFound"]
    with open(OUTPUT_CSV,"w",newline="",encoding="utf-8") as outf:
        w = csv.DictWriter(outf, fieldnames=out_fields)
        w.writeheader()
        for rec in sorted_data:
            w.writerow(rec)

    print("\nFinal sorted by WeightedCoverageScore (desc):")
    for rec in sorted_data:
        print(f"{rec['Model']}: coverage={rec['CoverageScore']:.2f}, Weighted={rec['WeightedCoverageScore']:.2f}, authorsFound={rec['AuthorsFound']}")

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
    ax.set_title(f"Top UK Biobank Authors: Weighted Coverage (Threshold={SIMILARITY_THRESHOLD:.2f})")
    ax.set_xticks(x, models, rotation=45, ha='right')
    ax.legend()

    for bar in c_bars + w_bars:
        val = bar.get_height()
        ax.annotate(f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()

