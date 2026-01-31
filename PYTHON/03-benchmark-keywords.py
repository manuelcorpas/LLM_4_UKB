import csv
import re
import io
import math
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util

##############################################################################
# 1) HARDCODED TOP KEYWORDS & FREQUENCIES
##############################################################################
TOP_KEYWORDS = [
    {"keyword": "Humans", "count": 6547},
    {"keyword": "Female", "count": 3469},
    {"keyword": "Male", "count": 3277},
    {"keyword": "Middle Aged", "count": 2774},
    {"keyword": "United Kingdom", "count": 2689},
    {"keyword": "Aged", "count": 2298},
    {"keyword": "Risk Factors", "count": 2264},
    {"keyword": "Adult", "count": 2014},
    {"keyword": "Genome-Wide Association Study", "count": 1940},
    {"keyword": "Biological Specimen Banks", "count": 1897},
    {"keyword": "Polymorphism, Single Nucleotide", "count": 1390},
    {"keyword": "Genetic Predisposition to Disease", "count": 1336},
    {"keyword": "Prospective Studies", "count": 1304},
    {"keyword": "Cohort Studies", "count": 857},
    {"keyword": "Mendelian Randomization Analysis", "count": 751},
    {"keyword": "Phenotype", "count": 748},
    {"keyword": "Cardiovascular Diseases", "count": 711},
    {"keyword": "Diabetes Mellitus, Type 2", "count": 544},
    {"keyword": "Multifactorial Inheritance", "count": 508},
    {"keyword": "Brain", "count": 499}
]

# For convenience, also store just the keywords:
TOP_20 = [x["keyword"] for x in TOP_KEYWORDS]

##############################################################################
# 2) CSV & OUTPUT
##############################################################################
LLM_RESPONSES_CSV = "DATA/01-most-common-keyword.csv"
OUTPUT_CSV = "RESULTS/BENCHMARK/keywords_results.csv"

##############################################################################
# 3) SYNONYMS
##############################################################################
SYNONYMS = {
    "humans": ["people","participants","human beings","subjects","population"],
    "female": ["women","females"],
    "male": ["men","males"],
    "middle aged": ["middle-aged","middle age","middle-age"],
    "united kingdom": ["uk","britain","england","scotland","wales"],
    "aged": ["elderly","older adults"],
    "risk factors": ["risk factor","risk profile","risk indicator","risk conditions"],
    "adult": ["adults","grown-ups"],
    "genome-wide association study": [
        "gwas","genome wide association",
        "genome-wide association studies","genome wide association studies"
    ],
    "biological specimen banks": [
        "biobanks","bio banks","sample repository","uk biobank"
    ],
    "polymorphism, single nucleotide": [
        "snp","snps","single nucleotide polymorphism","single-nucleotide polymorphisms"
    ],
    "genetic predisposition to disease": [
        "genetic risk","polygenic risk","inherited risk","genetic predisposition"
    ],
    "prospective studies": ["prospective study","longitudinal study"],
    "cohort studies": ["cohort study","cohort-based study","cohort-based analysis"],
    "mendelian randomization analysis": [
        "mendelian randomization","mendelian randomisation","mr"
    ],
    "phenotype": ["traits","phenotypic trait","phenotypic expression"],
    "cardiovascular diseases": ["heart disease","cvd","stroke","hypertension","cardiovascular disease"],
    "diabetes mellitus, type 2": [
        "type 2 diabetes","t2d","type-2 diabetes","adult-onset diabetes"
    ],
    "multifactorial inheritance": ["polygenic inheritance","complex inheritance"],
    "brain": ["brain mri","brain imaging","neuroimaging","cns","cerebral"]
}

##############################################################################
# 4) BERT SEMANTIC MATCHING PARAMS
##############################################################################
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.20  # fairly lenient
DEBUG = True

##############################################################################
# HELPER
##############################################################################

def chunk_text(text):
    # naive sentence approach
    import re
    chunks = re.split(r"[.!?\n]+", text)
    return [c.strip() for c in chunks if c.strip()]

def gather_variants(kw):
    base = kw.lower()
    vs = [kw]
    if base in SYNONYMS:
        vs.extend(SYNONYMS[base])
    return list(set(vs))

##############################################################################
# MAIN
##############################################################################

def main():
    from sentence_transformers import SentenceTransformer, util

    # load model
    model = SentenceTransformer(MODEL_NAME)
    total_freq = sum(x["count"] for x in TOP_KEYWORDS)

    # read CSV
    with open(LLM_RESPONSES_CSV, "r", encoding="utf-8") as f:
        lines = f.readlines()

    first_line = lines[0].lower()
    import csv
    if "model" in first_line and "response" in first_line:
        reader= csv.DictReader(lines)
        rows= list(reader)
    else:
        new_lines= lines[1:]
        fieldnames=["Model","Response"]
        reader= csv.DictReader(new_lines, fieldnames=fieldnames)
        rows= list(reader)

    results = []

    for row in rows:
        model_name = row.get("Model","Unknown").strip()
        raw_text   = row.get("Response","").strip()

        # chunk
        segments = chunk_text(raw_text)
        if not segments:
            results.append({
                "Model":model_name,
                "CoverageScore":0.0,
                "WeightedCoverageScore":0.0,
                "KeywordsFound":0
            })
            continue

        seg_embs = model.encode(segments, convert_to_tensor=True)
        found_count = 0
        weighted_sum=0.0

        if DEBUG:
            print(f"\nModel: {model_name}, #Segments={len(segments)}")

        for kw_info in TOP_KEYWORDS:
            keyword= kw_info["keyword"]
            freq   = kw_info["count"]
            best_sim=0.0

            variants= gather_variants(keyword)
            for var_text in variants:
                var_emb= model.encode(var_text, convert_to_tensor=True)
                sim_scores= util.cos_sim(var_emb, seg_embs)
                max_sim= float(sim_scores.max())
                if max_sim> best_sim:
                    best_sim= max_sim

            if best_sim>= SIMILARITY_THRESHOLD:
                found_count+=1
                weighted_sum+= freq
                if DEBUG:
                    print(f" => MATCH {keyword}, sim={best_sim:.2f}")
            else:
                if DEBUG and best_sim>0.10:
                    print(f" => NO match {keyword}, best_sim={best_sim:.2f}")

        coverage_sc = found_count/len(TOP_KEYWORDS)
        weighted_sc = weighted_sum/total_freq

        results.append({
            "Model": model_name,
            "CoverageScore": coverage_sc,
            "WeightedCoverageScore": weighted_sc,
            "KeywordsFound": found_count
        })

    # sort by WeightedCoverageScore desc
    sorted_data= sorted(results, key=lambda x: x["WeightedCoverageScore"], reverse=True)

    # write CSV
    out_fields= ["Model","CoverageScore","WeightedCoverageScore","KeywordsFound"]
    with open(OUTPUT_CSV,"w",newline="",encoding="utf-8") as outf:
        w= csv.DictWriter(outf, fieldnames=out_fields)
        w.writeheader()
        for rec in sorted_data:
            w.writerow(rec)

    # print
    print("Final sorted results by WeightedCoverageScore (desc):")
    for r in sorted_data:
        print(f"{r['Model']}: coverage={r['CoverageScore']:.2f}, weighted={r['WeightedCoverageScore']:.2f}, found={r['KeywordsFound']}")

    plot_results(sorted_data)


def plot_results(data):
    import numpy as np
    import matplotlib.pyplot as plt

    models = [r["Model"] for r in data]
    coverage= [r["CoverageScore"] for r in data]
    weighted=[r["WeightedCoverageScore"] for r in data]

    x= np.arange(len(models))
    width= 0.35

    fig, ax= plt.subplots(figsize=(10,6))
    b1= ax.bar(x - width/2, coverage, width, label="CoverageScore", color="skyblue")
    b2= ax.bar(x + width/2, weighted, width, label="WeightedCoverageScore", color="salmon")

    ax.set_ylabel("Scores")
    ax.set_title(f"LLM Keyword Benchmark (Coverage-based, threshold={SIMILARITY_THRESHOLD:.2f})")
    ax.set_xticks(x, models, rotation=45, ha='right')
    ax.legend()

    for bar in b1 + b2:
        val=bar.get_height()
        ax.annotate(f"{val:.2f}",
                    xy=(bar.get_x()+ bar.get_width()/2, val),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center',va='bottom',fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()

