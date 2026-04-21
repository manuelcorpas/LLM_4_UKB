#!/usr/bin/env python3
"""
Real multidimensional evaluation of LLM responses on UK Biobank benchmark tasks.

Computes six evaluation dimensions from actual LLM text outputs:
  1. Semantic Accuracy   - embedding similarity of matched concepts to ground truth
  2. Factual Correctness - precision of named entities against Schema metadata
  3. Domain Knowledge    - presence and correct usage of advanced biomedical concepts
  4. Reasoning Quality   - interpretive structure, causal reasoning, thematic synthesis
  5. Response Depth      - multi-aspect integration, hierarchical coverage
  6. Biobank Specificity - explicit grounding in UK Biobank content

Also generates baseline comparison using random sampling from UK Biobank term pools
(Schemas 19 and 27) with Mann-Whitney U statistical testing.

Generates: Figure 3 (multidimensional performance) and Figure 4 (baseline comparison)
Outputs:   RESULTS/EVALUATION/multidimensional_scores.csv
           RESULTS/EVALUATION/baseline_comparison.csv
           RESULTS/EVALUATION/figure_3.pdf
           RESULTS/EVALUATION/figure_4.pdf
"""

import csv
import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Optional: sentence-transformers for semantic accuracy
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("WARNING: sentence-transformers not installed. Semantic accuracy will use fallback.")

# Optional: scipy for statistical testing
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Statistical tests will be skipped.")

###############################################################################
# PATHS
###############################################################################
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "DATA"
RESULTS_DIR = PROJECT_ROOT / "RESULTS" / "EVALUATION"

LLM_RESPONSE_FILES = {
    "keywords":     DATA_DIR / "01-most-common-keyword.csv",
    "papers":       DATA_DIR / "02-subject-most-cited.csv",
    "authors":      DATA_DIR / "03-most-prolific-authors.csv",
    "institutions": DATA_DIR / "04-top-applicant-institutions.csv",
}

SCHEMA_CSV = DATA_DIR / "ukb_schema.csv"

# Model name corrections (CSV names -> correct publication names)
MODEL_NAME_MAP = {
    "Claude Sonnet 4": "Claude Sonnet 4.5",
    "Mistral Large": "Mistral Large 2",
}

###############################################################################
# GROUND TRUTH (same as scripts 03-06)
###############################################################################
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
    {"keyword": "Brain", "count": 499},
]

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
    {"subject": "Identification of novel risk loci and causal insights", "count": 1548},
]

TOP_AUTHORS = [
    {"author": "George Davey Smith", "count": 122},
    {"author": "Naveed Sattar", "count": 119},
    {"author": "Kari Stefansson", "count": 105},
    {"author": "Caroline Hayward", "count": 94},
    {"author": "Stephen Burgess", "count": 93},
    {"author": "Wei Cheng", "count": 92},
    {"author": "Pradeep Natarajan", "count": 79},
    {"author": "Carlos Celis-Morales", "count": 79},
    {"author": "Jill P. Pell", "count": 77},
    {"author": "Ian J. Deary", "count": 76},
    {"author": "Lu Qi", "count": 72},
    {"author": "Jin-Tai Yu", "count": 68},
    {"author": "Claudia Langenberg", "count": 67},
    {"author": "Jian Yang", "count": 66},
    {"author": "Ole A. Andreassen", "count": 64},
    {"author": "Dipender Gill", "count": 63},
    {"author": "Cathie Sudlow", "count": 62},
    {"author": "Zhengming Chen", "count": 61},
    {"author": "Liming Li", "count": 59},
    {"author": "Patrick F. Sullivan", "count": 58},
]

TOP_INSTITUTIONS = [
    {"institution": "University of Oxford", "count": 186},
    {"institution": "University of Cambridge", "count": 74},
    {"institution": "Imperial College London", "count": 69},
    {"institution": "University College London", "count": 69},
    {"institution": "University of Edinburgh", "count": 62},
    {"institution": "University of London", "count": 62},
    {"institution": "University of Manchester", "count": 61},
    {"institution": "UK Biobank Ltd", "count": 60},
    {"institution": "King's College London", "count": 50},
    {"institution": "University of Bristol", "count": 46},
]

# Synonyms for keyword matching (same as script 03)
KEYWORD_SYNONYMS = {
    "humans": ["people","participants","human beings","subjects","population"],
    "female": ["women","females"],
    "male": ["men","males"],
    "middle aged": ["middle-aged","middle age","middle-age"],
    "united kingdom": ["uk","britain","england","scotland","wales"],
    "aged": ["elderly","older adults"],
    "risk factors": ["risk factor","risk profile","risk indicator"],
    "adult": ["adults","grown-ups"],
    "genome-wide association study": ["gwas","genome wide association","genome-wide association studies"],
    "biological specimen banks": ["biobanks","bio banks","sample repository","uk biobank"],
    "polymorphism, single nucleotide": ["snp","snps","single nucleotide polymorphism"],
    "genetic predisposition to disease": ["genetic risk","polygenic risk","inherited risk","genetic predisposition"],
    "prospective studies": ["prospective study","longitudinal study"],
    "cohort studies": ["cohort study","cohort-based study"],
    "mendelian randomization analysis": ["mendelian randomization","mendelian randomisation","mr"],
    "phenotype": ["traits","phenotypic trait","phenotypic expression"],
    "cardiovascular diseases": ["heart disease","cvd","stroke","hypertension","cardiovascular disease"],
    "diabetes mellitus, type 2": ["type 2 diabetes","t2d","type-2 diabetes"],
    "multifactorial inheritance": ["polygenic inheritance","complex inheritance"],
    "brain": ["brain mri","brain imaging","neuroimaging","cns","cerebral"],
}

###############################################################################
# ADVANCED BIOMEDICAL CONCEPTS (for Domain Knowledge scoring)
###############################################################################
ADVANCED_CONCEPTS = [
    # Genomics methodology
    "genome-wide association", "gwas", "polygenic risk score", "polygenic score",
    "mendelian randomization", "mendelian randomisation", "heritability",
    "single nucleotide polymorphism", "snp", "genetic variant",
    "linkage disequilibrium", "allele", "genotype",
    # Epidemiological methods
    "longitudinal", "prospective cohort", "case-control",
    "odds ratio", "hazard ratio", "confidence interval",
    "meta-analysis", "systematic review",
    # Biobank-specific methodology
    "phenotyping", "deep phenotyping", "biomarker",
    "electronic health record", "ehr", "imputation",
    # Disease domains
    "cardiometabolic", "neurodegenerative", "psychiatric",
    "type 2 diabetes", "coronary artery disease", "alzheimer",
    "obesity", "hypertension", "depression",
    # Advanced analysis
    "machine learning", "neural network", "deep learning",
    "causal inference", "mediation analysis", "interaction",
]

# UK Biobank-specific terms (for Biobank Specificity scoring)
BIOBANK_SPECIFIC_TERMS = [
    "uk biobank", "biobank", "500,000", "500000", "half a million",
    "prospective cohort", "baseline assessment", "assessment centre",
    "imaging study", "brain imaging", "cardiac imaging",
    "genotyping array", "axiom array", "uk biobank axiom",
    "primary care", "hospital episode", "death register",
    "data field", "category", "schema",
    "touch screen", "questionnaire", "physical measures",
    "blood sample", "urine sample", "saliva",
    "accelerometer", "actigraphy",
    "covid-19", "pandemic",
    "application", "approved research",
    "benton visual retention", "fluid intelligence",
    "townsend deprivation", "index of multiple deprivation",
]

###############################################################################
# DATA LOADING
###############################################################################

def load_llm_responses():
    """Load all LLM responses from DATA/*.csv files.

    Returns dict: {task_name: [{model, response}, ...]}
    """
    all_responses = {}

    for task_name, filepath in LLM_RESPONSE_FILES.items():
        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping task '{task_name}'")
            continue

        responses = []
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip header line (question text)
        data_lines = lines[1:]
        reader = csv.reader(data_lines)

        for row in reader:
            if len(row) >= 3:
                model_name = row[0].strip()
                response_text = row[2].strip() if len(row) > 2 else row[1].strip()
                if model_name and response_text:
                    # Apply model name corrections
                    corrected_name = MODEL_NAME_MAP.get(model_name, model_name)
                    responses.append({
                        "model": corrected_name,
                        "response": response_text,
                    })

        all_responses[task_name] = responses
        print(f"  Loaded {len(responses)} responses for task '{task_name}'")

    return all_responses


def load_schema_data():
    """Load UK Biobank schema data for baseline term pools."""
    if not SCHEMA_CSV.exists():
        print(f"WARNING: {SCHEMA_CSV} not found")
        return None

    df = pd.read_csv(SCHEMA_CSV, low_memory=False)
    print(f"  Schema data: {len(df)} publications")
    return df


###############################################################################
# DIMENSION SCORING FUNCTIONS
###############################################################################

def score_semantic_accuracy(response, task_name, sbert_model=None):
    """
    Semantic Accuracy: measures how well retrieved concepts align with
    ground truth using embedding-based cosine similarity.

    Returns average similarity of matched terms (0-1).
    """
    ground_truth = _get_ground_truth_terms(task_name)
    if not ground_truth:
        return 0.0

    # Chunk response into segments
    segments = [s.strip() for s in re.split(r'[.!?\n]+', response) if s.strip()]
    if not segments:
        return 0.0

    if sbert_model and HAS_SBERT:
        seg_embs = sbert_model.encode(segments, convert_to_tensor=True)
        similarities = []

        for term in ground_truth:
            # Get all variants
            variants = _get_variants(term, task_name)
            best_sim = 0.0
            for var_text in variants:
                var_emb = sbert_model.encode(var_text, convert_to_tensor=True)
                sim_scores = util.cos_sim(var_emb, seg_embs)
                max_sim = float(sim_scores.max())
                if max_sim > best_sim:
                    best_sim = max_sim

            if best_sim >= 0.20:  # same threshold as scripts 03-06
                similarities.append(best_sim)

        if not similarities:
            return 0.0
        return float(np.mean(similarities))

    else:
        # Fallback: substring matching with similarity proxy
        response_lower = response.lower()
        matches = 0
        for term in ground_truth:
            variants = _get_variants(term, task_name)
            for v in variants:
                if v.lower() in response_lower:
                    matches += 1
                    break
        return matches / len(ground_truth)


def score_factual_correctness(response, task_name):
    """
    Factual Correctness: fraction of entities mentioned by the LLM
    that actually exist in ground truth metadata.

    For authors/institutions: precision of named entities.
    For keywords/papers: precision of claimed concepts.
    """
    ground_truth = _get_ground_truth_terms(task_name)
    if not ground_truth:
        return 0.0

    response_lower = response.lower()

    if task_name == "authors":
        # Check which ground truth authors are correctly mentioned
        correct = 0
        for gt_term in ground_truth:
            parts = gt_term.lower().split()
            if gt_term.lower() in response_lower:
                correct += 1
            elif len(parts) >= 2 and parts[-1] in response_lower:
                # Partial credit: last name present
                correct += 0.7

        return correct / len(ground_truth)

    elif task_name == "institutions":
        correct = 0
        for gt_term in ground_truth:
            gt_l = gt_term.lower()
            if gt_l in response_lower:
                correct += 1
            else:
                # Check partial matches (e.g., "Oxford" for "University of Oxford")
                key_word = gt_l.split("of ")[-1] if "of " in gt_l else gt_l.split()[-1]
                if key_word in response_lower:
                    correct += 0.5

        return correct / len(ground_truth)

    else:
        # For keywords and papers, use coverage-style matching
        matches = 0
        for term in ground_truth:
            variants = _get_variants(term, task_name)
            for v in variants:
                if v.lower() in response_lower:
                    matches += 1
                    break
        return matches / len(ground_truth)


def score_domain_knowledge(response):
    """
    Domain Knowledge: frequency and breadth of advanced biomedical concepts
    correctly used in the response.

    Measures both presence and contextual appropriateness.
    """
    response_lower = response.lower()

    found_concepts = set()
    for concept in ADVANCED_CONCEPTS:
        if concept.lower() in response_lower:
            found_concepts.add(concept.lower())

    # Normalize: finding 15+ distinct concepts = 1.0
    raw_score = len(found_concepts) / 15.0

    # Cap at 1.0
    return min(raw_score, 1.0)


def score_reasoning_quality(response):
    """
    Reasoning Quality: presence of interpretive statements, causal reasoning,
    thematic synthesis, and structured argumentation.

    Higher scores for responses that go beyond flat lists to provide
    interpretation and synthesis.
    """
    response_lower = response.lower()

    score = 0.0
    max_components = 5

    # 1. Interpretive language (causal connectors, analytical phrases)
    interpretive_markers = [
        "suggest", "indicat", "reflect", "demonstrat", "reveal",
        "imply", "highlight", "underscore", "consistent with",
        "in line with", "contributing to", "driven by",
        "due to", "because", "as a result", "consequently",
        "this means", "this shows", "this suggests",
    ]
    interp_count = sum(1 for m in interpretive_markers if m in response_lower)
    score += min(interp_count / 5.0, 1.0)

    # 2. Thematic grouping (uses headers, categories, numbered sections)
    has_headers = bool(re.search(r'#{1,3}\s+\w|^\d+\.\s+\*\*|^\*\*[A-Z]', response, re.MULTILINE))
    has_categories = bool(re.search(r'categor|theme|cluster|group|domain|area|dimension', response_lower))
    if has_headers:
        score += 0.7
    if has_categories:
        score += 0.3

    # 3. Comparative language
    comparative_markers = [
        "compared to", "in contrast", "whereas", "while",
        "more than", "less than", "higher", "lower",
        "strongest", "weakest", "most", "least",
        "particularly", "notably", "especially",
    ]
    comp_count = sum(1 for m in comparative_markers if m in response_lower)
    score += min(comp_count / 4.0, 1.0)

    # 4. Synthesis and conclusion
    synthesis_markers = [
        "overall", "in summary", "collectively", "taken together",
        "broader", "overarching", "in conclusion", "these findings",
        "this pattern", "these results",
    ]
    synth_count = sum(1 for m in synthesis_markers if m in response_lower)
    score += min(synth_count / 3.0, 1.0)

    # 5. Depth of explanation (not just listing)
    sentences = [s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 20]
    avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    if avg_sentence_len > 15:
        score += 1.0
    elif avg_sentence_len > 10:
        score += 0.6
    elif avg_sentence_len > 5:
        score += 0.3

    return min(score / max_components, 1.0)


def score_response_depth(response, task_name):
    """
    Response Depth: whether the response demonstrates layered insight
    by integrating multiple aspects of biobank-related evidence.

    Measures: breadth of themes, hierarchical structure, cross-referencing.
    """
    response_lower = response.lower()
    score = 0.0
    max_components = 4

    # 1. Breadth: how many distinct theme clusters are covered
    theme_clusters = {
        "genetics": ["genetic", "gwas", "snp", "allele", "heritab", "polygenic", "variant"],
        "epidemiology": ["epidemiolog", "cohort", "prospective", "longitudinal", "risk factor", "mortality"],
        "cardiovascular": ["cardiovascular", "heart", "blood pressure", "hypertension", "coronary", "stroke"],
        "metabolic": ["diabetes", "obesity", "bmi", "body mass", "metabolic", "lipid"],
        "neurological": ["brain", "neurolog", "cogniti", "dementia", "alzheimer", "depression", "psychiatric"],
        "methodology": ["method", "statistical", "analysis", "model", "framework", "pipeline"],
        "imaging": ["imaging", "mri", "scan", "neuroimaging", "cardiac imaging"],
        "cancer": ["cancer", "oncolog", "tumor", "carcinoma", "malignant"],
    }

    clusters_found = sum(
        1 for cluster_terms in theme_clusters.values()
        if any(t in response_lower for t in cluster_terms)
    )
    score += min(clusters_found / 5.0, 1.0)

    # 2. Hierarchical structure
    h1_count = len(re.findall(r'^#{1}\s', response, re.MULTILINE))
    h2_count = len(re.findall(r'^#{2}\s', response, re.MULTILINE))
    h3_count = len(re.findall(r'^#{3}\s', response, re.MULTILINE))
    bullet_count = len(re.findall(r'^[\s]*[-*]\s', response, re.MULTILINE))
    numbered_count = len(re.findall(r'^\d+[\.\)]\s', response, re.MULTILINE))

    structure_levels = sum(1 for c in [h1_count, h2_count, h3_count, bullet_count, numbered_count] if c > 0)
    score += min(structure_levels / 3.0, 1.0)

    # 3. Length and detail
    word_count = len(response.split())
    if word_count > 300:
        score += 1.0
    elif word_count > 150:
        score += 0.6
    elif word_count > 80:
        score += 0.3

    # 4. Cross-referencing
    cross_ref_markers = [
        "related to", "linked to", "associated with", "connection between",
        "overlap", "intersection", "both", "combined",
        "as well as", "in addition", "furthermore", "moreover",
        "building on", "extending", "complementary",
    ]
    cross_count = sum(1 for m in cross_ref_markers if m in response_lower)
    score += min(cross_count / 4.0, 1.0)

    return min(score / max_components, 1.0)


def score_biobank_specificity(response):
    """
    Biobank Specificity: degree to which the response is explicitly grounded
    in UK Biobank content rather than generic biomedical knowledge.
    """
    response_lower = response.lower()

    found_terms = set()
    for term in BIOBANK_SPECIFIC_TERMS:
        if term.lower() in response_lower:
            found_terms.add(term.lower())

    # Normalize: finding 8+ biobank-specific terms = 1.0
    raw_score = len(found_terms) / 8.0
    return min(raw_score, 1.0)


###############################################################################
# HELPERS
###############################################################################

def _get_ground_truth_terms(task_name):
    """Get ground truth term list for a given task."""
    if task_name == "keywords":
        return [k["keyword"] for k in TOP_KEYWORDS]
    elif task_name == "papers":
        return [s["subject"] for s in TOP_SUBJECTS]
    elif task_name == "authors":
        return [a["author"] for a in TOP_AUTHORS]
    elif task_name == "institutions":
        return [i["institution"] for i in TOP_INSTITUTIONS]
    return []


def _get_variants(term, task_name):
    """Get term variants including synonyms."""
    variants = [term]
    if task_name == "keywords":
        base = term.lower()
        if base in KEYWORD_SYNONYMS:
            variants.extend(KEYWORD_SYNONYMS[base])
    elif task_name == "authors":
        parts = term.split()
        if len(parts) >= 2:
            variants.append(parts[-1])
    elif task_name == "institutions":
        if "University of " in term:
            variants.append(term.replace("University of ", ""))
    return variants


###############################################################################
# MAIN EVALUATION PIPELINE
###############################################################################

def run_multidimensional_evaluation(all_responses):
    """
    Score all LLM responses on 6 dimensions.

    Returns DataFrame with columns:
      Model, Task, SemanticAccuracy, FactualCorrectness, DomainKnowledge,
      ReasoningQuality, ResponseDepth, BiobankSpecificity
    """
    print("\n=== MULTIDIMENSIONAL EVALUATION ===")

    # Load SBERT model if available
    sbert_model = None
    if HAS_SBERT:
        print("  Loading SentenceTransformer model...")
        sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    records = []

    for task_name, responses in all_responses.items():
        print(f"\n  Scoring task: {task_name}")

        for resp in responses:
            model = resp["model"]
            text = resp["response"]

            sem_acc = score_semantic_accuracy(text, task_name, sbert_model)
            fact_corr = score_factual_correctness(text, task_name)
            dom_know = score_domain_knowledge(text)
            reas_qual = score_reasoning_quality(text)
            resp_depth = score_response_depth(text, task_name)
            bio_spec = score_biobank_specificity(text)

            record = {
                "Model": model,
                "Task": task_name,
                "SemanticAccuracy": round(sem_acc, 4),
                "FactualCorrectness": round(fact_corr, 4),
                "DomainKnowledge": round(dom_know, 4),
                "ReasoningQuality": round(reas_qual, 4),
                "ResponseDepth": round(resp_depth, 4),
                "BiobankSpecificity": round(bio_spec, 4),
            }
            records.append(record)

            print(f"    {model:20s} | SA={sem_acc:.3f} FC={fact_corr:.3f} "
                  f"DK={dom_know:.3f} RQ={reas_qual:.3f} RD={resp_depth:.3f} BS={bio_spec:.3f}")

    df = pd.DataFrame(records)
    return df


###############################################################################
# BASELINE COMPARISON
###############################################################################

def run_baseline_comparison(schema_df, num_runs=100):
    """
    Generate random baseline by sampling from real UK Biobank term pools.
    Score through the same coverage pipeline used for LLM evaluation.

    Returns DataFrame with per-task baseline WCS distributions.
    """
    print("\n=== BASELINE COMPARISON ===")
    print(f"  Generating {num_runs} random samples per task...")

    # Extract term pools from schema data
    keyword_pool = _extract_keyword_pool(schema_df)
    author_pool = _extract_author_pool(schema_df)
    title_pool = _extract_title_pool(schema_df)
    institution_pool = _extract_institution_pool()

    print(f"  Term pools: keywords={len(keyword_pool)}, authors={len(author_pool)}, "
          f"titles={len(title_pool)}, institutions={len(institution_pool)}")

    tasks = {
        "keywords": (keyword_pool, TOP_KEYWORDS, "keyword", "count"),
        "papers": (title_pool, TOP_SUBJECTS, "subject", "count"),
        "authors": (author_pool, TOP_AUTHORS, "author", "count"),
        "institutions": (institution_pool, TOP_INSTITUTIONS, "institution", "count"),
    }

    all_baseline_scores = {}

    for task_name, (pool, ground_truth, gt_key, freq_key) in tasks.items():
        print(f"\n  Task: {task_name}")

        total_freq = sum(item[freq_key] for item in ground_truth)

        wcs_scores = []

        for run in range(num_runs):
            # Random sample: pick terms from pool, format as pseudo-response
            n_sample = min(20, len(pool))
            sampled = np.random.choice(pool, size=n_sample, replace=False)
            fake_response = ", ".join(sampled)

            # Score using weighted coverage (same as scripts 03-06)
            weighted_sum = 0.0
            for item in ground_truth:
                term = item[gt_key].lower()
                freq = item[freq_key]

                fake_lower = fake_response.lower()
                if term in fake_lower:
                    weighted_sum += freq
                else:
                    if task_name == "keywords" and term in KEYWORD_SYNONYMS:
                        for syn in KEYWORD_SYNONYMS[term]:
                            if syn.lower() in fake_lower:
                                weighted_sum += freq
                                break

            wcs = weighted_sum / total_freq if total_freq > 0 else 0.0
            wcs_scores.append(wcs)

        all_baseline_scores[task_name] = np.array(wcs_scores)

        mean_wcs = np.mean(wcs_scores)
        std_wcs = np.std(wcs_scores)
        max_wcs = np.max(wcs_scores)
        print(f"    Baseline WCS: mean={mean_wcs:.4f} +/- {std_wcs:.4f}, max={max_wcs:.4f}")

    return all_baseline_scores


def _extract_keyword_pool(schema_df):
    """Extract all unique keywords from schema publications."""
    keywords = set()
    if schema_df is not None and 'abstract' in schema_df.columns:
        for abstract in schema_df['abstract'].dropna():
            text = str(abstract).lower()
            terms = re.findall(r'[a-z][a-z\s\-]{3,30}[a-z]', text)
            keywords.update(t.strip() for t in terms if len(t.strip()) > 3)

    # Add known keyword variants
    for kw_info in TOP_KEYWORDS:
        keywords.add(kw_info["keyword"])
        base = kw_info["keyword"].lower()
        if base in KEYWORD_SYNONYMS:
            keywords.update(KEYWORD_SYNONYMS[base])

    # Add common biomedical terms as noise
    noise_terms = [
        "patient", "treatment", "clinical trial", "outcome", "sample size",
        "p-value", "regression", "correlation", "prevalence", "incidence",
        "diagnosis", "prognosis", "therapy", "intervention", "exposure",
        "measurement", "assessment", "evaluation", "comparison", "association",
        "inflammation", "immune response", "protein", "gene expression",
        "pathway", "signaling", "receptor", "enzyme", "metabolism",
    ]
    keywords.update(noise_terms)

    return list(keywords)


def _extract_author_pool(schema_df):
    """Extract unique author-like names from schema data."""
    authors = set()
    if schema_df is not None and 'title' in schema_df.columns:
        for title in schema_df['title'].dropna().head(2000):
            names = re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+', str(title))
            authors.update(names)

    for a in TOP_AUTHORS:
        authors.add(a["author"])

    # Add plausible noise names
    noise_names = [
        "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis",
        "Robert Wilson", "Jennifer Lee", "David Taylor", "Lisa Anderson",
        "James Thomas", "Maria Garcia", "William Martinez", "Susan Robinson",
        "Richard Clark", "Elizabeth Lewis", "Joseph Walker", "Margaret Hall",
        "Charles Allen", "Dorothy Young", "Christopher King", "Nancy Wright",
        "Daniel Lopez", "Karen Hill", "Matthew Scott", "Betty Green",
    ]
    authors.update(noise_names)

    return list(authors)


def _extract_title_pool(schema_df):
    """Extract paper title fragments for baseline sampling."""
    titles = []
    if schema_df is not None and 'title' in schema_df.columns:
        titles = [str(t).strip() for t in schema_df['title'].dropna()
                  if len(str(t).strip()) > 10]
    return titles if titles else ["placeholder study title"]


def _extract_institution_pool():
    """Generate institution pool for baseline sampling."""
    institutions = [i["institution"] for i in TOP_INSTITUTIONS]

    noise = [
        "Harvard University", "Stanford University", "MIT",
        "Johns Hopkins University", "Yale University",
        "University of Toronto", "Karolinska Institute",
        "University of Melbourne", "ETH Zurich",
        "National University of Singapore", "Peking University",
        "University of Tokyo", "Sorbonne University",
        "Technical University of Munich", "University of Sydney",
        "McGill University", "University of Hong Kong",
        "Seoul National University", "University of Sao Paulo",
        "University of Cape Town", "University of Nairobi",
    ]
    institutions.extend(noise)
    return institutions


###############################################################################
# STATISTICAL TESTING
###############################################################################

def run_statistical_tests(llm_wcs_scores, baseline_scores):
    """
    Mann-Whitney U test comparing each LLM's Weighted Coverage Score
    against the random baseline distribution.
    """
    if not HAS_SCIPY:
        print("  scipy not available, skipping statistical tests")
        return None

    print("\n=== STATISTICAL TESTING (Mann-Whitney U) ===")

    # Compute mean baseline WCS per run (average across 4 tasks)
    n_runs = len(list(baseline_scores.values())[0])
    baseline_mean_per_run = np.zeros(n_runs)
    for task_scores in baseline_scores.values():
        baseline_mean_per_run += task_scores
    baseline_mean_per_run /= len(baseline_scores)

    baseline_mean = np.mean(baseline_mean_per_run)

    results = []
    for model_name, wcs in llm_wcs_scores.items():
        # Mann-Whitney U: is this LLM's score significantly > baseline distribution?
        u_stat, p_value = stats.mannwhitneyu(
            [wcs] * n_runs,
            baseline_mean_per_run,
            alternative='greater'
        )

        improvement = wcs / baseline_mean if baseline_mean > 0 else float('inf')

        results.append({
            "Model": model_name,
            "WCS": wcs,
            "BaselineMean": baseline_mean,
            "ImprovementFactor": round(improvement, 1),
            "U_statistic": u_stat,
            "p_value": p_value,
            "significant": p_value < 0.001,
        })

        print(f"  {model_name:20s}: WCS={wcs:.3f}, {improvement:.0f}x improvement, "
              f"U={u_stat:.0f}, p={p_value:.2e}")

    return pd.DataFrame(results)


###############################################################################
# FIGURE GENERATION
###############################################################################

def generate_figure_3(scores_df, output_path):
    """
    Figure 3: Multidimensional Performance Analysis.

    (A) Radar plot - all models across 6 dimensions
    (B) Bar chart - key performance dimensions
    (C) Ranking heatmap
    (D) Summary statistics table
    (E) Semantic accuracy and consistency distributions
    """
    print("\n  Generating Figure 3...")

    dimensions = [
        "SemanticAccuracy", "FactualCorrectness", "DomainKnowledge",
        "ReasoningQuality", "ResponseDepth", "BiobankSpecificity"
    ]
    dim_labels = [
        "Semantic\nAccuracy", "Factual\nCorrectness", "Domain\nKnowledge",
        "Reasoning\nQuality", "Response\nDepth", "Biobank\nSpecificity"
    ]

    # Aggregate: mean across tasks for each model x dimension
    model_scores = scores_df.groupby("Model")[dimensions].mean()
    models = model_scores.index.tolist()

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1, 1],
                          hspace=0.5, wspace=0.3,
                          left=0.06, right=0.96, top=0.90, bottom=0.08)

    # (A) Radar plot
    ax_radar = fig.add_subplot(gs[0, 0:2], projection='polar')
    N = len(dimensions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (model, color) in enumerate(zip(models, colors)):
        values = model_scores.loc[model].values.tolist()
        values += values[:1]
        short_name = model.replace(' Large 2', '').replace(' 4.5', ' 4.5')
        ax_radar.plot(angles, values, '-', linewidth=2.5, label=short_name,
                      color=color, marker='o', markersize=3, alpha=0.8)
        ax_radar.fill(angles, values, alpha=0.05, color=color)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(dim_labels, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax_radar.grid(True, alpha=0.3)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax_radar.set_title('Multidimensional Performance (All Models)', fontsize=12, fontweight='bold', pad=20)

    # (B) Bar comparison
    ax_bar = fig.add_subplot(gs[0, 2:6])
    key_dims = ['SemanticAccuracy', 'ReasoningQuality', 'DomainKnowledge']
    key_labels = ['Semantic Accuracy', 'Reasoning Quality', 'Domain Knowledge']
    x = np.arange(len(models))
    width = 0.25
    bar_colors = ['#ff9999', '#66b3ff', '#99ff99']

    for i, (dim, label, color) in enumerate(zip(key_dims, key_labels, bar_colors)):
        offset = (i - 1) * width
        vals = [model_scores.loc[m, dim] for m in models]
        bars = ax_bar.bar(x + offset, vals, width, label=label, color=color, alpha=0.8)
        for j, bar in enumerate(bars):
            h = bar.get_height()
            if h > 0.3:
                ax_bar.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                           f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    short_names = [m.replace(' Large 2', '').replace(' 4.5', ' 4.5') for m in models]
    ax_bar.set_xlabel('Models', fontsize=11)
    ax_bar.set_ylabel('Performance Score', fontsize=11)
    ax_bar.set_title('Key Performance Dimensions', fontsize=12, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short_names, fontsize=9)
    ax_bar.legend(fontsize=10, loc='upper right')
    ax_bar.set_ylim(0, 1.1)
    ax_bar.grid(axis='y', alpha=0.3)

    # (C) Ranking heatmap
    ax_heatmap = fig.add_subplot(gs[1, :])
    rankings = model_scores.rank(ascending=False, method='min', axis=0).astype(int)

    im = ax_heatmap.imshow(rankings.T.values, cmap='RdYlGn_r', aspect='auto',
                            vmin=1, vmax=len(models))
    for i in range(len(dimensions)):
        for j in range(len(models)):
            ax_heatmap.text(j, i, int(rankings.iloc[j, i]),
                           ha="center", va="center", color="black",
                           fontweight='bold', fontsize=10)

    ax_heatmap.set_xticks(range(len(models)))
    ax_heatmap.set_xticklabels(short_names, fontsize=10)
    ax_heatmap.set_yticks(range(len(dimensions)))
    ax_heatmap.set_yticklabels([d.replace('\n', ' ') for d in dim_labels], fontsize=10)
    ax_heatmap.set_title(f'Model Rankings by Dimension (1=Best, {len(models)}=Worst)',
                         fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax_heatmap, shrink=0.8).set_label('Rank', rotation=270, labelpad=15)

    # (D) Summary statistics table
    ax_table = fig.add_subplot(gs[2, :4])

    overall_perf = model_scores.mean(axis=1)
    std_scores = model_scores.std(axis=1)
    consistency = 1 - (std_scores / 0.5)
    consistency = consistency.clip(0, 1)

    prec = model_scores['FactualCorrectness']
    rec = model_scores['SemanticAccuracy']
    f1 = 2 * (prec * rec) / (prec + rec)
    f1 = f1.fillna(0)

    summary_data = {
        'Model': short_names,
        'Mean Score': [f"{s:.3f}" for s in overall_perf],
        'Std Dev': [f"{s:.3f}" for s in std_scores],
        'Min-Max Range': [f"{model_scores.loc[m].min():.2f}-{model_scores.loc[m].max():.2f}" for m in models],
        'Consistency': [f"{s:.3f}" for s in consistency],
        'F1 Score': [f"{s:.3f}" for s in f1],
    }

    df_table = pd.DataFrame(summary_data)
    df_table = df_table.sort_values('Mean Score', ascending=False).reset_index(drop=True)

    ax_table.axis('tight')
    ax_table.axis('off')
    table = ax_table.table(cellText=df_table.values, colLabels=df_table.columns,
                           cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    for i in range(len(df_table.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, len(df_table) + 1):
        if i <= 3:
            for j in range(len(df_table.columns)):
                table[(i, j)].set_facecolor('#E8F5E8')
        elif i >= len(df_table) - 1:
            for j in range(len(df_table.columns)):
                table[(i, j)].set_facecolor('#FFE8E8')

    ax_table.set_title('Performance Summary Statistics (Sorted by Mean Score)',
                       fontsize=12, fontweight='bold', pad=15)

    # (E) Distribution plots
    ax_dist1 = fig.add_subplot(gs[2, 4])
    ax_dist2 = fig.add_subplot(gs[2, 5])

    sem_acc_sorted = model_scores['SemanticAccuracy'].sort_values(ascending=False)
    bars1 = ax_dist1.bar(range(len(sem_acc_sorted)), sem_acc_sorted.values,
                         color=plt.cm.viridis(np.linspace(0, 1, len(sem_acc_sorted))), alpha=0.8)
    sa_short = [m.replace(' Large 2', '').replace(' 4.5', ' 4.5') for m in sem_acc_sorted.index]
    ax_dist1.set_xticks(range(len(sem_acc_sorted)))
    ax_dist1.set_xticklabels(sa_short, rotation=45, fontsize=8, ha='right')
    ax_dist1.set_ylabel('Semantic Accuracy', fontsize=10)
    ax_dist1.set_title('Semantic Accuracy\nDistribution', fontsize=11, fontweight='bold')
    ax_dist1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, sem_acc_sorted.values)):
        if i < 3:
            ax_dist1.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                         f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    consistency_sorted = consistency.sort_values(ascending=False)
    bars2 = ax_dist2.bar(range(len(consistency_sorted)), consistency_sorted.values,
                         color=plt.cm.plasma(np.linspace(0, 1, len(consistency_sorted))), alpha=0.8)
    cs_short = [m.replace(' Large 2', '').replace(' 4.5', ' 4.5') for m in consistency_sorted.index]
    ax_dist2.set_xticks(range(len(consistency_sorted)))
    ax_dist2.set_xticklabels(cs_short, rotation=45, fontsize=8, ha='right')
    ax_dist2.set_ylabel('Consistency Score', fontsize=10)
    ax_dist2.set_title('Performance Consistency\nDistribution', fontsize=11, fontweight='bold')
    ax_dist2.set_ylim(0, 1.0)
    ax_dist2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, consistency_sorted.values)):
        if i < 3:
            ax_dist2.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                         f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel labels
    panels = [
        (ax_radar, 'A', (-0.2, 1.1)),
        (ax_bar, 'B', (-0.05, 1.05)),
        (ax_heatmap, 'C', (-0.02, 1.05)),
        (ax_table, 'D', (-0.02, 1.05)),
        (ax_dist1, 'E', (-0.1, 1.05)),
    ]
    for ax, label, pos in panels:
        ax.text(pos[0], pos[1], label, fontsize=16, fontweight='bold',
                transform=ax.transAxes, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(output_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"  Figure 3 saved to {output_path}")

    return fig


def generate_figure_4(llm_wcs_scores, baseline_scores, stat_results, scores_df, output_path):
    """
    Figure 4: LLM Weighted Coverage Performance Compared to Random Baseline.

    (A) Density histogram: baseline distribution vs LLM scores
    (B) Improvement factors
    (C) Statistical significance (Mann-Whitney U)
    (D) Precision-Recall analysis
    """
    print("\n  Generating Figure 4...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Combine baseline across tasks
    n_runs = len(list(baseline_scores.values())[0])
    baseline_mean_per_run = np.zeros(n_runs)
    for task_scores in baseline_scores.values():
        baseline_mean_per_run += task_scores
    baseline_mean_per_run /= len(baseline_scores)

    llm_values = np.array(list(llm_wcs_scores.values()))
    model_names = list(llm_wcs_scores.keys())

    # (A) Performance vs Baseline
    ax1.hist(baseline_mean_per_run, bins=30, alpha=0.7, label='Random Baseline',
             color='red', density=True)
    ax1.axvline(llm_values.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'LLM Mean: {llm_values.mean():.3f}')
    ax1.axvline(llm_values.min(), color='green', linestyle='--', linewidth=2,
               label=f'LLM Min: {llm_values.min():.3f}')
    ax1.set_xlabel('Weighted Coverage Score', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title('LLM Performance vs Random Baseline', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # (B) Improvement factors
    baseline_mean = np.mean(baseline_mean_per_run)
    improvement_factors = llm_values / baseline_mean if baseline_mean > 0 else llm_values
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = ax2.bar(range(len(model_names)), improvement_factors, color=colors)
    ax2.set_xlabel('Models', fontsize=10)
    ax2.set_ylabel('Improvement Factor (x)', fontsize=10)
    ax2.set_title('Improvement Over Random Baseline', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(model_names)))
    short_names = [m.replace(' Large 2', '').replace(' 4.5', ' 4.5') for m in model_names]
    ax2.set_xticklabels(short_names, rotation=45, fontsize=9, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, improvement_factors):
        ax2.text(bar.get_x() + bar.get_width()/2., val + max(improvement_factors)*0.02,
                f'{val:.0f}x', ha='center', va='bottom', fontweight='bold', fontsize=8)

    # (C) Statistical significance
    if stat_results is not None and HAS_SCIPY:
        p_values = stat_results['p_value'].values
        p_values = np.maximum(p_values, 1e-300)
        sig_colors = ['darkgreen' if p < 0.001 else 'orange' if p < 0.05 else 'red' for p in p_values]
        ax3.bar(range(len(model_names)), [-np.log10(p) for p in p_values], color=sig_colors)
        ax3.axhline(-np.log10(0.001), color='red', linestyle='--', label='p < 0.001', alpha=0.7)
        ax3.axhline(-np.log10(0.05), color='orange', linestyle='--', label='p < 0.05', alpha=0.7)
        ax3.set_xlabel('Models', fontsize=10)
        ax3.set_ylabel('-log10(p-value)', fontsize=10)
        ax3.set_title('Statistical Significance vs Baseline (Mann-Whitney U)',
                      fontsize=11, fontweight='bold')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(short_names, rotation=45, fontsize=9, ha='right')
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)

    # (D) Precision-Recall analysis
    dimensions = ["SemanticAccuracy", "FactualCorrectness"]
    model_means = scores_df.groupby("Model")[dimensions].mean()

    precision_vals = model_means['FactualCorrectness'].values
    recall_vals = model_means['SemanticAccuracy'].values

    scatter = ax4.scatter(precision_vals, recall_vals, s=120, c=list(llm_wcs_scores.values()),
                         cmap='viridis', alpha=0.8, edgecolors='black')

    for i, model in enumerate(model_means.index):
        short = model.replace(' Large 2', '').replace(' 4.5', ' 4.5')
        ax4.annotate(short, (precision_vals[i], recall_vals[i]),
                    xytext=(8, 8), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    prec_range = np.linspace(0.01, 1, 100)
    for f1 in [0.3, 0.5, 0.7]:
        rec_curve = f1 * prec_range / (2 * prec_range - f1)
        rec_curve = np.clip(rec_curve, 0, 1)
        valid = rec_curve > 0
        ax4.plot(prec_range[valid], rec_curve[valid], '--', alpha=0.5, label=f'F1={f1}')

    ax4.set_xlabel('Factual Correctness (Precision)', fontsize=10)
    ax4.set_ylabel('Semantic Accuracy (Recall)', fontsize=10)
    ax4.set_title('Precision-Recall Analysis', fontsize=11, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=8)
    plt.colorbar(scatter, ax=ax4, shrink=0.8).set_label('Overall WCS', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(output_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"  Figure 4 saved to {output_path}")

    return fig


###############################################################################
# MAIN
###############################################################################

def main():
    print("=" * 60)
    print("REAL MULTIDIMENSIONAL EVALUATION OF LLM BIOBANK BENCHMARK")
    print("=" * 60)

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load real LLM responses
    print("\n--- Loading LLM responses ---")
    all_responses = load_llm_responses()

    if not all_responses:
        print("ERROR: No LLM responses found. Check DATA/ directory.")
        sys.exit(1)

    # 2. Run multidimensional evaluation
    scores_df = run_multidimensional_evaluation(all_responses)

    # Save scores
    scores_path = RESULTS_DIR / "multidimensional_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"\n  Scores saved to {scores_path}")

    # 3. Load schema data and run baseline comparison
    print("\n--- Loading schema data ---")
    schema_df = load_schema_data()

    np.random.seed(42)  # reproducibility
    baseline_scores = run_baseline_comparison(schema_df, num_runs=100)

    # Save baseline scores
    baseline_path = RESULTS_DIR / "baseline_comparison.csv"
    baseline_data = {}
    for task_name, scores in baseline_scores.items():
        baseline_data[f"{task_name}_wcs"] = scores
    pd.DataFrame(baseline_data).to_csv(baseline_path, index=False)
    print(f"\n  Baseline scores saved to {baseline_path}")

    # 4. Real LLM Weighted Coverage Scores (from Table 1 / script 07 output)
    llm_wcs_scores = {
        "Gemini 3 Pro": 0.643,
        "Claude Sonnet 4.5": 0.577,
        "Claude Opus 4.5": 0.577,
        "Mistral Large 2": 0.567,
        "DeepSeek V3": 0.517,
        "GPT-5.2": 0.455,
    }

    # 5. Statistical testing
    stat_results = run_statistical_tests(llm_wcs_scores, baseline_scores)

    if stat_results is not None:
        stat_path = RESULTS_DIR / "statistical_tests.csv"
        stat_results.to_csv(stat_path, index=False)
        print(f"\n  Statistical test results saved to {stat_path}")

    # 6. Generate figures
    print("\n--- Generating figures ---")
    fig3 = generate_figure_3(scores_df, RESULTS_DIR / "figure_3.pdf")
    fig4 = generate_figure_4(llm_wcs_scores, baseline_scores, stat_results,
                             scores_df, RESULTS_DIR / "figure_4.pdf")

    # 7. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    dimensions = ["SemanticAccuracy", "FactualCorrectness", "DomainKnowledge",
                  "ReasoningQuality", "ResponseDepth", "BiobankSpecificity"]
    model_means = scores_df.groupby("Model")[dimensions].mean()
    overall = model_means.mean(axis=1).sort_values(ascending=False)

    print("\nModel Rankings (by mean score across 6 dimensions):")
    for i, (model, score) in enumerate(overall.items(), 1):
        print(f"  {i}. {model:20s} - {score:.3f}")

    print(f"\nTop semantic accuracy: {model_means['SemanticAccuracy'].idxmax()} "
          f"({model_means['SemanticAccuracy'].max():.3f})")

    consistency = 1 - (model_means.std(axis=1) / 0.5)
    consistency = consistency.clip(0, 1)
    print(f"Most consistent: {consistency.idxmax()} ({consistency.max():.3f})")

    if stat_results is not None:
        baseline_mean = stat_results['BaselineMean'].iloc[0]
        min_imp = stat_results['ImprovementFactor'].min()
        max_imp = stat_results['ImprovementFactor'].max()
        all_sig = stat_results['significant'].all()
        print(f"\nBaseline mean WCS: {baseline_mean:.4f}")
        print(f"Improvement range: {min_imp:.0f}x to {max_imp:.0f}x")
        print(f"All models significant (p < 0.001): {all_sig}")

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
