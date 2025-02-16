# LLM_4_UKB: Large Language Models for Mining UK Biobank Insights

This repository hosts the code and data for our paper:  
**“Large Language Models for Mining Biobank-Derived Insights into Health and Disease.”**

We benchmark multiple LLMs to assess how well they retrieve key insights from **UK Biobank**–related literature, including top keywords, most-cited papers, prolific authors, and leading applicant institutions.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Directory Structure](#directory-structure)  
3. [Installation & Setup](#installation--setup)  
4. [How to Run](#how-to-run)  
   - [Scripts Overview](#scripts-overview)  
5. [Usage Tips](#usage-tips)  
6. [License & Citation](#license--citation)  
7. [Contact](#contact)

---

## Project Overview

We evaluated several well-known Large Language Models (LLMs)—including GPT, Claude, Gemini, Mistral, Llama, and DeepSeek—on multiple tasks derived from **UK Biobank** data:

1. **Top Keywords** in publications (frequency analysis)  
2. **Most Cited Papers** based on citation counts from the UK Biobank’s metadata  
3. **Prolific Authors** (most publications)  
4. **Leading Applicant Institutions** by number of applications  

We measure each model’s **Coverage Score** (breadth of matched terms) and **Weighted Coverage Score** (emphasizing high-frequency or high-impact terms). This approach highlights how effectively each LLM retrieves domain-specific content.

---

## Directory Structure

Below is a high-level overview of the repository:

LLM_4_UKB/ ├── DATA/ │ ├── 01-most-common-keyword.csv │ ├── 02-subject-most-cited.csv │ ├── 03-most-prolific-authors.csv │ ├── 04-top-applicant-institutions.csv │ ├── ARCHIVE/ │ ├── bart_temp.csv │ ├── combined_bart_keywords.csv │ ├── combined_keywords_bart.csv │ ├── publication_cleaned.xml │ └── schema_19.txt ├── PYTHON/ │ ├── 00-00-ukb-schema-publication-reports.py │ ├── 00-condense-UKB-abstracts.py │ ├── 01-benchmark_llm_keywords.py │ ├── 02-benchmark_llm_papers.py │ ├── 03-benchmark_llm_authors.py │ ├── 04-benchmark_llm_institutions.py │ ├── 05-benchmark-summary-results.py │ ├── 06-query-ukb-bart.py │ └── ARCHIVE/ ├── README.md ├── RESULTS/ │ ├── 00-ukb-pub-schema.txt │ ├── BART/ │ ├── BENCHMARK/ │ ├── REPORTS/ │ └── opt-125m/ ├── requirements.txt └── ...

markdown
Copy
Edit

### Notable Folders

- **DATA/**  
  Contains CSV files of LLM outputs, metadata, and partial extracts.  
- **PYTHON/**  
  Core scripts for data preprocessing, benchmarking, and query handling.  
- **RESULTS/**  
  Holds generated figures, CSVs, and final or intermediate outputs.  
- **requirements.txt**  
  Python dependencies list.

---

## Installation & Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/manuelcorpas/LLM_4_UKB.git
   cd LLM_4_UKB
Create a virtual environment (optional):

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Ensure Python 3.8+ is available.

How to Run
Scripts Overview
00-00-ukb-schema-publication-reports.py / 00-condense-UKB-abstracts.py
Preprocessing or cleaning of raw UK Biobank data.

01-benchmark_llm_keywords.py
Evaluates coverage of top keywords in UK Biobank abstracts.

Input: DATA/01-most-common-keyword.csv
Output: RESULTS/BENCHMARK/keywords_results.csv
02-benchmark_llm_papers.py
Assesses retrieval of the most-cited papers.

Input: DATA/02-subject-most-cited.csv
Output: RESULTS/BENCHMARK/top-cited-paper-coverage.csv
03-benchmark_llm_authors.py
Checks coverage of top authors by publication count.

Input: DATA/03-most-prolific-authors.csv
Output: RESULTS/BENCHMARK/authors_coverage.csv
04-benchmark_llm_institutions.py
Evaluates coverage of leading applicant institutions.

Input: DATA/04-top-applicant-institutions.csv
Output: RESULTS/BENCHMARK/institutions_coverage.csv
05-benchmark-summary-results.py
Aggregates results from scripts 1–4 for an overall performance score.

06-query-ukb-bart.py
Example of querying a BART model or merging results (not always essential for the main pipeline).

Example Workflow
bash
Copy
Edit
# (Optional) Preprocess data
python PYTHON/00-condense-UKB-abstracts.py

# Run benchmark scripts
python PYTHON/01-benchmark_llm_keywords.py
python PYTHON/02-benchmark_llm_papers.py
python PYTHON/03-benchmark_llm_authors.py
python PYTHON/04-benchmark_llm_institutions.py

# Summarize results
python PYTHON/05-benchmark-summary-results.py
Usage Tips
The RESULTS/opt-125m/ folder may contain large model checkpoints. If these are not required, consider removing or archiving them to keep the repo size manageable.
Adjust your SIMILARITY_THRESHOLD in the benchmark scripts (default is 0.20) to tweak matching strictness.
For advanced usage (e.g., training, additional queries), see scripts in PYTHON/ARCHIVE/.
License & Citation
License: [Choose your open-source license, e.g., MIT or Apache 2.0.]
Citation: If you use or extend this code, please cite our paper:
bibtex
Copy
Edit
@misc{CorpasIacoangeli2025LLMUKB,
  title    = {Large Language Models for Mining Biobank-Derived Insights into Health and Disease},
  author   = {Corpas, Manuel and Iacoangeli, Alfredo, ...},
  year     = {2025},
  archivePrefix = {bioRxiv},
  eprint        = {YOUR_PAPER_DOI_OR_ID}
}
Contact
Maintainer: Manuel Corpas
Co-Author: Alfredo Iacoangeli
