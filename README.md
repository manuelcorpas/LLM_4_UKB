# LLM_4_UKB: Large Language Models for Mining UK Biobank Insights

This repository hosts the code and data for our paper:

**"Benchmarking Large Language Models for Extracting Biobank-Derived Insights into Health and Disease"**

We benchmark multiple frontier LLMs to assess how well they retrieve key insights from **UK Biobank**–related literature, including top keywords, most-cited papers, prolific authors, and leading applicant institutions.

## Latest Update: January 2026

We have updated the benchmark to evaluate the latest frontier models:

| Model | Provider | Overall Score |
|-------|----------|---------------|
| **Gemini 3 Pro** | Google | 0.643 |
| **Claude Sonnet 4** | Anthropic | 0.577 |
| **Claude Opus 4.5** | Anthropic | 0.577 |
| **Mistral Large** | Mistral AI | 0.567 |
| **DeepSeek V3** | DeepSeek | 0.517 |
| **GPT-5.2** | OpenAI | 0.455 |

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

We evaluated six frontier Large Language Models (LLMs)—**Gemini 3 Pro**, **Claude Opus 4.5**, **Claude Sonnet 4**, **GPT-5.2**, **Mistral Large**, and **DeepSeek V3**—on multiple tasks derived from **UK Biobank** data:

1. **Top Keywords** in publications (frequency analysis)  
2. **Most Cited Papers** based on citation counts from the UK Biobank’s metadata  
3. **Prolific Authors** (most publications)  
4. **Leading Applicant Institutions** by number of applications  

We measure each model’s **Coverage Score** (breadth of matched terms) and **Weighted Coverage Score** (emphasizing high-frequency or high-impact terms). This helps evaluate how effectively each LLM retrieves domain-specific content.

---

## Directory Structure

Below is a high-level overview of the repository:
```
LLM_4_UKB/
├── DATA/
│   ├── 01-most-common-keyword.csv
│   ├── 02-subject-most-cited.csv
│   ├── 03-most-prolific-authors.csv
│   ├── 04-top-applicant-institutions.csv
│   ├── bart_temp.csv
│   ├── combined_bart_keywords.csv
│   ├── combined_keywords_bart.csv
├── PYTHON/
│   ├── 00-collect-model-responses.py      # NEW: Collect LLM responses via API
│   ├── 00-00-ukb-schema-publication-reports.py
│   ├── 00-condense-UKB-abstracts.py
│   ├── 01-benchmark_llm_keywords.py
│   ├── 02-benchmark_llm_papers.py
│   ├── 03-benchmark_llm_authors.py
│   ├── 04-benchmark_llm_institutions.py
│   ├── 05-benchmark-summary-results.py
│   ├── 06-00-biobank-eval.py              # Generate publication figures
│   ├── 06-query-ukb-bart.py
│   └── ARCHIVE/
├── README.md
├── RESULTS/
│   ├── 00-ukb-pub-schema.txt
│   ├── BART/
│   ├── BENCHMARK/
│   ├── REPORTS/
├── requirements.txt
└── ...
```
**Notable Folders**  
- **DATA/**  
  Contains CSV files with LLM outputs, metadata, or partial extracts used for benchmarking.  
- **PYTHON/**  
  Core scripts for data preprocessing, benchmarking, and handling queries.  
- **RESULTS/**  
  Holds generated figures, benchmark outputs, and intermediate or final analysis files.  
- **requirements.txt**  
  Lists Python dependencies (e.g., `sentence-transformers`, `matplotlib`, `numpy`, etc.).

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/manuelcorpas/LLM_4_UKB.git
   cd LLM_4_UKB
   ```

2. **(Optional) Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have **Python 3.8+**.

---

## How to Run

### Scripts Overview

0. **`00-collect-model-responses.py`** *(NEW)*
   Collects responses from frontier LLMs via their APIs. Requires API keys in `.env` file.
   - Supports: Anthropic, OpenAI, Google, Mistral, DeepSeek
   - *Output:* Populates `DATA/01-04` CSV files with model responses

1. **`00-00-ukb-schema-publication-reports.py` / `00-condense-UKB-abstracts.py`**
   Preprocessing or cleaning of raw UK Biobank data.

2. **`01-benchmark_llm_keywords.py`**  
   Evaluates coverage of top keywords in UK Biobank abstracts.  
   - *Input:* `DATA/01-most-common-keyword.csv`  
   - *Output:* `RESULTS/BENCHMARK/keywords_results.csv`

3. **`02-benchmark_llm_papers.py`**  
   Assesses retrieval of the most-cited papers.  
   - *Input:* `DATA/02-subject-most-cited.csv`  
   - *Output:* `RESULTS/BENCHMARK/top-cited-paper-coverage.csv`

4. **`03-benchmark_llm_authors.py`**  
   Checks coverage of top authors by publication count.  
   - *Input:* `DATA/03-most-prolific-authors.csv`  
   - *Output:* `RESULTS/BENCHMARK/authors_coverage.csv`

5. **`04-benchmark_llm_institutions.py`**  
   Evaluates coverage of leading applicant institutions.  
   - *Input:* `DATA/04-top-applicant-institutions.csv`  
   - *Output:* `RESULTS/BENCHMARK/institutions_coverage.csv`

6. **`05-benchmark-summary-results.py`**
   Aggregates or summarizes cross-script outputs for an overall performance ranking.
   - *Output:* `RESULTS/BENCHMARK/final_overall_ranking.csv`

7. **`06-00-biobank-eval.py`**
   Generates publication-quality figures (Figure 3 and Figure 4) for the manuscript.
   - *Output:* `figure_3_clean_fixed.png`, `baseline_comparison_clean.png`

8. **`06-query-ukb-bart.py`**
   Demonstrates querying a BART model or merging partial results. Not always needed for the main pipeline.

**Example Workflow**:
```bash
# Step 1: Collect LLM responses (requires API keys in .env)
python PYTHON/00-collect-model-responses.py

# Step 2: Run benchmark evaluation scripts
python PYTHON/01-benchmark_llm_keywords.py
python PYTHON/02-benchmark_llm_papers.py
python PYTHON/03-benchmark_llm_authors.py
python PYTHON/04-benchmark_llm_institutions.py

# Step 3: Summarize results
python PYTHON/05-benchmark-summary-results.py

# Step 4: Generate publication figures
python PYTHON/06-00-biobank-eval.py
```

**API Keys Setup** (for `00-collect-model-responses.py`):
Create a `.env` file in the project root with:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

---

## Usage Tips

- The `RESULTS/opt-125m/` folder may contain large model checkpoints. If unnecessary, consider removing or ignoring them to keep the repository size low.
- Adjust the `SIMILARITY_THRESHOLD` in each benchmark script (default is `0.20`) to control how strictly embeddings match key terms.
- For additional or legacy scripts (e.g., training, advanced queries), see `PYTHON/ARCHIVE/`.

---

## License & Citation

- **License**: MIT  
- **Citation**: If you use or extend this code, please cite our paper:

```bibtex
@misc{CorpasIacoangeli2025LLMUKB,
  title    = {Large Language Models for Mining Biobank-Derived Insights into Health and Disease},
  author   = {Corpas, Manuel and Iacoangeli, Alfredo},
  year     = {2025},
  archivePrefix = {bioRxiv},
  eprint        = {PAPER_DOI}
}
```

---

## Contact

- **Maintainer**: [Manuel Corpas](mailto:M.Corpas@westminster.ac.uk)  
- **Co-Author**: [Alfredo Iacoangeli](mailto:alfredo.iacoangeli@kcl.ac.uk)

Feel free to open an issue or pull request if you encounter bugs or want to contribute improvements.

## Repo policy
- This repository tracks **code and configuration only**
- `RESULTS/` contains local artefacts and is intentionally not versioned
- Do not create multiple clones of this repo inside iCloud

