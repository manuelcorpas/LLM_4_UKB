# UK Biobank Publication Analysis

Pipeline for analyzing UK Biobank publication trends using NLP models.

## Structure
- `00-process_data.py`: Process XML publications
- `01-train.py`: Train the model
- `02-query_model.py`: Run specific queries
- `run_pipeline.py`: Run full pipeline with different models

## Setup
1. Create virtual environment:
```python
python -m venv ukb_env
source ukb_env/bin/activate  # On Windows: ukb_env\Scripts\activate
pip install -r requirements.txt
