#!/usr/bin/env python3
"""
Collect LLM Responses for UK Biobank Benchmark
===============================================
This script queries current frontier models (January 2026) with the 4 benchmark
questions and saves responses to CSV files for evaluation.

Models tested (8 frontier models):
- Claude Opus 4.5, Claude Sonnet 4.5 (Anthropic)
- GPT-5.2 (OpenAI)
- Gemini 3 Pro (Google)
- Grok 4.1 (xAI)
- Mistral Large (Mistral)
- DeepSeek V3 (DeepSeek)
- Kimi K2 (Moonshot)

Usage:
    python 02-collect-model-responses.py

Environment variables required:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GOOGLE_API_KEY
    MISTRAL_API_KEY
    XAI_API_KEY
    DEEPSEEK_API_KEY
    MOONSHOT_API_KEY

Author: Manuel Corpas
Date: January 2026
"""

import os
import csv
import time
from datetime import datetime
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded API keys from {env_path}")
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables only.")
    print("      Install with: pip install python-dotenv")

# API clients - install with: pip install anthropic openai google-generativeai mistralai
try:
    import anthropic
except ImportError:
    anthropic = None
    print("Warning: anthropic package not installed. Run: pip install anthropic")

try:
    import openai
except ImportError:
    openai = None
    print("Warning: openai package not installed. Run: pip install openai")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None
    print("Warning: mistralai not installed. Run: pip install mistralai")


# ============================================================================
# BENCHMARK QUESTIONS
# ============================================================================
# IMPORTANT: The file names must match what the evaluation scripts expect!
# - 03-benchmark-keywords.py reads DATA/01-most-common-keyword.csv
# - 04-benchmark-papers.py reads DATA/02-subject-most-cited.csv (expects PAPER subjects)
# - 05-benchmark-authors.py reads DATA/03-most-prolific-authors.csv
# - 06-benchmark-institutions.py reads DATA/04-top-applicant-institutions.csv (expects INSTITUTIONS)

BENCHMARK_QUESTIONS = {
    "01-keywords": {
        "question": "What is the Subject of the Most Commonly Occurring Keywords in UK Biobank Papers?",
        "output_file": "DATA/01-most-common-keyword.csv",
        "description": "Evaluates keyword recognition"
    },
    "02-papers": {
        "question": "What is the Subject of the Top Most Cited Papers Relating to the UK Biobank?",
        "output_file": "DATA/02-subject-most-cited.csv",
        "description": "Evaluates recognition of highly-cited paper topics"
    },
    "03-authors": {
        "question": "What Are the Top 20 Most Prolific Authors Publishing on the UK Biobank?",
        "output_file": "DATA/03-most-prolific-authors.csv",
        "description": "Evaluates author name recognition"
    },
    "04-institutions": {
        "question": "What Are the Top 10 Leading Institutions in Terms of Number of Applications to UK Biobank?",
        "output_file": "DATA/04-top-applicant-institutions.csv",
        "description": "Evaluates institution name recognition"
    }
}

# ============================================================================
# MODEL CONFIGURATIONS (January 2026 Frontier Models)
# ============================================================================
# Updated based on current model availability (January 2026)
# Verify model_id values match your API access level.

MODELS = {
    # Anthropic - https://docs.anthropic.com/claude/docs/models-overview
    "Claude Opus 4.5": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",  # Flagship reasoning model
        "url": "https://claude.ai"
    },
    "Claude Sonnet 4": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",  # High-end reasoning
        "url": "https://claude.ai"
    },
    # OpenAI - https://platform.openai.com/docs/models
    "GPT-5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2",  # Latest GPT-5 series (Dec 2025)
        "url": "https://chatgpt.com"
    },
    # Google - https://ai.google.dev/models
    "Gemini 3 Pro": {
        "provider": "google",
        "model_id": "gemini-3-pro-preview",  # Latest Gemini 3
        "url": "https://gemini.google.com"
    },
    # Mistral - https://docs.mistral.ai/getting-started/models/
    "Mistral Large": {
        "provider": "mistral",
        "model_id": "mistral-large-latest",
        "url": "https://chat.mistral.ai"
    },
    # DeepSeek - https://platform.deepseek.com/api-docs
    "DeepSeek V3": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "url": "https://chat.deepseek.com"
    }
}


# ============================================================================
# API QUERY FUNCTIONS
# ============================================================================

def query_anthropic(model_id: str, question: str) -> str:
    """Query Anthropic Claude models."""
    if anthropic is None:
        return "[ERROR: anthropic package not installed]"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR: ANTHROPIC_API_KEY not set]"

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model_id,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def query_openai(model_id: str, question: str) -> str:
    """Query OpenAI models."""
    if openai is None:
        return "[ERROR: openai package not installed]"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "[ERROR: OPENAI_API_KEY not set]"

    client = openai.OpenAI(api_key=api_key)

    try:
        # GPT-5+ models don't support max_tokens parameter
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def query_google(model_id: str, question: str) -> str:
    """Query Google Gemini models."""
    if genai is None:
        return "[ERROR: google-generativeai package not installed]"

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "[ERROR: GOOGLE_API_KEY not set]"

    genai.configure(api_key=api_key)

    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def query_mistral(model_id: str, question: str) -> str:
    """Query Mistral models."""
    if Mistral is None:
        return "[ERROR: mistralai package not installed]"

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return "[ERROR: MISTRAL_API_KEY not set]"

    client = Mistral(api_key=api_key)

    try:
        response = client.chat.complete(
            model=model_id,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def query_deepseek(model_id: str, question: str) -> str:
    """Query DeepSeek models (uses OpenAI-compatible API)."""
    if openai is None:
        return "[ERROR: openai package not installed]"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "[ERROR: DEEPSEEK_API_KEY not set]"

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": question}
            ],
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def query_xai(model_id: str, question: str) -> str:
    """Query xAI Grok models (uses OpenAI-compatible API)."""
    if openai is None:
        return "[ERROR: openai package not installed]"

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        return "[ERROR: XAI_API_KEY not set]"

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": question}
            ],
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def query_moonshot(model_id: str, question: str) -> str:
    """Query Moonshot Kimi models (uses OpenAI-compatible API)."""
    if openai is None:
        return "[ERROR: openai package not installed]"

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        return "[ERROR: MOONSHOT_API_KEY not set]"

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1"
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": question}
            ],
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def query_model(model_name: str, question: str) -> str:
    """Route query to appropriate provider."""
    config = MODELS.get(model_name)
    if not config:
        return f"[ERROR: Unknown model {model_name}]"

    provider = config["provider"]
    model_id = config["model_id"]

    print(f"  Querying {model_name} ({model_id})...", end=" ", flush=True)
    start = time.time()

    if provider == "anthropic":
        response = query_anthropic(model_id, question)
    elif provider == "openai":
        response = query_openai(model_id, question)
    elif provider == "google":
        response = query_google(model_id, question)
    elif provider == "mistral":
        response = query_mistral(model_id, question)
    elif provider == "deepseek":
        response = query_deepseek(model_id, question)
    elif provider == "xai":
        response = query_xai(model_id, question)
    elif provider == "moonshot":
        response = query_moonshot(model_id, question)
    else:
        response = f"[ERROR: Unknown provider {provider}]"

    elapsed = time.time() - start

    if response.startswith("[ERROR"):
        print(f"FAILED ({elapsed:.1f}s)")
    else:
        print(f"OK ({elapsed:.1f}s, {len(response)} chars)")

    return response


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def collect_responses_for_question(question_key: str, question_config: dict):
    """Collect responses from all models for a single question."""
    question = question_config["question"]
    output_file = question_config["output_file"]

    print(f"\n{'='*70}")
    print(f"Question: {question}")
    print(f"Output: {output_file}")
    print('='*70)

    responses = []

    for model_name, model_config in MODELS.items():
        response = query_model(model_name, question)
        responses.append({
            "Model": model_name,
            "URL": model_config["url"],
            "Response": response.replace("\n", " ").replace(",", ";")  # Clean for CSV
        })
        time.sleep(1)  # Rate limiting

    # Write to CSV
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # Write question as header (matching original format)
        f.write(f"{question},,\n")

        # Write responses
        for resp in responses:
            # Escape quotes in response
            clean_response = resp["Response"].replace('"', '""')
            f.write(f'{resp["Model"]},{resp["URL"]},"{clean_response}"\n')

    print(f"\nSaved {len(responses)} responses to {output_file}")
    return responses


def main():
    """Main entry point."""
    print("="*70)
    print("UK Biobank LLM Benchmark - Response Collection")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {len(MODELS)}")
    print("="*70)

    # Check API keys
    print("\nAPI Key Status:")
    for key_name in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                     "MISTRAL_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY",
                     "MOONSHOT_API_KEY"]:
        status = "SET" if os.environ.get(key_name) else "NOT SET"
        print(f"  {key_name}: {status}")

    print("\n" + "-"*70)
    print("Starting data collection...")
    print("-"*70)

    all_responses = {}

    for q_key, q_config in BENCHMARK_QUESTIONS.items():
        all_responses[q_key] = collect_responses_for_question(q_key, q_config)

    print("\n" + "="*70)
    print("COLLECTION COMPLETE")
    print("="*70)
    print("\nOutput files created:")
    for q_key, q_config in BENCHMARK_QUESTIONS.items():
        print(f"  - {q_config['output_file']}")

    print("\nNext steps:")
    print("  1. Review the responses for any errors")
    print("  2. Run the evaluation pipeline:")
    print("     python 03-benchmark-keywords.py")
    print("     python 04-benchmark-papers.py")
    print("     python 05-benchmark-authors.py")
    print("     python 06-benchmark-institutions.py")
    print("     python 07-benchmark-summary.py")
    print("     python 08-multidimensional-eval.py")


if __name__ == "__main__":
    main()
