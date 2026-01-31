#!/usr/bin/env python3
"""
Generate Figure 2 Panels for LLM-UKB Benchmark Paper
=====================================================
Creates publication-quality figures for the 4 benchmark tasks plus overall ranking.

Panels:
- 2a: Keywords benchmark
- 2b: Most cited papers benchmark
- 2c: Authors benchmark
- 2d: Institutions benchmark
- 2e: Overall ranking

Author: Manuel Corpas
Date: January 2026
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File paths
RESULTS_DIR = Path("RESULTS/BENCHMARK")
OUTPUT_DIR = Path("RESULTS/FIGURES_2026")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme for consistency
COLORS = {
    'coverage': '#4ECDC4',      # Teal
    'weighted': '#FF6B6B',      # Coral
    'overall': '#45B7D1',       # Sky blue
    'bars': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
}

# Model display order (by overall score)
MODEL_ORDER = [
    'Gemini 3 Pro',
    'Claude Sonnet 4',
    'Claude Opus 4.5',
    'Mistral Large',
    'DeepSeek V3',
    'GPT-5.2'
]


def load_csv(filepath):
    """Load benchmark results from CSV."""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['Model'].strip()
            data[model] = {
                'coverage': float(row['CoverageScore']),
                'weighted': float(row['WeightedCoverageScore'])
            }
    return data


def create_benchmark_panel(data, title, output_file, ylabel="Score"):
    """Create a single benchmark panel with coverage and weighted scores."""
    # Order models
    models = [m for m in MODEL_ORDER if m in data]
    coverage = [data[m]['coverage'] for m in models]
    weighted = [data[m]['weighted'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, coverage, width, label='Coverage Score',
                   color=COLORS['coverage'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, weighted, width, label='Weighted Coverage',
                   color=COLORS['weighted'], edgecolor='white', linewidth=0.5)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.15)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_file = str(output_file).replace('.png', '.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file} + PDF")


def create_overall_panel(output_file):
    """Create panel 2e: Overall ranking."""
    # Load final ranking
    data = {}
    with open(RESULTS_DIR / "final_overall_ranking.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['Model'].strip()
            data[model] = {
                'overall_weighted': float(row['OverallWeighted']),
                'overall_coverage': float(row['OverallCoverage']),
                'keywords': float(row['KeywordsWeighted']),
                'papers': float(row['SubjectsWeighted']),
                'authors': float(row['AuthorsWeighted']),
                'institutions': float(row['InstitutionsWeighted'])
            }

    # Order models by overall score
    models = [m for m in MODEL_ORDER if m in data]
    overall = [data[m]['overall_weighted'] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use single consistent color
    bars = ax.bar(models, overall, color=COLORS['overall'], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Overall Weighted Coverage', fontsize=12)
    ax.set_title('Overall LLM Performance Ranking (January 2026)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_file = str(output_file).replace('.png', '.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file} + PDF")


def create_combined_figure():
    """Create a combined 2x3 figure with all panels."""
    # Load all data
    keywords_data = load_csv(RESULTS_DIR / "keywords_results.csv")
    papers_data = load_csv(RESULTS_DIR / "top-cited-paper-coverage.csv")
    authors_data = load_csv(RESULTS_DIR / "authors_coverage.csv")
    institutions_data = load_csv(RESULTS_DIR / "institutions_coverage.csv")

    # Load overall ranking
    overall_data = {}
    with open(RESULTS_DIR / "final_overall_ranking.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['Model'].strip()
            overall_data[model] = float(row['OverallWeighted'])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    datasets = [
        (keywords_data, 'A. Keywords', axes[0, 0]),
        (papers_data, 'B. Most Cited Papers', axes[0, 1]),
        (authors_data, 'C. Prolific Authors', axes[0, 2]),
        (institutions_data, 'D. Institutions', axes[1, 0]),
    ]

    for data, title, ax in datasets:
        models = [m for m in MODEL_ORDER if m in data]
        coverage = [data[m]['coverage'] for m in models]
        weighted = [data[m]['weighted'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, coverage, width, label='Coverage',
                       color=COLORS['coverage'], edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, weighted, width, label='Weighted',
                       color=COLORS['weighted'], edgecolor='white', linewidth=0.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=0, ha='center', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        if title == 'A. Keywords':
            ax.legend(loc='upper right', fontsize=8)

    # Panel E: Overall ranking
    ax = axes[1, 1]
    models = [m for m in MODEL_ORDER if m in overall_data]
    overall = [overall_data[m] for m in models]

    bars = ax.bar(models, overall, color=COLORS['overall'], edgecolor='white', linewidth=0.5)
    ax.set_title('E. Overall Ranking', fontsize=12, fontweight='bold')
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=0, ha='center', fontsize=8)
    ax.set_ylim(0, 0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    # Remove empty subplot
    axes[1, 2].axis('off')

    plt.suptitle('Figure 2: LLM Benchmark Results (January 2026)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig2_combined.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / "Fig2_combined.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'Fig2_combined.png'} + PDF")


def main():
    print("="*60)
    print("Generating Figure 2 Panels - January 2026 Update")
    print("="*60)

    # Load data
    keywords_data = load_csv(RESULTS_DIR / "keywords_results.csv")
    papers_data = load_csv(RESULTS_DIR / "top-cited-paper-coverage.csv")
    authors_data = load_csv(RESULTS_DIR / "authors_coverage.csv")
    institutions_data = load_csv(RESULTS_DIR / "institutions_coverage.csv")

    print("\nGenerating individual panels...")

    # Panel 2a: Keywords
    create_benchmark_panel(
        keywords_data,
        "Figure 2A: Keyword Recognition Benchmark",
        OUTPUT_DIR / "Fig2a_keywords.png"
    )

    # Panel 2b: Most cited papers
    create_benchmark_panel(
        papers_data,
        "Figure 2B: Most Cited Papers Benchmark",
        OUTPUT_DIR / "Fig2b_papers.png"
    )

    # Panel 2c: Authors
    create_benchmark_panel(
        authors_data,
        "Figure 2C: Prolific Authors Benchmark",
        OUTPUT_DIR / "Fig2c_authors.png"
    )

    # Panel 2d: Institutions
    create_benchmark_panel(
        institutions_data,
        "Figure 2D: Institutions Benchmark",
        OUTPUT_DIR / "Fig2d_institutions.png"
    )

    # Panel 2e: Overall
    create_overall_panel(OUTPUT_DIR / "Fig2e_overall.png")

    # Combined figure
    print("\nGenerating combined figure...")
    create_combined_figure()

    print("\n" + "="*60)
    print("Figure 2 generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
