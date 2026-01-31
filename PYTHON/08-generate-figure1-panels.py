#!/usr/bin/env python3
"""
Generate Figure 1 for PLOS Computational Biology submission
UK Biobank Schema Data - Manual positioning for clean layout

PLOS specs: TIF format, 300 DPI, max 2250x2625 px, RGB mode
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
import os
from PIL import Image

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

SCHEMA_19 = '/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/UKBiobank/LLM_4_UKB/DATA/schema_19.txt'
SCHEMA_27 = '/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/UKBiobank/LLM_4_UKB/DATA/schema_27.txt'
OUTPUT_DIR = '/Users/superintelligent/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/01-LLMS-UKBB/REVISION_2/FIGURES'

CORAL = '#FF7F7F'
LIGHT_BLUE = '#87CEEB'
DARK_BLUE = '#4682B4'
LIGHT_GREEN = '#90EE90'


def load_data():
    pub_df = pd.read_csv(SCHEMA_19, sep='\t', quoting=csv.QUOTE_NONE,
                         dtype=str, on_bad_lines='skip', engine="python")
    pub_df['cite_total'] = pd.to_numeric(pub_df.get('cite_total'), errors='coerce')
    pub_df['year_pub'] = pd.to_numeric(pub_df.get('year_pub'), errors='coerce')

    app_df = pd.read_csv(SCHEMA_27, sep='\t', quoting=csv.QUOTE_NONE,
                         dtype=str, on_bad_lines='skip', engine="python")
    return pub_df, app_df


def create_figure1():
    print("Loading data...")
    pub_df, app_df = load_data()

    # Extract data
    year_counts = pub_df['year_pub'].dropna().astype(int).value_counts().sort_index()
    year_counts = year_counts[(year_counts.index >= 2013) & (year_counts.index <= 2025)]

    keywords = pub_df['keywords'].dropna().str.split('|').explode().str.strip()
    top_keywords = keywords[keywords != ''].value_counts().head(15)

    most_cited = pub_df.dropna(subset=['cite_total']).nlargest(10, 'cite_total')[['title', 'journal', 'cite_total']]

    authors = pub_df['authors'].dropna().str.split('|').explode().str.strip()
    top_authors = authors[authors != ''].value_counts().head(15)

    top_institutions = app_df['institution'].dropna().value_counts().head(10)

    # Create figure with manual axes positioning [left, bottom, width, height]
    fig = plt.figure(figsize=(7.5, 8.75))

    # Row 1: Panels A and B (top) - B moved right to avoid label overlap
    ax_a = fig.add_axes([0.08, 0.72, 0.28, 0.22])
    ax_b = fig.add_axes([0.55, 0.72, 0.43, 0.22])

    # Row 2: Panels C and D (middle) - C needs left margin for titles, D needs left margin for names
    ax_c = fig.add_axes([0.28, 0.40, 0.22, 0.26])
    ax_d = fig.add_axes([0.68, 0.40, 0.30, 0.26])

    # Row 3: Panel E (bottom, full width) - more bottom margin for rotated labels
    ax_e = fig.add_axes([0.08, 0.12, 0.90, 0.22])

    # === Panel A: Publications by Year ===
    years = year_counts.index.tolist()
    counts = year_counts.values.tolist()
    ax_a.bar(years, counts, color=CORAL, edgecolor='#CC6666', linewidth=0.5)
    ax_a.set_xlabel('Year', fontsize=8)
    ax_a.set_ylabel('Number of Publications', fontsize=8)
    ax_a.set_title('(A) Publications by Year', fontsize=9, fontweight='bold')
    ax_a.tick_params(axis='both', labelsize=6)
    ax_a.set_xticks(years[::2])

    # === Panel B: Top Keywords ===
    kw_names = top_keywords.index.tolist()
    # Truncate long keyword names
    kw_names_short = [k[:25] + '...' if len(k) > 25 else k for k in kw_names]
    kw_counts = top_keywords.values.tolist()
    ax_b.barh(range(len(kw_names)), kw_counts, color=LIGHT_BLUE, edgecolor='#5CACEE', linewidth=0.5)
    ax_b.set_yticks(range(len(kw_names)))
    ax_b.set_yticklabels(kw_names_short, fontsize=5.5)
    ax_b.set_xlabel('Count', fontsize=8)
    ax_b.set_title('(B) Top 15 Keywords', fontsize=9, fontweight='bold')
    ax_b.invert_yaxis()
    ax_b.tick_params(axis='x', labelsize=6)

    # === Panel C: Most Cited Papers ===
    labels = [str(row['title'])[:38] + '...' if len(str(row['title'])) > 38 else str(row['title'])
              for _, row in most_cited.iterrows()]
    citations = most_cited['cite_total'].tolist()
    ax_c.barh(range(len(labels)), citations, color=DARK_BLUE, edgecolor='#36648B', linewidth=0.5)
    ax_c.set_yticks(range(len(labels)))
    ax_c.set_yticklabels(labels, fontsize=5)
    ax_c.set_xlabel('Citations', fontsize=8)
    ax_c.set_title('(C) Most Cited Papers', fontsize=9, fontweight='bold')
    ax_c.invert_yaxis()
    ax_c.tick_params(axis='x', labelsize=6)

    # === Panel D: Top Authors ===
    author_names = top_authors.index.tolist()
    author_counts = top_authors.values.tolist()
    ax_d.barh(range(len(author_names)), author_counts, color=LIGHT_GREEN, edgecolor='#66CD00', linewidth=0.5)
    ax_d.set_yticks(range(len(author_names)))
    ax_d.set_yticklabels(author_names, fontsize=6)
    ax_d.set_xlabel('Publications', fontsize=8)
    ax_d.set_title('(D) Top 15 Authors', fontsize=9, fontweight='bold')
    ax_d.invert_yaxis()
    ax_d.tick_params(axis='x', labelsize=6)

    # === Panel E: Top Institutions ===
    inst_names = top_institutions.index.tolist()
    inst_counts = top_institutions.values.tolist()
    ax_e.bar(range(len(inst_names)), inst_counts, color=DARK_BLUE, edgecolor='#36648B', linewidth=0.5)
    ax_e.set_xticks(range(len(inst_names)))
    ax_e.set_xticklabels(inst_names, rotation=35, ha='right', fontsize=6)
    ax_e.set_ylabel('Applications', fontsize=8)
    ax_e.set_title('(E) Top 10 Applicant Institutions', fontsize=9, fontweight='bold')
    ax_e.tick_params(axis='y', labelsize=6)

    # Save PNG
    png_path = os.path.join(OUTPUT_DIR, 'Fig1_new.png')
    fig.savefig(png_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved PNG: {png_path}")

    # Convert to RGB TIF with size check
    tif_path = os.path.join(OUTPUT_DIR, 'Fig1.tif')
    with Image.open(png_path) as img:
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            rgb_img = background
        else:
            rgb_img = img.convert('RGB')

        w, h = rgb_img.size
        if w > 2250 or h > 2625:
            ratio = min(2250/w, 2625/h)
            new_size = (int(w * ratio), int(h * ratio))
            rgb_img = rgb_img.resize(new_size, Image.LANCZOS)
            print(f"Resized: {w}x{h} -> {new_size[0]}x{new_size[1]}")

        rgb_img.save(tif_path, format='TIFF', compression='tiff_lzw', dpi=(300, 300))
    print(f"Saved TIF: {tif_path}")

    plt.close()

    with Image.open(tif_path) as img:
        print(f"\nFinal: {img.size[0]} x {img.size[1]} px, {img.mode}")
        print("PASS" if img.size[0] <= 2250 and img.size[1] <= 2625 else "FAIL")


if __name__ == "__main__":
    create_figure1()
