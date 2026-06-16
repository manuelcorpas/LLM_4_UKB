#!/usr/bin/env python3
"""
Generate Figure 1 for PLOS Computational Biology submission.
UK Biobank Schema Data (Schema 19 publications + Schema 27 applications).

Reads repo-relative DATA/schema_19.txt and DATA/schema_27.txt and writes
Fig1.{png,tif,pdf} into RESULTS/EVALUATION/. Prints the corpus counts so the
manuscript figures can be reconciled against the data (publications, of which
how many carry an abstract, and applications).

PLOS specs: TIF, 300 DPI, max 2250x2625 px, RGB.
"""

from pathlib import Path
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_19 = PROJECT_ROOT / "DATA" / "schema_19.txt"
SCHEMA_27 = PROJECT_ROOT / "DATA" / "schema_27.txt"
OUTPUT_DIR = PROJECT_ROOT / "RESULTS" / "EVALUATION"

CORAL = "#FF7F7F"
LIGHT_BLUE = "#87CEEB"
DARK_BLUE = "#4682B4"
LIGHT_GREEN = "#90EE90"


def load_data():
    pub_df = pd.read_csv(SCHEMA_19, sep="\t", quoting=csv.QUOTE_NONE,
                         dtype=str, on_bad_lines="skip", engine="python")
    pub_df["cite_total"] = pd.to_numeric(pub_df.get("cite_total"), errors="coerce")
    pub_df["year_pub"] = pd.to_numeric(pub_df.get("year_pub"), errors="coerce")
    app_df = pd.read_csv(SCHEMA_27, sep="\t", quoting=csv.QUOTE_NONE,
                         dtype=str, on_bad_lines="skip", engine="python")
    return pub_df, app_df


def create_figure1():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    pub_df, app_df = load_data()

    n_pub = len(pub_df)
    n_abs = pub_df["abstract"].dropna().astype(str).str.strip().ne("").sum() if "abstract" in pub_df.columns else 0
    n_app = len(app_df)
    print(f"  Schema 19: {n_pub} publications, of which {n_abs} carry an abstract")
    print(f"  Schema 27: {n_app} applications")

    year_counts = pub_df["year_pub"].dropna().astype(int).value_counts().sort_index()
    year_counts = year_counts[(year_counts.index >= 2013) & (year_counts.index <= 2025)]
    print(f"  Panel A (publications by year 2013-2025) sums to: {int(year_counts.sum())}")

    keywords = pub_df["keywords"].dropna().str.split("|").explode().str.strip()
    top_keywords = keywords[keywords != ""].value_counts().head(15)
    most_cited = pub_df.dropna(subset=["cite_total"]).nlargest(10, "cite_total")[["title", "journal", "cite_total"]]
    authors = pub_df["authors"].dropna().str.split("|").explode().str.strip()
    top_authors = authors[authors != ""].value_counts().head(15)
    top_institutions = app_df["institution"].dropna().value_counts().head(10)

    fig = plt.figure(figsize=(7.5, 8.75))
    ax_a = fig.add_axes([0.08, 0.72, 0.28, 0.22])
    ax_b = fig.add_axes([0.55, 0.72, 0.43, 0.22])
    ax_c = fig.add_axes([0.28, 0.40, 0.22, 0.26])
    ax_d = fig.add_axes([0.68, 0.40, 0.30, 0.26])
    ax_e = fig.add_axes([0.08, 0.12, 0.90, 0.22])

    # (A) Publications by Year
    years = year_counts.index.tolist()
    counts = year_counts.values.tolist()
    ax_a.bar(years, counts, color=CORAL, edgecolor="#CC6666", linewidth=0.5)
    ax_a.set_xlabel("Year", fontsize=8)
    ax_a.set_ylabel("Number of Publications", fontsize=8)
    ax_a.set_title("(A) Publications by Year", fontsize=9, fontweight="bold")
    ax_a.tick_params(axis="both", labelsize=6)
    ax_a.set_xticks(years[::2])

    # (B) Top Keywords
    kw_names = top_keywords.index.tolist()
    kw_names_short = [k[:25] + "..." if len(k) > 25 else k for k in kw_names]
    ax_b.barh(range(len(kw_names)), top_keywords.values.tolist(), color=LIGHT_BLUE,
              edgecolor="#5CACEE", linewidth=0.5)
    ax_b.set_yticks(range(len(kw_names)))
    ax_b.set_yticklabels(kw_names_short, fontsize=5.5)
    ax_b.set_xlabel("Count", fontsize=8)
    ax_b.set_title("(B) Top 15 Keywords", fontsize=9, fontweight="bold")
    ax_b.invert_yaxis()
    ax_b.tick_params(axis="x", labelsize=6)

    # (C) Most Cited Papers
    labels = [str(r["title"])[:38] + "..." if len(str(r["title"])) > 38 else str(r["title"])
              for _, r in most_cited.iterrows()]
    ax_c.barh(range(len(labels)), most_cited["cite_total"].tolist(), color=DARK_BLUE,
              edgecolor="#36648B", linewidth=0.5)
    ax_c.set_yticks(range(len(labels)))
    ax_c.set_yticklabels(labels, fontsize=5)
    ax_c.set_xlabel("Citations", fontsize=8)
    ax_c.set_title("(C) Most Cited Papers", fontsize=9, fontweight="bold")
    ax_c.invert_yaxis()
    ax_c.tick_params(axis="x", labelsize=6)

    # (D) Top Authors
    ax_d.barh(range(len(top_authors)), top_authors.values.tolist(), color=LIGHT_GREEN,
              edgecolor="#66CD00", linewidth=0.5)
    ax_d.set_yticks(range(len(top_authors)))
    ax_d.set_yticklabels(top_authors.index.tolist(), fontsize=6)
    ax_d.set_xlabel("Publications", fontsize=8)
    ax_d.set_title("(D) Top 15 Authors", fontsize=9, fontweight="bold")
    ax_d.invert_yaxis()
    ax_d.tick_params(axis="x", labelsize=6)

    # (E) Top Institutions
    ax_e.bar(range(len(top_institutions)), top_institutions.values.tolist(), color=DARK_BLUE,
             edgecolor="#36648B", linewidth=0.5)
    ax_e.set_xticks(range(len(top_institutions)))
    ax_e.set_xticklabels(top_institutions.index.tolist(), rotation=35, ha="right", fontsize=6)
    ax_e.set_ylabel("Applications", fontsize=8)
    ax_e.set_title("(E) Top 10 Applicant Institutions", fontsize=9, fontweight="bold")
    ax_e.tick_params(axis="y", labelsize=6)

    png_path = OUTPUT_DIR / "Fig1.png"
    fig.savefig(png_path, dpi=300, facecolor="white", edgecolor="none")
    fig.savefig(OUTPUT_DIR / "Fig1.pdf", facecolor="white", edgecolor="none")

    tif_path = OUTPUT_DIR / "Fig1.tif"
    with Image.open(png_path) as img:
        rgb_img = img.convert("RGB")
        w, h = rgb_img.size
        if w > 2250 or h > 2625:
            ratio = min(2250 / w, 2625 / h)
            rgb_img = rgb_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        rgb_img.save(tif_path, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    plt.close()
    with Image.open(tif_path) as img:
        print(f"  Fig1.tif: {img.size[0]} x {img.size[1]} px, {img.mode} "
              f"({'PASS' if img.size[0] <= 2250 and img.size[1] <= 2625 else 'FAIL'})")
    print(f"  Wrote Fig1.png/tif/pdf to {OUTPUT_DIR}")


if __name__ == "__main__":
    create_figure1()
