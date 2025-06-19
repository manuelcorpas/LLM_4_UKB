#!/usr/bin/env python3
"""
00-build-schema-and-keywords-relaxed.py

Extracts publication data from UK Biobank–related XML export (custom <publication> schema).
Builds:
- ukb_schema.pkl — cleaned list of text entries (title + abstract)
- ukb_keyword_frequencies.pkl — keyword frequency dictionary
- ukb_schema.csv — CSV table for inspection/reporting
"""

import os
import re
import pickle
import csv
from collections import Counter
from bs4 import BeautifulSoup
import pandas as pd

# ---- CONFIG ----
INPUT_XML = "DATA/publication_cleaned.xml"
SCHEMA_PKL = "DATA/ukb_schema.pkl"
FREQ_PKL = "DATA/ukb_keyword_frequencies.pkl"
CSV_EXPORT = "DATA/ukb_schema.csv"

# ---- HELPERS ----
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def tokenize(text):
    return re.findall(r"\b\w{4,}\b", text)

# ---- MAIN FUNCTION ----
def main():
    if not os.path.exists(INPUT_XML):
        print(f"❌ Input file not found: {INPUT_XML}")
        return

    with open(INPUT_XML, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")

    records = soup.find_all("publication")
    entries = []
    texts = []

    for pub in records:
        pub_id = pub.get("pub_id")
        title_raw = pub.get("title")
        abstract_raw = pub.get("abstract")

        if not title_raw and not abstract_raw:
            continue

        title = BeautifulSoup(title_raw or "", "html.parser").get_text()
        abstract = BeautifulSoup(abstract_raw or "", "html.parser").get_text()
        text = clean_text(f"{title} {abstract}")
        if not text.strip():
            continue

        entries.append({"pub_id": pub_id, "title": title.strip(), "abstract": abstract.strip(), "text": text})
        texts.append(text)

    # Save .pkl
    with open(SCHEMA_PKL, "wb") as f:
        pickle.dump(texts, f)
    print(f"✅ Saved {len(texts)} text entries to {SCHEMA_PKL}")

    # Save .csv
    df = pd.DataFrame(entries)
    df.to_csv(CSV_EXPORT, index=False)
    print(f"📄 Exported to CSV: {CSV_EXPORT}")

    # Save keyword frequencies
    tokens = [token for t in texts for token in tokenize(t)]
    freq = dict(Counter(tokens).most_common(1000))
    with open(FREQ_PKL, "wb") as f:
        pickle.dump(freq, f)
    print(f"✅ Saved keyword frequency dictionary to {FREQ_PKL}")

if __name__ == "__main__":
    main()
