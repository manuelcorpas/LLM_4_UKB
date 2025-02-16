# condense_ukbb_abstracts_fix.py
#
# This script automatically fixes common XML parsing errors (like unescaped ampersands)
# before parsing the file, then extracts and summarizes the UK Biobank abstracts.

import re
import xml.etree.ElementTree as ET
from transformers import pipeline

########################################
# CONFIGURATION
########################################

# Path to your local XML file containing UK Biobank abstracts
XML_FILE_PATH = "DATA/publication_cleaned.xml"  # Replace with your path

# Summarization model name on Hugging Face
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# Maximum length (tokens) of the summary output
MAX_SUMMARY_LENGTH = 150

# Where to store intermediate summaries
INTERMEDIATE_SUMMARIES_FILE = "RESULTS/BART/summaries_intermediate.txt"

# Where to store the final single summary
GOLD_REFERENCE_FILE = "RESULTS/BART/gold_reference.txt"


########################################
# HELPER FUNCTIONS
########################################

def fix_xml_characters(xml_data: str) -> str:
    """
    Attempt to fix common XML issues by escaping stray ampersands that are not already valid entities.
    This is a naive approach but often resolves 'not well-formed (invalid token)' parse errors.
    """
    # Regex: Replace any '&' that is NOT followed by 'amp;', 'lt;', 'gt;', 'quot;', 'apos;', or '#digits;'
    # with '&amp;'.
    pattern = r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;)"
    fixed_data = re.sub(pattern, "&amp;", xml_data)

    # You can add more replacements here if necessary.
    return fixed_data


def clean_html(raw_html: str) -> str:
    """Remove HTML tags (<p>, etc.) and decode typical HTML entities."""
    # remove HTML tags via regex
    cleaned = re.sub(r"<[^>]+>", " ", raw_html)
    # replace numeric entities
    cleaned = cleaned.replace("&#183;", ".")
    cleaned = cleaned.replace("&#215;", "×")
    cleaned = cleaned.replace("&#8722;", "-")
    cleaned = cleaned.replace("&#177;", "+/-")
    cleaned = cleaned.replace("&#215;", "x")
    # strip extra whitespace
    cleaned = re.sub(r"\\s+", " ", cleaned)
    return cleaned.strip()


########################################
# MAIN SCRIPT
########################################

def main():
    # 1. Read the raw XML file
    with open(XML_FILE_PATH, "r", encoding="utf-8") as f:
        xml_data = f.read()

    # 2. Attempt to fix common invalid tokens (especially stray ampersands)
    fixed_xml = fix_xml_characters(xml_data)

    # 3. Parse the corrected XML data
    try:
        # parse from string
        root = ET.fromstring(fixed_xml)
    except ET.ParseError as e:
        print("[ERROR] Could not parse even after fix. Details:", e)
        return

    # Build an ElementTree
    tree = ET.ElementTree(root)

    # 4. Prepare summarization pipeline
    summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL)

    # 5. Extract abstracts
    abstracts = []
    for pub in root.findall("publication"):
        raw_abstract = pub.get("abstract", "")
        if raw_abstract:
            # Clean the abstract text
            cleaned = clean_html(raw_abstract)
            if cleaned.strip():
                abstracts.append(cleaned)

    # 6. Summarize each abstract individually
    summaries = []
    for i, abstract in enumerate(abstracts, 1):
        try:
            summary_out = summarizer(
                abstract,
                max_length=MAX_SUMMARY_LENGTH,
                min_length=30,
                do_sample=False
            )
            summary_text = summary_out[0]["summary_text"]
            summaries.append(summary_text)
            print(f"Summarized abstract {i}/{len(abstracts)}")
        except Exception as e:
            print(f"[WARNING] Summarization failed for abstract {i}: {e}")
            summaries.append("[ERROR] Could not summarize.")

    # 7. Save intermediate summaries
    with open(INTERMEDIATE_SUMMARIES_FILE, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(s + "\n\n")

    # 8. Produce a single final summary by summarizing all summaries
    combined_summaries = "\n".join(summaries)
    try:
        final_summary_out = summarizer(
            combined_summaries,
            max_length=MAX_SUMMARY_LENGTH,
            min_length=50,
            do_sample=False
        )
        final_summary = final_summary_out[0]["summary_text"]
    except Exception as e:
        print(f"[WARNING] Could not summarize combined summaries: {e}")
        final_summary = "".join(summaries)

    # 9. Save the final gold reference
    with open(GOLD_REFERENCE_FILE, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print("\n====================================================")
    print("Finished!")
    print(f"- Intermediate summaries saved to: {INTERMEDIATE_SUMMARIES_FILE}")
    print(f"- Final single gold reference summary saved to: {GOLD_REFERENCE_FILE}")


if __name__ == "__main__":
    main()

