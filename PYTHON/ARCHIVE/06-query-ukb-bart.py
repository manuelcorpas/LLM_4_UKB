#!/usr/bin/env python3

"""
Self-contained script to:
1. Read large text from RESULTS/BART/summaries_intermediate.txt
2. Summarize it in smaller chunks using BART
3. (Optional) Summarize the chunk summaries again
4. Perform question-answering on the final summary
"""

import os
from transformers import pipeline

# Path to the (potentially large) text file you want to summarize with BART
BART_INPUT_FILE = "RESULTS/BART/summaries_intermediate.txt"

# The specific question you want answered after summarization
QUESTION = "What is the Subject of the Top Most Cited Papers Relating to the UK Biobank?"

# BART model for summarization
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# A dedicated QA model, fine-tuned on SQuAD
QA_MODEL = "distilbert-base-cased-distilled-squad"

def chunk_text(text, max_chars=2000):
    """
    Splits `text` into chunks of length `max_chars` to avoid
    exceeding model token limits or running out of memory.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks

def main():
    # 1. Read the raw text
    if not os.path.exists(BART_INPUT_FILE):
        print(f"[ERROR] File not found: {BART_INPUT_FILE}")
        return

    with open(BART_INPUT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        print("[ERROR] The file is empty or unreadable.")
        return

    # 2. Split the raw text into manageable chunks
    text_chunks = chunk_text(raw_text, max_chars=2000)
    if not text_chunks:
        print("[ERROR] No valid text chunks found.")
        return

    # 3. Summarize each chunk with BART
    #    We force CPU (device=-1) to avoid MPS issues, but you can switch to GPU if available.
    print(f"[INFO] Loading summarization model '{SUMMARIZATION_MODEL}'...")
    bart_summarizer = pipeline(
        "summarization",
        model=SUMMARIZATION_MODEL,
        device=-1  # -1 -> CPU. Change to 0 for GPU with CUDA, if available.
    )

    chunk_summaries = []
    for i, chunk in enumerate(text_chunks, 1):
        print(f"[INFO] Summarizing chunk {i}/{len(text_chunks)} (length={len(chunk)} chars)...")
        summary = bart_summarizer(
            chunk,
            max_length=256,  # Adjust based on desired summary length
            min_length=50,   # Adjust as needed
            do_sample=False
        )
        chunk_summaries.append(summary[0]["summary_text"])

    # Combine chunk summaries into one text
    combined_summary = "\n".join(chunk_summaries)

    # 4. Optionally re-summarize the combined summary to get a more cohesive final summary
    print("[INFO] Re-summarizing combined text...")
    final_summary = bart_summarizer(
        combined_summary,
        max_length=256,
        min_length=50,
        do_sample=False
    )[0]["summary_text"]

    print("\n================= FINAL BART SUMMARY =================")
    print(final_summary)
    print("======================================================\n")

    # 5. Perform QA on this final summary using a QA-specialized model
    print(f"[INFO] Loading QA model '{QA_MODEL}'...")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=-1)

    print(f"[INFO] Asking: '{QUESTION}'")
    try:
        response = qa_pipeline(question=QUESTION, context=final_summary)
        answer = response["answer"]
    except Exception as e:
        print(f"[ERROR] QA failed: {e}")
        return

    # Print the final QA result
    print("===================== QA RESULT ======================")
    print(f"Question: {QUESTION}")
    print(f"Answer: {answer}")
    print("======================================================\n")

if __name__ == "__main__":
    main()

