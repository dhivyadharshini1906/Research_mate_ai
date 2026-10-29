from pathlib import Path
import re
try:
    import pymupdf as fitz
except ImportError:
    import fitz
def extract_pages(pdf_path):
    pages = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text: pages.append({"page_number": idx + 1, "text": clean_text(text)})
    return pages
def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_pages(pages, chunk_words=220, overlap_words=40):
    if chunk_words < 30: raise ValueError("chunk_words must be at least 30")
    if overlap_words < 0 or overlap_words >= chunk_words: raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")
    chunks, idx = [], 0
    for page in pages:
        words, start = page["text"].split(), 0
        while start < len(words):
            piece = " ".join(words[start:start + chunk_words]).strip()
            if piece:
                chunks.append({"page_number": int(page["page_number"]), "chunk_index": idx, "text": piece}); idx += 1
            if start + chunk_words >= len(words): break
            start += chunk_words - overlap_words
    return chunks
def guess_title(pages, fallback):
    if not pages: return Path(fallback).stem.replace("_", " ").strip() or "Untitled paper"
    lines = [x.strip() for x in pages[0]["text"].splitlines() if x.strip()]
    for line in lines[:8]:
        if 8 <= len(line) <= 220: return line
    return Path(fallback).stem.replace("_", " ").strip() or "Untitled paper"
