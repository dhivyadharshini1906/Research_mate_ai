# ResearchMate AI — Research Paper Intelligence Platform

A portfolio web application for exploring research papers, asking evidence-grounded questions, comparing candidate objectives/methods/limitations, and keeping notes.

## Features
- Upload multiple text-based PDFs and preserve page numbers.
- Overlapping text chunks and semantic embeddings.
- Semantic search with Sentence Transformers + FAISS.
- Optional LLM answers with passage-number citations.
- Retrieval-only fallback without an API key.
- Multi-paper comparison using a transparent heuristic baseline.
- SQLite by default; optional PostgreSQL.
- Delete papers, save notes, and export comparison CSV.
- Unit tests for text processing.

## Architecture
Streamlit UI → PyMuPDF page extraction → overlapping chunks → Sentence Transformers embeddings → FAISS retrieval → optional LLM answer generation. SQLAlchemy stores paper metadata, chunks, and notes.

## Setup (Windows / VS Code)
Python 3.11 recommended. Open a terminal in this folder:
```powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use Command Prompt: `.venv\\Scripts\\activate.bat`

Set an optional API key in the current PowerShell session to enable generated answers:
```powershell
$env:OPENAI_API_KEY="your-key-here"
```
Never commit or share your API key. Without it, the app still retrieves and displays evidence passages, but does not synthesize an LLM answer.

Run:
```powershell
python -m streamlit run app/main.py
```
Or if `streamlit` is in your PATH:
```powershell
streamlit run app/main.py
```
The first run may download the embedding model and take a few minutes.


## Optional PostgreSQL
SQLite works by default. To use PostgreSQL, install a driver and set `DATABASE_URL`:
```powershell
pip install "psycopg[binary]"
$env:DATABASE_URL="postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/researchmate"
streamlit run app/main.py
```
Create the `researchmate` database first. Do not commit your password.

## Usage
1. Open Library and upload text-based PDFs.
2. Wait for indexing.
3. Open Ask papers and enter a question.
4. Inspect citations and the retrieved evidence passages.
5. Use Compare to select at least two papers.
6. Save research notes and export comparison CSV as needed.

## Evaluation plan
Create 30–50 labelled questions with expected facts and supporting paper/page passages, including questions whose answers are absent. Measure Recall@k, citation support, answer correctness by human review, abstention quality, latency, and API cost. Do not claim high accuracy before measuring it.

## Limitations
- Scanned PDFs need OCR, not included.
- Title detection and comparison extraction are heuristic.
- Without an API key, answers are retrieval-only.
- LLM output can be wrong; verify citations.
- No authentication or multi-user isolation. Do not publicly deploy with sensitive documents until access controls are added.
