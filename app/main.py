import os
import re
import hashlib
from pathlib import Path
import streamlit as st
import pandas as pd
from sqlalchemy import select, func

from app.config import UPLOAD_DIR, MAX_UPLOAD_MB, DATABASE_URL, LLM_MODEL, BASE_DIR
from app.db import init_db, SessionLocal, Paper, Chunk, Note
from app.processing import extract_pages, chunk_pages, guess_title
from app.retrieval import SemanticIndex
from app.llm import generate_grounded_answer, get_active_api_key
from app.analysis_tools import compare_papers

# Configure Streamlit page
st.set_page_config(
    page_title="ResearchMate AI — Research Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Load CSS
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_index():
    return SemanticIndex()


def get_db():
    return SessionLocal()


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:180] or "paper.pdf"


def read_papers(db):
    return db.scalars(select(Paper).order_by(Paper.uploaded_at.desc())).all()


def upload_papers(uploaded_files):
    index = get_index()
    results = []
    with get_db() as db:
        for file in uploaded_files:
            if not file.name.lower().endswith(".pdf"):
                results.append((file.name, "Skipped", "Only PDF files are supported."))
                continue

            data = file.getvalue()
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                results.append((file.name, "Skipped", f"File exceeds {MAX_UPLOAD_MB} MB limit."))
                continue

            filename = hashlib.sha256(data).hexdigest()[:12] + "_" + safe_filename(file.name)
            if db.scalar(select(Paper).where(Paper.filename == filename)):
                results.append((file.name, "Duplicate", "Already uploaded and indexed in your library."))
                continue

            dest = UPLOAD_DIR / filename
            dest.write_bytes(data)

            try:
                pages = extract_pages(dest)
                if not pages:
                    dest.unlink(missing_ok=True)
                    results.append((file.name, "Failed", "No extractable text. Scanned PDFs require OCR."))
                    continue

                paper = Paper(
                    title=guess_title(pages, file.name),
                    filename=filename,
                    path=str(dest),
                    page_count=max(p["page_number"] for p in pages)
                )
                db.add(paper)
                db.flush()

                chunks = chunk_pages(pages)
                for c in chunks:
                    db.add(Chunk(
                        paper_id=paper.id,
                        page_number=c["page_number"],
                        chunk_index=c["chunk_index"],
                        text=c["text"]
                    ))
                db.commit()

                index.add_chunks([{"paper_id": paper.id, "title": paper.title, **c} for c in chunks])
                results.append((file.name, "Indexed", f"Processed {len(pages)} pages & {len(chunks)} semantic chunks."))
            except Exception as exc:
                db.rollback()
                dest.unlink(missing_ok=True)
                results.append((file.name, "Error", str(exc)))

    return results


# Fetch data counts
with get_db() as db:
    all_papers = read_papers(db)
    total_papers_count = len(all_papers)
    total_pages_count = sum(p.page_count for p in all_papers)
    total_chunks_count = db.scalar(select(func.count(Chunk.id))) or 0
    total_notes_count = db.scalar(select(func.count(Note.id))) or 0

# Session state initialization for API Key & Model
if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.getenv("OPENAI_API_KEY", "")
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = os.getenv("LLM_MODEL", LLM_MODEL)

idx_engine = get_index()
active_key = st.session_state["api_key"]
active_model = st.session_state["selected_model"]

# ==========================================
# SIDEBAR NAVIGATION & SYSTEM HEALTH
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:1.5rem; padding-bottom:1rem; border-bottom:1px solid rgba(255,255,255,0.08);">
        <div style="background:linear-gradient(135deg, #6366f1, #8b5cf6); width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center; box-shadow:0 4px 14px rgba(99,102,241,0.4); font-size:1.4rem;">⚡</div>
        <div>
            <div style="font-size:1.18rem; font-weight:800; color:#f8fafc; letter-spacing:-0.02em;">ResearchMate AI</div>
            <div style="font-size:0.75rem; color:#818cf8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Paper Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### ⚙️ System Diagnostics")
    
    # Engine status pill
    engine_label = idx_engine.backend_name if hasattr(idx_engine, "backend_name") else "Semantic Engine"
    st.markdown(f"""
    <div style="background:rgba(19,27,46,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:0.75rem; margin-bottom:1rem; font-size:0.82rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="color:#94a3b8;">Database:</span>
            <span style="color:#6ee7b7; font-weight:600;">● Active (SQLite)</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="color:#94a3b8;">Index Engine:</span>
            <span style="color:#a5b4fc; font-weight:600;">{engine_label}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#94a3b8;">Synthesis:</span>
            <span style="color:{'#6ee7b7' if active_key else '#fcd34d'}; font-weight:600;">{'● LLM Ready' if active_key else '○ Evidence Mode'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # LLM Settings Card
    st.markdown("##### 🔑 LLM Engine Settings")
    with st.container():
        if active_key:
            masked = active_key[:7] + "••••" + active_key[-4:] if len(active_key) > 12 else "••••••••"
            status_html = f"""
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem; padding:0.45rem 0.65rem; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25); border-radius:8px; font-size:0.78rem;">
                <span style="color:#6ee7b7; font-weight:600;">● Key Active</span>
                <code style="color:#a5b4fc; background:transparent; font-size:0.75rem;">{masked}</code>
            </div>
            """
        else:
            status_html = """
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem; padding:0.45rem 0.65rem; background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.25); border-radius:8px; font-size:0.78rem;">
                <span style="color:#fcd34d; font-weight:600;">○ Retrieval-Only</span>
                <span style="color:#94a3b8; font-size:0.75rem;">No API key</span>
            </div>
            """
        st.markdown(status_html, unsafe_allow_html=True)

        with st.form("llm_api_key_form", clear_on_submit=False):
            input_key = st.text_input(
                "OpenAI API Key",
                value=active_key,
                type="password",
                placeholder="Paste sk-proj-... here",
                help="Stored securely in your active browser session."
            )
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            model_index = model_options.index(active_model) if active_model in model_options else 0
            chosen_model = st.selectbox(
                "Model",
                model_options,
                index=model_index,
                help="Select generative model."
            )
            
            c_save, c_clear = st.columns([3, 2])
            with c_save:
                submitted = st.form_submit_button("💾 Save Key", type="primary", use_container_width=True)
            with c_clear:
                cleared = st.form_submit_button("✖ Clear", use_container_width=True)

            if submitted:
                cleaned_key = input_key.strip()
                st.session_state["api_key"] = cleaned_key
                st.session_state["selected_model"] = chosen_model
                os.environ["OPENAI_API_KEY"] = cleaned_key
                os.environ["LLM_MODEL"] = chosen_model
                if cleaned_key:
                    st.toast("✅ OpenAI Key & Model successfully saved!", icon="🔑")
                else:
                    st.toast("Switched to Retrieval-Only Mode.", icon="ℹ️")
                st.rerun()

            if cleared:
                st.session_state["api_key"] = ""
                os.environ["OPENAI_API_KEY"] = ""
                st.toast("API Key cleared. Switched to Evidence Mode.", icon="🗑️")
                st.rerun()


    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748b; line-height:1.5;">
        <strong>Storage Directory:</strong><br>
        Local filesystem isolation<br>
        <span style="color:#94a3b8;">Chunk size: 220 words / 40 overlap</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# HERO HEADER & LIVE STATS
# ==========================================
st.markdown("""
<div class="rm-hero-badge">✦ Grounded Research Intelligence</div>
<h1 class="rm-hero-title">Explore, Synthesize & Compare Research Papers</h1>
<p class="rm-hero-subtitle">
    Evidence-grounded answers with precise page-level citations, multi-paper comparative synthesis, and persistent research notes.
</p>
""", unsafe_allow_html=True)

# Dynamic Stat Cards
st.markdown(f"""
<div class="rm-stat-grid">
    <div class="rm-stat-card">
        <div class="rm-stat-label">📚 Library Papers</div>
        <div class="rm-stat-value">{total_papers_count}</div>
        <div class="rm-stat-sub">Indexed documents</div>
    </div>
    <div class="rm-stat-card">
        <div class="rm-stat-label">📄 Extracted Pages</div>
        <div class="rm-stat-value">{total_pages_count}</div>
        <div class="rm-stat-sub">With preserved numbering</div>
    </div>
    <div class="rm-stat-card">
        <div class="rm-stat-label">🧩 Semantic Chunks</div>
        <div class="rm-stat-value">{total_chunks_count}</div>
        <div class="rm-stat-sub">Dense vector segments</div>
    </div>
    <div class="rm-stat-card">
        <div class="rm-stat-label">📝 Research Notes</div>
        <div class="rm-stat-value">{total_notes_count}</div>
        <div class="rm-stat-sub">Grounded observations</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_lib, tab_ask, tab_compare, tab_notes, tab_about = st.tabs([
    "📂 Library & Upload",
    "💬 Ask & Synthesize",
    "⚖️ Cross-Paper Compare",
    "📝 Research Notes",
    "ℹ️ Architecture & About"
])

# ==========================================
# TAB 1: LIBRARY & INGESTION
# ==========================================
with tab_lib:
    # Dedicated Upload Card
    st.markdown(f"""
    <div class="rm-glass-card" style="padding:1.5rem; margin-bottom:1.5rem;">
        <div class="rm-card-header" style="margin-bottom:1rem; padding-bottom:0.75rem;">
            <div>
                <h3 style="margin:0; font-size:1.25rem; font-weight:700; color:#f8fafc;">📤 Ingest Research Papers</h3>
                <p style="margin:0.25rem 0 0 0; font-size:0.85rem; color:#94a3b8;">
                    Upload single or multiple research papers (PDF) to extract pages, create dense semantic embeddings, and index into your local library.
                </p>
            </div>
            <span class="rm-badge rm-badge-indigo">PDF Format • Max {MAX_UPLOAD_MB} MB each</span>
        </div>
    """, unsafe_allow_html=True)

    # Unique key for uploader to allow clean reset
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    uploaded = st.file_uploader(
        "Choose PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state['uploader_key']}",
        help=f"Accepts research papers in PDF format up to {MAX_UPLOAD_MB} MB each."
    )

    if uploaded:
        total_size_mb = sum(len(f.getvalue()) for f in uploaded) / (1024 * 1024)
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25); border-radius:10px; padding:0.6rem 1rem; margin:0.8rem 0;">
            <span style="font-size:0.9rem; font-weight:600; color:#c4b5fd;">
                📄 {len(uploaded)} file{'s' if len(uploaded) != 1 else ''} selected ({total_size_mb:.2f} MB total)
            </span>
            <span style="font-size:0.78rem; color:#94a3b8;">Ready for semantic extraction</span>
        </div>
        """, unsafe_allow_html=True)

        c_ingest, c_clear = st.columns([3, 1])
        with c_ingest:
            start_ingest = st.button(
                f"🚀 Ingest & Index ({len(uploaded)} Document{'s' if len(uploaded) != 1 else ''})",
                type="primary",
                use_container_width=True
            )
        with c_clear:
            if st.button("✖ Clear Selection", use_container_width=True):
                st.session_state["uploader_key"] += 1
                st.rerun()

        if start_ingest:
            with st.spinner("Extracting text pages, generating overlapping chunks, and embedding vectors..."):
                outcomes = upload_papers(uploaded)
                st.session_state["upload_results"] = outcomes
                st.session_state["uploader_key"] += 1
            st.rerun()
    else:
        # 3-feature capability row
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:1rem; margin-top:1.25rem;">
            <div style="background:rgba(15,23,42,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:0.9rem;">
                <div style="font-weight:700; color:#a5b4fc; font-size:0.85rem; margin-bottom:4px;">📑 Exact Page Preserved</div>
                <div style="font-size:0.78rem; color:#94a3b8; line-height:1.5;">PyMuPDF extracts per page preserving exact page numbers for authentic citations.</div>
            </div>
            <div style="background:rgba(15,23,42,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:0.9rem;">
                <div style="font-weight:700; color:#6ee7b7; font-size:0.85rem; margin-bottom:4px;">✂️ Overlapping Chunks</div>
                <div style="font-size:0.78rem; color:#94a3b8; line-height:1.5;">220-word windows with 40-word overlap prevent boundary truncation of findings.</div>
            </div>
            <div style="background:rgba(15,23,42,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:0.9rem;">
                <div style="font-weight:700; color:#fcd34d; font-size:0.85rem; margin-bottom:4px;">⚡ SHA-256 Deduplication</div>
                <div style="font-size:0.78rem; color:#94a3b8; line-height:1.5;">Automatic duplicate detection ensures clean storage and zero vector redundancy.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Ingestion outcomes banner
    if "upload_results" in st.session_state and st.session_state["upload_results"]:
        st.markdown("<div style='margin-bottom:1rem;'><strong>📋 Recent Ingestion Status:</strong></div>", unsafe_allow_html=True)
        for name, status, detail in st.session_state["upload_results"]:
            badge_class = "rm-badge-emerald" if status == "Indexed" else "rm-badge-amber"
            icon = "✅" if status == "Indexed" else ("ℹ️" if status == "Duplicate" else "⚠️")
            st.markdown(f"""
            <div style="background:rgba(19,27,46,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.5rem; display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:600; color:#f1f5f9; display:flex; align-items:center; gap:8px;">{icon} {name}</span>
                <span class="rm-badge {badge_class}">{status} · {detail}</span>
            </div>
            """, unsafe_allow_html=True)
        if st.button("✕ Dismiss Notification", type="secondary", key="dismiss_upload_notif"):
            del st.session_state["upload_results"]
            st.rerun()


    st.markdown("---")
    st.markdown("### 📚 Your Paper Collection")

    with get_db() as db:
        papers = read_papers(db)

    if papers:
        filter_col, view_col = st.columns([3, 1])
        with filter_col:
            search_query = st.text_input("🔍 Search library papers by title or filename", placeholder="Type to filter...")
        
        filtered_papers = [p for p in papers if not search_query or search_query.lower() in p.title.lower() or search_query.lower() in p.filename.lower()]
        
        st.markdown(f"<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:1rem;'>Showing {len(filtered_papers)} of {len(papers)} papers</div>", unsafe_allow_html=True)

        for p in filtered_papers:
            col_card, col_action = st.columns([5, 1])
            with col_card:
                st.markdown(f"""
                <div class="rm-paper-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
                        <span style="font-size:1.05rem; font-weight:700; color:#f8fafc;">📄 {p.title}</span>
                        <span class="rm-badge rm-badge-indigo">ID: {p.id}</span>
                    </div>
                    <div style="display:flex; gap:12px; font-size:0.8rem; color:#94a3b8; margin-top:8px;">
                        <span>📖 <strong>{p.page_count}</strong> pages</span>
                        <span>📁 <code>{p.filename}</code></span>
                        <span>🕒 {p.uploaded_at.strftime('%b %d, %Y · %H:%M')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_action:
                st.write("")
                if st.button("🗑️ Delete", key=f"del_{p.id}", type="secondary", help="Remove paper and vector chunks"):
                    pid = p.id
                    get_index().remove_paper(pid)
                    with get_db() as db_action:
                        paper_obj = db_action.get(Paper, pid)
                        if paper_obj:
                            Path(paper_obj.path).unlink(missing_ok=True)
                            db_action.delete(paper_obj)
                            db_action.commit()
                    st.success(f"Removed paper #{pid}")
                    st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; background:rgba(19,27,46,0.4); border:1px dashed rgba(255,255,255,0.1); border-radius:16px;">
            <div style="font-size:2.5rem; margin-bottom:0.75rem;">📂</div>
            <div style="font-size:1.2rem; font-weight:700; color:#f1f5f9;">Your research library is empty</div>
            <div style="font-size:0.9rem; color:#94a3b8; margin-top:0.35rem;">Upload text-based PDF papers above to start semantic search and question answering.</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: ASK & SYNTHESIZE
# ==========================================
with tab_ask:
    st.markdown("### 💬 Evidence-Grounded Research Questioning")
    st.caption("Ask questions across multiple papers. Answers are strictly bounded by retrieved evidence with passage citations.")

    with get_db() as db:
        papers = read_papers(db)
        pmap = {f"{p.id}: {p.title}": p.id for p in papers}

    if not papers:
        st.warning("⚠️ No papers in your library yet. Ingest documents in the Library tab first.")
    else:
        # Prompt Inspiration Chips
        st.markdown("<div style='font-size:0.82rem; font-weight:600; color:#94a3b8; margin-bottom:6px;'>⚡ Quick Prompts:</div>", unsafe_allow_html=True)
        q_cols = st.columns(4)
        sample_q = ""
        with q_cols[0]:
            if st.button("🎯 Research Objectives", use_container_width=True):
                sample_q = "What are the primary objectives and research questions investigated in these papers?"
        with q_cols[1]:
            if st.button("🔬 Methodologies", use_container_width=True):
                sample_q = "Compare the methodologies, architectures, and experimental approaches used."
        with q_cols[2]:
            if st.button("⚠️ Limitations", use_container_width=True):
                sample_q = "What explicit limitations, failure modes, and future research directions are discussed?"
        with q_cols[3]:
            if st.button("📊 Datasets & Benchmarks", use_container_width=True):
                sample_q = "What datasets, benchmarks, evaluation metrics, and baselines were tested?"

        scope = st.multiselect(
            "Search Scope",
            ["All papers"] + list(pmap.keys()),
            default=["All papers"],
            help="Filter search to specific papers or search the entire library."
        )

        initial_val = sample_q if sample_q else st.session_state.get("last_question", "")
        question = st.text_area(
            "Your Research Question",
            value=initial_val,
            placeholder="e.g. Synthesize how each paper handles domain adaptation and cite the specific evidence.",
            height=100
        )

        slider_col, btn_col = st.columns([3, 1])
        with slider_col:
            top_k = st.slider("Evidence Passages to Retrieve (Top-K)", min_value=2, max_value=10, value=5)
        with btn_col:
            st.write("")
            ask_clicked = st.button("✨ Synthesize Answer", type="primary", use_container_width=True)

        if ask_clicked:
            if not question.strip():
                st.warning("Please enter a research question first.")
            else:
                st.session_state["last_question"] = question.strip()
                target_ids = [pmap[x] for x in scope if x in pmap] if scope and "All papers" not in scope else None
                with st.spinner("Retrieving relevant passages and synthesizing evidence..."):
                    contexts = get_index().search(question.strip(), top_k=top_k, paper_ids=target_ids)
                    answer = generate_grounded_answer(question.strip(), contexts)

                st.markdown("""
                <div class="rm-answer-container">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                        <span style="font-weight:700; font-size:1.15rem; color:#f8fafc; display:flex; align-items:center; gap:8px;">
                            <span>🧠</span> Synthesized Research Intelligence
                        </span>
                        <span class="rm-badge rm-badge-indigo">Evidence-Grounded</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(answer)

                st.markdown("---")
                st.markdown(f"### 📑 Retrieved Evidence Passages ({len(contexts)})")

                if contexts:
                    for i, c in enumerate(contexts, 1):
                        score_pct = int(min(max(c.get("score", 0.0), 0.0), 1.0) * 100)
                        st.markdown(f"""
                        <div class="rm-evidence-box">
                            <div class="rm-evidence-meta">
                                <div>
                                    <strong style="color:#f1f5f9; font-size:0.95rem;">[{i}] {c['title']}</strong>
                                    <span style="margin-left:8px;" class="rm-badge rm-badge-cyan">Page {c['page_number']}</span>
                                </div>
                                <div>
                                    <span class="rm-badge rm-badge-emerald">Relevance: {score_pct}% (Score {c['score']:.3f})</span>
                                </div>
                            </div>
                            <div class="rm-evidence-text">
                                {c['text']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No matching passages found. Try rephrasing your question or broadening search scope.")

# ==========================================
# TAB 3: CROSS-PAPER COMPARISON
# ==========================================
with tab_compare:
    st.markdown("### ⚖️ Multi-Paper Comparative Matrix")
    st.caption("Side-by-side heuristic extraction of core objectives, methodologies, and limitations across papers.")

    with get_db() as db:
        papers = read_papers(db)
        pmap = {f"{p.id}: {p.title}": p for p in papers}

    if len(papers) < 2:
        st.info("💡 Upload at least two papers in the Library tab to run a comparative matrix.")
    else:
        selected_keys = st.multiselect(
            "Select 2 to 5 papers to compare",
            options=list(pmap.keys()),
            default=list(pmap.keys())[:min(len(pmap), 3)],
            max_selections=5
        )

        if st.button("📊 Generate Comparison Matrix", type="primary", disabled=len(selected_keys) < 2):
            with st.spinner("Extracting candidate objectives, methodologies, and limitations..."):
                comp_data = []
                with get_db() as db_comp:
                    for key in selected_keys:
                        p = pmap[key]
                        chunks = db_comp.scalars(
                            select(Chunk).where(Chunk.paper_id == p.id).order_by(Chunk.chunk_index)
                        ).all()
                        comp_data.append({
                            "title": p.title,
                            "chunks": [{"text": c.text, "page_number": c.page_number} for c in chunks]
                        })

                result = compare_papers(comp_data)
                frame = pd.DataFrame(result)

            # Display comparative cards
            for item in result:
                st.markdown(f"""
                <div class="rm-glass-card" style="margin-bottom:1.25rem;">
                    <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin-bottom:1rem; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:0.5rem;">
                        📄 {item['title']}
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:1.2rem;">
                        <div style="background:rgba(15,23,42,0.5); padding:1rem; border-radius:10px; border-left:3px solid #6366f1;">
                            <div class="rm-comp-badge" style="background:rgba(99,102,241,0.2); color:#a5b4fc;">🎯 Objective / Problem</div>
                            <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.5;">{item.get('objective / problem', 'N/A')}</div>
                        </div>
                        <div style="background:rgba(15,23,42,0.5); padding:1rem; border-radius:10px; border-left:3px solid #10b981;">
                            <div class="rm-comp-badge" style="background:rgba(16,185,129,0.2); color:#6ee7b7;">🔬 Methods & Approach</div>
                            <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.5;">{item.get('methods', 'N/A')}</div>
                        </div>
                        <div style="background:rgba(15,23,42,0.5); padding:1rem; border-radius:10px; border-left:3px solid #f59e0b;">
                            <div class="rm-comp-badge" style="background:rgba(245,158,11,0.2); color:#fcd34d;">⚠️ Limitations & Future Work</div>
                            <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.5;">{item.get('limitations', 'N/A')}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            csv_data = frame.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Comparison CSV",
                data=csv_data,
                file_name="researchmate_paper_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==========================================
# TAB 4: RESEARCH NOTES
# ==========================================
with tab_notes:
    st.markdown("### 📝 Research Notebook")
    st.caption("Record structured observations, hypotheses, and citation links across papers.")

    with get_db() as db:
        papers = read_papers(db)
        note_opts = {"General Research Note": None}
        note_opts.update({f"{p.id}: {p.title}": p.id for p in papers})

    col_compose, col_saved = st.columns([1, 1])

    with col_compose:
        st.markdown("##### ✏️ New Note Entry")
        selected_paper_for_note = st.selectbox("Link to Paper (Optional)", list(note_opts.keys()))
        note_content = st.text_area(
            "Note Body",
            placeholder="Record key findings, methodology critiques, or ideas for follow-up experiments...",
            height=140
        )
        if st.button("💾 Save Research Note", type="primary"):
            if not note_content.strip():
                st.warning("Note body cannot be empty.")
            else:
                with get_db() as db_note:
                    db_note.add(Note(
                        paper_id=note_opts[selected_paper_for_note],
                        body=note_content.strip()
                    ))
                    db_note.commit()
                st.success("Note saved successfully!")
                st.rerun()

    with col_saved:
        st.markdown("##### 📋 Saved Observations")
        with get_db() as db_list:
            saved_notes = db_list.execute(
                select(Note, Paper.title)
                .outerjoin(Paper, Note.paper_id == Paper.id)
                .order_by(Note.created_at.desc())
            ).all()

        if saved_notes:
            for note_item, paper_title in saved_notes:
                st.markdown(f"""
                <div class="rm-glass-card" style="padding:1rem; margin-bottom:0.75rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span class="rm-badge rm-badge-indigo">📄 {paper_title or 'General Note'}</span>
                        <span style="font-size:0.75rem; color:#64748b;">{note_item.created_at.strftime('%Y-%m-%d %H:%M')}</span>
                    </div>
                    <div style="font-size:0.9rem; color:#e2e8f0; line-height:1.5;">
                        {note_item.body}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No research notes saved yet. Write your first observation on the left.")

# ==========================================
# TAB 5: ARCHITECTURE & ABOUT
# ==========================================
with tab_about:
    st.markdown("### 🏛️ Platform Architecture & Engineering Design")
    
    # Architecture Flow Diagram
    st.markdown("""
    <div class="rm-glass-card" style="margin-bottom:1.5rem;">
        <div style="font-weight:700; font-size:1.1rem; color:#f1f5f9; margin-bottom:1.25rem;">
            End-to-End Grounded Retrieval-Augmented Generation (RAG) Pipeline
        </div>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:1rem;">
            <div class="rm-pipeline-step">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">📑</div>
                <div style="font-weight:700; color:#f1f5f9; font-size:0.9rem;">1. PyMuPDF Ingestion</div>
                <div style="font-size:0.76rem; color:#94a3b8; margin-top:4px;">Page-level text extraction preserving page coordinate index</div>
            </div>
            <div class="rm-pipeline-step">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">✂️</div>
                <div style="font-weight:700; color:#f1f5f9; font-size:0.9rem;">2. Sliding Window</div>
                <div style="font-size:0.76rem; color:#94a3b8; margin-top:4px;">220-word chunks with 40-word overlapping boundaries</div>
            </div>
            <div class="rm-pipeline-step">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">🧠</div>
                <div style="font-weight:700; color:#f1f5f9; font-size:0.9rem;">3. Vector Embeddings</div>
                <div style="font-size:0.76rem; color:#94a3b8; margin-top:4px;">Dense semantic vectors + fallback index</div>
            </div>
            <div class="rm-pipeline-step">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">🔍</div>
                <div style="font-weight:700; color:#f1f5f9; font-size:0.9rem;">4. Semantic Search</div>
                <div style="font-size:0.76rem; color:#94a3b8; margin-top:4px;">Inner product similarity & paper filtering</div>
            </div>
            <div class="rm-pipeline-step">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">✨</div>
                <div style="font-weight:700; color:#f1f5f9; font-size:0.9rem;">5. Grounded Synthesis</div>
                <div style="font-size:0.76rem; color:#94a3b8; margin-top:4px;">LLM cited generation with untrusted doc sandboxing</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_eval, col_specs = st.columns(2)
    with col_eval:
        st.markdown("##### 🔬 Evaluation & Integrity Standards")
        st.markdown("""
        - **Citation Fidelity:** Every claim is required to anchor to explicit numbered passage citations (`[1]`, `[2]`).
        - **Untrusted Document Prompting:** Model system prompts treat retrieved text as untrusted data to mitigate prompt injection.
        - **Deterministic Fallback:** When no LLM key is configured, the system provides zero-hallucination passage retrieval without synthesizing untrusted speculation.
        """)

    with col_specs:
        st.markdown("##### ⚙️ Technical Specifications")
        st.markdown(f"""
        - **Database:** SQLAlchemy 2.0 ORM (`{DATABASE_URL.split('://')[0]}`)
        - **Retrieval Index:** `{idx_engine.backend_name if hasattr(idx_engine, 'backend_name') else 'Dense Vector'}`
        - **Default LLM:** `{LLM_MODEL}`
        - **Chunk Parameters:** `220 words / 40 overlap`
        - **Max Ingestion Limit:** `{MAX_UPLOAD_MB} MB per PDF`
        """)

    st.caption("ResearchMate AI Portfolio Platform — Designed for academic document intelligence and rigorous multi-paper cross-examination.")
