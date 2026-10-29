import json
import numpy as np
from pathlib import Path
from app.config import EMBEDDING_MODEL, INDEX_DIR

# Resilient retrieval engine: supports FAISS, SentenceTransformers with NumPy, or TF-IDF fallback
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

if not HAS_ST:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity


class SemanticIndex:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.metadata = []
        self.index_path = INDEX_DIR / "researchmate.faiss"
        self.meta_path = INDEX_DIR / "researchmate_metadata.json"
        self.vectors_path = INDEX_DIR / "researchmate_vectors.npy"

        if HAS_ST:
            self.model = SentenceTransformer(model_name)
            if HAS_FAISS:
                self.backend_name = "Sentence Transformers + FAISS"
                self.index = None
                if self.index_path.exists() and self.meta_path.exists():
                    self.index = faiss.read_index(str(self.index_path))
                    self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            else:
                self.backend_name = "Sentence Transformers (Neural Dense)"
                self.vectors = None
                if self.vectors_path.exists() and self.meta_path.exists():
                    self.vectors = np.load(str(self.vectors_path))
                    self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self.backend_name = "TF-IDF Neural-Lexical Index"
            self.vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
            self.vectors = None
            if self.meta_path.exists():
                self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if self.metadata:
                    self._rebuild_tfidf()

    def _rebuild_tfidf(self):
        if not self.metadata:
            self.vectors = None
            return
        corpus = [m["text"] for m in self.metadata]
        self.vectors = self.vectorizer.fit_transform(corpus)

    def add_chunks(self, chunks):
        if not chunks:
            return

        new_meta = [{k: c[k] for k in ("paper_id", "title", "page_number", "text")} for c in chunks]
        self.metadata.extend(new_meta)
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

        if HAS_ST:
            texts = [c["text"] for c in chunks]
            new_vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            if HAS_FAISS:
                if self.index is None:
                    self.index = faiss.IndexFlatIP(new_vecs.shape[1])
                self.index.add(new_vecs)
                faiss.write_index(self.index, str(self.index_path))
            else:
                if self.vectors is None:
                    self.vectors = new_vecs
                else:
                    self.vectors = np.vstack([self.vectors, new_vecs])
                np.save(str(self.vectors_path), self.vectors)
        else:
            self._rebuild_tfidf()

    def search(self, query, top_k=6, paper_ids=None):
        if not self.metadata:
            return []

        results = []
        if HAS_ST:
            q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            if HAS_FAISS and self.index is not None:
                scores, ids = self.index.search(q, min(max(top_k * 4, top_k), len(self.metadata)))
                for score, idx in zip(scores[0], ids[0]):
                    if idx < 0:
                        continue
                    m = self.metadata[int(idx)]
                    if paper_ids and m["paper_id"] not in paper_ids:
                        continue
                    results.append({**m, "score": float(score)})
                    if len(results) >= top_k:
                        break
            elif self.vectors is not None:
                sims = np.dot(self.vectors, q[0])
                ranked_indices = np.argsort(-sims)
                for idx in ranked_indices:
                    m = self.metadata[int(idx)]
                    if paper_ids and m["paper_id"] not in paper_ids:
                        continue
                    results.append({**m, "score": float(sims[idx])})
                    if len(results) >= top_k:
                        break
        else:
            if self.vectors is None or len(self.metadata) == 0:
                return []
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.vectors)[0]
            ranked_indices = np.argsort(-sims)
            for idx in ranked_indices:
                if sims[idx] <= 0:
                    continue
                m = self.metadata[int(idx)]
                if paper_ids and m["paper_id"] not in paper_ids:
                    continue
                results.append({**m, "score": float(sims[idx])})
                if len(results) >= top_k:
                    break

        return results

    def remove_paper(self, paper_id):
        kept = [m for m in self.metadata if m["paper_id"] != paper_id]
        if not kept:
            self.metadata = []
            for p in (self.index_path, self.meta_path, self.vectors_path):
                if p.exists():
                    p.unlink(missing_ok=True)
            if HAS_FAISS:
                self.index = None
            else:
                self.vectors = None
            return

        self.metadata = kept
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

        if HAS_ST:
            texts = [m["text"] for m in kept]
            vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            if HAS_FAISS:
                self.index = faiss.IndexFlatIP(vecs.shape[1])
                self.index.add(vecs)
                faiss.write_index(self.index, str(self.index_path))
            else:
                self.vectors = vecs
                np.save(str(self.vectors_path), self.vectors)
        else:
            self._rebuild_tfidf()
