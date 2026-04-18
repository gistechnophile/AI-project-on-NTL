"""
Lightweight RAG engine for NTL-demography literature retrieval.
Session 8: Retrieval-Augmented Generation (chunking, embeddings, vector search).

Uses TF-IDF + cosine similarity via scikit-learn (no heavy dependencies).
Indexes extracted text from academic papers in Sessions/literature/extracted/.
"""
import os
import re
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DEFAULT_LITERATURE_DIR = Path("../Sessions/literature/extracted")
CHUNK_SIZE = 500       # words per chunk
CHUNK_OVERLAP = 100    # words overlap between chunks


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------
def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        # Clean up: remove excessive whitespace, keep printable
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if len(chunk) > 50:  # skip tiny fragments
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


def _load_papers(literature_dir: Path) -> List[str]:
    """Load all .txt files from literature directory and chunk them."""
    all_chunks = []
    if not literature_dir.exists():
        # Fallback: use default corpus if literature folder not found
        return build_default_corpus()

    txt_files = sorted(literature_dir.glob("*.txt"))
    if not txt_files:
        return build_default_corpus()

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8", errors="ignore")
            chunks = _chunk_text(text)
            # Tag each chunk with source filename
            tagged = [f"[{txt_file.stem}] {chunk}" for chunk in chunks]
            all_chunks.extend(tagged)
        except Exception:
            continue

    return all_chunks if all_chunks else build_default_corpus()


# ------------------------------------------------------------------
# RAG Engine
# ------------------------------------------------------------------
class LiteratureRAG:
    """
    Minimal RAG using TF-IDF + cosine similarity.
    No external heavy dependencies (langchain, chromadb, sentence-transformers).
    """
    def __init__(self, literature_dir: Path = None):
        self.literature_dir = literature_dir or DEFAULT_LITERATURE_DIR
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self.chunk_texts: List[str] = []
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        """Load papers, chunk, and fit TF-IDF."""
        self.chunk_texts = _load_papers(self.literature_dir)
        if not self.chunk_texts:
            self.chunk_texts = ["No literature corpus available."]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def query(self, question: str, k: int = 3) -> List[str]:
        """Return top-k most similar chunks to the question."""
        if self.tfidf_matrix is None or len(self.chunk_texts) == 0:
            return ["Literature database not initialized."]

        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idx:
            if scores[idx] > 0.01:  # similarity threshold
                results.append(self.chunk_texts[idx])
        if not results:
            return ["No highly relevant literature found for this query."]
        return results

    def reload(self):
        """Rebuild index (useful if literature folder changes)."""
        self._build_index()


# ------------------------------------------------------------------
# Default fallback corpus
# ------------------------------------------------------------------
def build_default_corpus() -> List[str]:
    """Seed corpus with key NTL-population findings (fallback)."""
    return [
        "[General] Nighttime lights are a strong proxy for economic activity and urban population density in developed regions.",
        "[General] In rural South Asia, low luminosity areas may still host substantial agricultural populations, leading to underestimation.",
        "[General] Calibration drift in DMSP-OLS sensors requires inter-calibration when comparing multi-year nighttime light composites.",
        "[General] VIIRS day-night band offers higher spatial resolution and reduced saturation compared to DMSP-OLS, improving population estimates.",
        "[General] Machine learning models such as random forests and convolutional neural networks can improve gridded population mapping by fusing nighttime lights with land-cover features.",
        "[General] Validation against census data remains essential; model error increases in informal settlements and conflict zones.",
        "[Pakistan] Pakistan's demographic transition shows rapid urbanization in Punjab and Sindh provinces, detectable via increasing nighttime luminosity trends.",
        "[Pakistan] Snow and cloud contamination in mountainous regions of Pakistan introduce noise into annual nighttime light composites.",
    ]


# ------------------------------------------------------------------
# CLI test
# ------------------------------------------------------------------
if __name__ == "__main__":
    rag = LiteratureRAG()
    print(f"Indexed {len(rag.chunk_texts)} chunks from literature.")

    test_queries = [
        "Why does my model underestimate rural Pakistan population?",
        "How does building volume improve population estimation?",
        "What causes NTL saturation in urban cores?",
    ]

    for q in test_queries:
        print(f"\n--- Query: {q} ---")
        for i, result in enumerate(rag.query(q, k=2), 1):
            # Truncate for display
            snippet = result[:300] + "..." if len(result) > 300 else result
            print(f"{i}. {snippet}")
