"""
rag_engine.py
=================================================
A self-contained Retrieval-Augmented QA + Summarization engine.

Design goal: it MUST run with zero internet access using pure
TF-IDF + TextRank (scikit-learn/numpy/networkx only). If you run
it on a machine with internet + `pip install sentence-transformers
transformers torch`, it will automatically upgrade itself to use
dense embeddings (MiniLM) for retrieval and a real generative model
(FLAN-T5) for abstractive answers/summaries — no code changes needed.

Pipeline:
  raw text  -> chunk()            -> list[Chunk]
  chunks    -> Retriever.index()  -> searchable index
  query     -> Retriever.search() -> top-k relevant chunks
  chunks    -> Summarizer.summarize() -> extractive or abstractive summary
  query+chunks -> answer()        -> extractive snippet or generative answer

Author: built for Rithish's NLP study stack.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------ #
# 1. Chunking
# ------------------------------------------------------------------ #

@dataclass
class Chunk:
    id: int
    text: str
    source: str = "doc"


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter — no NLTK/internet needed."""
    text = re.sub(r"\s+", " ", text.strip())
    # Split on '.', '!', '?' followed by space+capital, but keep abbreviations intact-ish.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def chunk_text(text: str, source: str = "doc", max_words: int = 80, overlap: int = 15) -> List[Chunk]:
    """
    Chunk text into overlapping windows of ~max_words, breaking on
    sentence boundaries where possible so retrieval doesn't cut a
    sentence in half.
    """
    sentences = split_sentences(text)
    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    cid = 0

    for sent in sentences:
        wc = len(sent.split())
        if current_len + wc > max_words and current:
            chunks.append(Chunk(id=cid, text=" ".join(current), source=source))
            cid += 1
            # overlap: carry the tail of the previous chunk forward
            overlap_words = " ".join(current).split()[-overlap:]
            current = [" ".join(overlap_words)] if overlap_words else []
            current_len = len(overlap_words)
        current.append(sent)
        current_len += wc

    if current:
        chunks.append(Chunk(id=cid, text=" ".join(current), source=source))

    return chunks


# ------------------------------------------------------------------ #
# 2. Retriever (TF-IDF by default, auto-upgrades to dense embeddings)
# ------------------------------------------------------------------ #

class Retriever:
    """
    TF-IDF + cosine-similarity retriever. If `sentence-transformers`
    is importable, silently swaps in dense embedding retrieval instead
    (better semantic matching, e.g. "car" ~ "automobile").
    """

    def __init__(self, use_dense: bool = True):
        self.chunks: List[Chunk] = []
        self.dense = None
        self.embeddings = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.mode = "tfidf"

        if use_dense:
            try:
                from sentence_transformers import SentenceTransformer  # noqa
                self.dense = SentenceTransformer("all-MiniLM-L6-v2")
                self.mode = "dense"
            except Exception:
                self.dense = None
                self.mode = "tfidf"

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        if self.mode == "dense":
            self.embeddings = self.dense.encode(texts, normalize_embeddings=True)
        else:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        """Returns list of (Chunk, score), highest score first."""
        if not self.chunks:
            return []

        if self.mode == "dense":
            q_emb = self.dense.encode([query], normalize_embeddings=True)
            sims = cosine_similarity(q_emb, self.embeddings)[0]
        else:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx if sims[i] > 0]


# ------------------------------------------------------------------ #
# 3. Summarizer (extractive TextRank by default, auto-upgrades to FLAN-T5)
# ------------------------------------------------------------------ #

class Summarizer:
    def __init__(self, use_generative: bool = True):
        self.gen_pipeline = None
        self.mode = "extractive"

        if use_generative:
            try:
                from transformers import pipeline  # noqa
                self.gen_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
                self.mode = "generative"
            except Exception:
                self.gen_pipeline = None
                self.mode = "extractive"

    def _textrank(self, text: str, num_sentences: int = 3) -> str:
        sentences = split_sentences(text)
        if len(sentences) <= num_sentences:
            return text

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return " ".join(sentences[:num_sentences])

        sim_matrix = cosine_similarity(matrix)
        np.fill_diagonal(sim_matrix, 0)
        graph = nx.from_numpy_array(sim_matrix)

        try:
            scores = nx.pagerank(graph, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            scores = {i: 1.0 for i in range(len(sentences))}

        ranked = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        top = sorted(ranked[:num_sentences], key=lambda x: x[1])  # restore original order
        return " ".join(s for _, _, s in top)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        if self.mode == "generative":
            try:
                out = self.gen_pipeline(text, max_length=130, min_length=30, do_sample=False)
                return out[0]["summary_text"]
            except Exception:
                pass  # fall through to extractive
        return self._textrank(text, num_sentences=num_sentences)


# ------------------------------------------------------------------ #
# 4. RAG pipeline — ties retrieval + summarization + QA together
# ------------------------------------------------------------------ #

class RAGPipeline:
    def __init__(self, use_dense: bool = True, use_generative: bool = True):
        self.retriever = Retriever(use_dense=use_dense)
        self.summarizer = Summarizer(use_generative=use_generative)
        self.qa_pipeline = None
        if use_generative:
            try:
                from transformers import pipeline  # noqa
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                )
            except Exception:
                self.qa_pipeline = None

    def ingest(self, text: str, source: str = "doc") -> None:
        chunks = chunk_text(text, source=source)
        self.retriever.index(chunks)

    def document_summary(self, text: str, num_sentences: int = 4) -> str:
        return self.summarizer.summarize(text, num_sentences=num_sentences)

    def answer(self, query: str, top_k: int = 3) -> dict:
        results = self.retriever.search(query, top_k=top_k)
        if not results:
            return {"answer": "No relevant context found in the document.", "context": [], "mode": "none"}

        context_text = " ".join(c.text for c, _ in results)

        if self.qa_pipeline is not None:
            try:
                result = self.qa_pipeline(question=query, context=context_text)
                return {
                    "answer": result["answer"],
                    "confidence": round(float(result["score"]), 3),
                    "context": [(c.text, round(s, 3)) for c, s in results],
                    "mode": "generative",
                }
            except Exception:
                pass

        # Extractive fallback: best-matching chunk stands in as the answer,
        # trimmed to the most relevant sentence within it.
        best_chunk, best_score = results[0]
        sentences = split_sentences(best_chunk.text)
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            matrix = vectorizer.fit_transform(sentences + [query])
            sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
            best_sentence = sentences[int(np.argmax(sims))]
        except ValueError:
            best_sentence = best_chunk.text

        return {
            "answer": best_sentence,
            "confidence": round(best_score, 3),
            "context": [(c.text, round(s, 3)) for c, s in results],
            "mode": "extractive",
        }

    def status(self) -> str:
        return (
            f"Retrieval mode : {self.retriever.mode}\n"
            f"Summary mode   : {self.summarizer.mode}\n"
            f"QA mode        : {'generative' if self.qa_pipeline else 'extractive-fallback'}"
        )


# ------------------------------------------------------------------ #
# 5. Demo
# ------------------------------------------------------------------ #

SAMPLE_DOC = """
Machine learning models learn patterns from data instead of following
hand-written rules. A classic pipeline starts with data collection,
followed by cleaning and feature engineering, then model training and
evaluation. Supervised learning uses labeled examples, where the model
learns a mapping from inputs to known outputs, such as predicting house
prices from features like square footage and location. Unsupervised
learning instead looks for structure in unlabeled data, such as grouping
customers into segments based on purchasing behavior using clustering
algorithms like k-means. Deep learning is a subset of machine learning
that uses neural networks with many layers to automatically learn
hierarchical feature representations, which has driven major advances in
computer vision and natural language processing. Overfitting occurs when
a model memorizes training data instead of learning generalizable
patterns, and is typically addressed with regularization, dropout, or
gathering more training data. Model evaluation for classification tasks
commonly relies on metrics such as accuracy, precision, recall, and F1
score, while regression tasks use metrics like mean squared error and
R-squared. Cross-validation, especially k-fold cross-validation, is a
standard technique for estimating how well a model will generalize to
unseen data by repeatedly training and testing on different splits of
the dataset. Feature scaling, such as standardization or min-max
normalization, is important for algorithms sensitive to the magnitude of
input features, including gradient descent-based models and distance-based
methods like k-nearest neighbors. Ensemble methods, such as random forests
and gradient boosting, combine predictions from multiple models to improve
robustness and accuracy compared to any single model alone.
"""


def run_demo() -> None:
    print("=" * 70)
    print("RAG-STYLE QA + SUMMARIZATION ENGINE — DEMO")
    print("=" * 70)

    pipeline = RAGPipeline(use_dense=True, use_generative=True)
    print("\n[Engine status]")
    print(pipeline.status())

    pipeline.ingest(SAMPLE_DOC, source="ml_notes.txt")

    print("\n" + "-" * 70)
    print("DOCUMENT SUMMARY")
    print("-" * 70)
    summary = pipeline.document_summary(SAMPLE_DOC, num_sentences=3)
    print(textwrap.fill(summary, width=70))

    questions = [
        "What is the difference between supervised and unsupervised learning?",
        "How is overfitting addressed?",
        "What metrics are used for regression evaluation?",
    ]

    print("\n" + "-" * 70)
    print("QUESTION ANSWERING")
    print("-" * 70)
    for q in questions:
        result = pipeline.answer(q, top_k=2)
        print(f"\nQ: {q}")
        print(f"A [{result['mode']}]: {textwrap.fill(result['answer'], width=70)}")


if __name__ == "__main__":
    run_demo()
