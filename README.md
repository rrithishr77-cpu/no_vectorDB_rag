# RAG-Style QA + Summarization Engine

A retrieval-augmented question-answering and summarization pipeline that
runs **fully offline** with zero API keys, and **auto-upgrades itself**
if you later install heavier NLP libraries — no code changes required.

## Why it's built this way

Most RAG tutorials assume you always have internet + a hosted LLM. This
one is designed to degrade gracefully:

| Component  | Base mode (always works)              | Auto-upgrades to (if installed)                  |
|------------|----------------------------------------|---------------------------------------------------|
| Retrieval  | TF-IDF + cosine similarity            | Dense embeddings via `sentence-transformers`       |
| Summary    | Extractive TextRank (PageRank on sentence graph) | Abstractive via `transformers` (BART-large-CNN) |
| QA         | Extractive best-sentence match        | Generative via `transformers` (DistilBERT-SQuAD)   |

At import time, each component tries to `import` the heavier library in a
`try/except`. If it's missing (or offline), it silently falls back — you
never see a crash, just a different `mode` in the output.

## Files

- `rag_engine.py` — the engine: chunking, retrieval, summarization, QA, and a runnable demo.
- `cli.py` — point it at any `.txt` file of yours and ask it questions interactively.

## Run the demo (no setup needed)

```bash
python3 rag_engine.py
```

## Run on your own notes

```bash
python3 cli.py my_notes.txt
```

## Unlock the upgraded (generative) mode

```bash
pip install sentence-transformers transformers torch
python3 cli.py my_notes.txt
```

You'll see `status()` report `dense` / `generative` instead of `tfidf` /
`extractive` — same code, better answers, because retrieval and QA now
understand meaning instead of just word overlap.

## How the pieces work (for study)

1. **Chunking** (`chunk_text`) — splits text into ~80-word, sentence-respecting,
   overlapping windows so context isn't cut mid-thought at chunk boundaries.
2. **Retrieval** (`Retriever`) — TF-IDF vectorizes each chunk; a query is
   vectorized the same way and ranked by cosine similarity. This *is*
   the "R" in RAG — it decides which chunks are relevant before any
   generation happens.
3. **Summarization** (`Summarizer`) — TextRank builds a graph where nodes
   are sentences and edge weights are their TF-IDF cosine similarity,
   then runs PageRank to find the most "central" sentences — the same
   algorithm behind Google's original PageRank, applied to sentences
   instead of web pages.
4. **QA** (`RAGPipeline.answer`) — retrieves top-k chunks, concatenates
   them as context, then either runs a real extractive-QA model on that
   context (if `transformers` is available) or falls back to picking the
   single sentence most similar to the question.

## Extend it

- Swap `chunk_text` for a recursive/semantic chunker.
- Swap TF-IDF for FAISS + real embeddings for large corpora (thousands of chunks).
- Add a `Chunk.metadata` field (page number, section) for citation-style answers.
- Wire `cli.py` into a Flask/FastAPI endpoint to demo it as a web app.
