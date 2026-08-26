"""
cli.py — command-line front-end for rag_engine.py

Usage:
    python3 cli.py path/to/document.txt
    python3 cli.py path/to/document.txt --summary-only
"""

import argparse
import sys
from pathlib import Path

from rag_engine import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG-style QA + Summarization over a text file.")
    parser.add_argument("file", help="Path to a .txt file to load")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to retrieve per question")
    parser.add_argument("--summary-only", action="store_true", help="Only print the summary, skip QA loop")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")

    print("Loading engine (auto-detects dense/generative upgrades if installed)...")
    pipeline = RAGPipeline(use_dense=True, use_generative=True)
    print(pipeline.status())

    pipeline.ingest(text, source=path.name)

    print("\n--- SUMMARY ---")
    print(pipeline.document_summary(text, num_sentences=5))

    if args.summary_only:
        return

    print("\n--- ASK QUESTIONS (Ctrl+C or empty line to quit) ---")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            break
        result = pipeline.answer(q, top_k=args.top_k)
        print(f"A [{result['mode']}, conf={result.get('confidence', '-')}]: {result['answer']}")


if __name__ == "__main__":
    main()
