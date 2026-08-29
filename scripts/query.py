"""CLI tool to query the RAG pipeline interactively or from command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Query the RAG Doctor pipeline.")
    parser.add_argument("question", nargs="?", type=str, help="Question to ask")
    parser.add_argument("--index-dir", type=str, default="indexes/v001", help="Path to index directory")
    parser.add_argument("--top-k", type=int, default=None, help="Number of context chunks to retrieve (defaults to index config)")

    args = parser.parse_args()

    rag = RAGPipeline(index_dir=args.index_dir, top_k=args.top_k)

    if args.question:
        run_query(rag, args.question, args.top_k)
    else:
        print(f"Loaded RAG Pipeline with index: {args.index_dir} (type 'exit' or 'quit' to stop)\n")
        while True:
            try:
                q = input("\nEnter your question: ").strip()
                if not q or q.lower() in ("exit", "quit", "q"):
                    break
                run_query(rag, q, args.top_k)
            except (KeyboardInterrupt, EOFError):
                break


def run_query(rag: RAGPipeline, question: str, top_k: int | None = None):
    effective_k = top_k if top_k is not None else rag.top_k
    print(f"\n--- Question: {question} ---")
    retrieved = rag.retriever.retrieve(question, top_k=effective_k)
    print(f"\nRetrieved {len(retrieved)} context chunks (top_k={effective_k}):")
    for i, chunk in enumerate(retrieved, 1):
        source = chunk["metadata"].get("relative_path", "unknown")
        print(f"  [{i}] Source: {source} (Score: {chunk['score']:.4f})")
        print(f"      Snippet: {chunk['content'][:140]}...\n")

    print("Generating answer from Groq...")
    result = rag.query(question, top_k=effective_k)
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
