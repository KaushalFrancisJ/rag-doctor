"""Smoke test script for RAG Doctor pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.chunker import chunk_directory
from pipeline.embedder import Embedder
from pipeline.generator import Generator
from pipeline.index import VectorIndex
from pipeline.retriever import Retriever

load_dotenv()

SAMPLE_QUESTIONS = [
    "How do background tasks work in FastAPI and how do you define them?",
    "What is the difference between async def and def in path operation functions?",
    "How does dependency injection work with Depends in FastAPI?",
]


def run_smoke_test(index_dir: str = "indexes/v001"):
    print("=" * 60)
    print("  RAG DOCTOR — SMOKE TEST")
    print("=" * 60)

    index_path = Path(index_dir)
    if not (index_path / "index.faiss").exists():
        print(f"\n[!] Index at '{index_dir}' not found. Building index first...")
        chunks = chunk_directory("corpus/active", chunk_size=500, overlap=100)
        print(f"    Loaded {len(chunks)} chunks from corpus/active")
        embedder = Embedder()
        index = VectorIndex.build(
            chunks=chunks,
            embedder=embedder,
            save_dir=index_dir,
            version="v001",
            chunk_size=500,
            overlap=100,
        )
        print(f"    Index v001 created successfully.")
    else:
        print(f"\n[+] Loading existing index from '{index_dir}'...")
        index = VectorIndex.load(index_dir)
        embedder = Embedder(model_name=index.config.get("embedding_model", "all-MiniLM-L6-v2"))
        print(f"    Index loaded with {index.config.get('total_chunks')} chunks.")

    retriever = Retriever(index=index, embedder=embedder, default_top_k=3)

    groq_key = os.getenv("GROQ_API_KEY")
    generator = None
    if groq_key:
        print(f"[+] GROQ_API_KEY detected. Using Groq model '{os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}'.")
        generator = Generator(api_key=groq_key)
    else:
        print("[!] Note: GROQ_API_KEY not set in environment or .env. Retrieval will be tested; LLM generation will be skipped.")

    print("\n" + "=" * 60)
    print("  RUNNING SAMPLE QUERIES")
    print("=" * 60)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n[{i}/{len(SAMPLE_QUESTIONS)}] QUESTION: {question}")
        chunks = retriever.retrieve(question, top_k=3)
        print(f"    Retrieved {len(chunks)} context chunks:")
        for rank, c in enumerate(chunks, 1):
            src = c["metadata"].get("relative_path", "unknown")
            score = c["score"]
            snippet = c["content"].replace("\n", " ")[:120]
            print(f"      ({rank}) [{src}] (score: {score:.4f}) -> {snippet}...")

        if generator:
            print("\n    Generating response from Groq...")
            res = generator.generate(question, chunks)
            print(f"\n    ANSWER:\n{res['answer']}\n")
            print("-" * 60)
        else:
            print("    (Skipping Groq generation since GROQ_API_KEY is not set)")
            print("-" * 60)

    print("\n=== SMOKE TEST COMPLETE ===")


if __name__ == "__main__":
    run_smoke_test()
