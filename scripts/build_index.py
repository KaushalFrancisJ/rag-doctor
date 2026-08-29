"""CLI script to build and version a FAISS index from a corpus directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.chunker import chunk_directory
from pipeline.embedder import Embedder
from pipeline.index import VectorIndex


def main():
    parser = argparse.ArgumentParser(description="Build and persist a FAISS index for RAG Doctor.")
    parser.add_argument("--corpus-dir", type=str, default="corpus/active", help="Path to corpus directory")
    parser.add_argument("--output-dir", type=str, default="indexes/v001", help="Output directory for index artifacts")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size in characters")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap in characters")
    parser.add_argument("--top-k", type=int, default=3, help="Default retrieval top_k")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--version", type=str, default="v001", help="Version tag (e.g. v001)")

    args = parser.parse_args()

    print(f"=== Building FAISS Index ({args.version}) ===")
    print(f"Corpus directory: {args.corpus_dir}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}, Top-K: {args.top_k}")
    print(f"Embedding model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")

    # 1. Chunk documents
    print("\n[1/3] Chunking documents...")
    chunks = chunk_directory(
        directory_path=args.corpus_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Generated {len(chunks)} chunks from corpus.")

    # 2. Embed and build index
    print("\n[2/3] Initializing embedder and encoding chunks...")
    embedder = Embedder(model_name=args.model_name)

    print("\n[3/3] Building and saving FAISS index...")
    index = VectorIndex.build(
        chunks=chunks,
        embedder=embedder,
        save_dir=args.output_dir,
        version=args.version,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
    )

    print(f"\nSuccessfully built and saved index to {args.output_dir}!")
    print(f"Total chunks: {index.config['total_chunks']}")
    print(f"Total documents: {index.config['total_documents']}")
    print(f"Dimension: {index.config['dimension']}")
    print(f"Config recorded in: {Path(args.output_dir) / 'config.json'}")


if __name__ == "__main__":
    main()
