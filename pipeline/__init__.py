"""RAG Pipeline module for RAG Doctor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.chunker import Chunk, chunk_directory, chunk_file, chunk_text
from pipeline.embedder import Embedder
from pipeline.generator import Generator
from pipeline.index import VectorIndex
from pipeline.retriever import Retriever


class RAGPipeline:
    """Convenience class combining Retriever and Generator for end-to-end question answering."""

    def __init__(
        self,
        index_dir: str | Path = "indexes/v001",
        top_k: int = 3,
        generator: Optional[Generator] = None,
    ):
        self.index = VectorIndex.load(index_dir)
        self.embedder = Embedder(
            model_name=self.index.config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        self.retriever = Retriever(index=self.index, embedder=self.embedder, default_top_k=top_k)
        self.generator = generator or Generator()
        self.top_k = top_k

    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve context and generate an answer for a user question."""
        k = top_k or self.top_k
        chunks = self.retriever.retrieve(question, top_k=k)
        result = self.generator.generate(question, chunks)
        return result


__all__ = [
    "Chunk",
    "chunk_text",
    "chunk_file",
    "chunk_directory",
    "Embedder",
    "VectorIndex",
    "Retriever",
    "Generator",
    "RAGPipeline",
]
