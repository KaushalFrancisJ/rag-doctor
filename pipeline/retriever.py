"""Retriever module for semantic similarity lookup over the vector index."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pipeline.embedder import Embedder
from pipeline.index import VectorIndex


class Retriever:
    """Performs semantic similarity retrieval using an Embedder and VectorIndex."""

    def __init__(
        self,
        index: VectorIndex,
        embedder: Optional[Embedder] = None,
        default_top_k: Optional[int] = None,
    ):
        self.index = index
        self.embedder = embedder or Embedder(
            model_name=index.config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        self.default_top_k = default_top_k if default_top_k is not None else int(index.config.get("top_k", 3))

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query string.

        Args:
            query: User question or search query.
            top_k: Number of chunks to retrieve (defaults to default_top_k).

        Returns:
            List of dictionaries with keys: chunk_id, content, score, metadata.
        """
        k = top_k if top_k is not None else self.default_top_k
        query_vector = self.embedder.embed_query(query)
        search_results = self.index.search(query_vector, top_k=k)

        formatted_results: List[Dict[str, Any]] = []
        for chunk, score in search_results:
            formatted_results.append({
                "chunk_id": chunk.id,
                "content": chunk.content,
                "score": score,
                "metadata": chunk.metadata,
            })

        return formatted_results
