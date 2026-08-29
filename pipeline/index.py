"""FAISS vector index management and persistence for RAG Doctor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pipeline.chunker import Chunk
from pipeline.embedder import Embedder


class VectorIndex:
    """FAISS-backed vector index supporting versioning, serialization, and retrieval."""

    def __init__(
        self,
        index: Any,
        chunks: List[Chunk],
        config: Dict[str, Any],
        index_dir: Optional[Path] = None,
    ):
        self.index = index
        self.chunks = chunks
        self.config = config
        self.index_dir = Path(index_dir) if index_dir else None

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        embedder: Embedder,
        save_dir: str | Path,
        version: str = "v001",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        top_k: Optional[int] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> VectorIndex:
        """Build FAISS index from chunks, compute embeddings, and persist to disk.

        Args:
            chunks: List of Chunk objects to index.
            embedder: Embedder instance to generate embeddings.
            save_dir: Directory where index, chunks, and config will be saved.
            version: Version tag (e.g. 'v001', 'candidate').
            chunk_size: Recorded chunk size.
            overlap: Recorded chunk overlap.
            top_k: Recorded default retrieval top_k.
            extra_config: Additional metadata to record in config.json.
        """
        import faiss

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if not chunks:
            raise ValueError("Cannot build index with empty chunks list")

        # Extract texts and compute embeddings
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)
        dimension = embeddings.shape[1]

        # Use Inner Product index (since embeddings are normalized, IP == Cosine similarity)
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)

        unique_docs = len({c.metadata.get("doc_id", c.metadata.get("source", "")) for c in chunks})

        config: Dict[str, Any] = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_size": chunk_size if chunk_size is not None else chunks[0].metadata.get("chunk_size"),
            "overlap": overlap if overlap is not None else chunks[0].metadata.get("overlap"),
        }
        if top_k is not None:
            config["top_k"] = top_k
        elif extra_config and "top_k" in extra_config:
            config["top_k"] = extra_config["top_k"]

        config.update({
            "embedding_model": embedder.model_name,
            "dimension": dimension,
            "metric": "inner_product_normalized_cosine",
            "total_chunks": len(chunks),
            "total_documents": unique_docs,
            **(extra_config or {}),
        })

        instance = cls(
            index=faiss_index,
            chunks=chunks,
            config=config,
            index_dir=save_path,
        )
        instance.save(save_path)
        return instance

    def save(self, target_dir: str | Path) -> None:
        """Persist FAISS index binary, chunks data, and config to directory."""
        import faiss

        dir_path = Path(target_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 1. Save FAISS index binary
        faiss.write_index(self.index, str(dir_path / "index.faiss"))

        # 2. Save chunks JSON
        chunks_data = [c.to_dict() for c in self.chunks]
        with open(dir_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # 3. Save config JSON
        with open(dir_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        self.index_dir = dir_path

    @classmethod
    def load(cls, load_dir: str | Path) -> VectorIndex:
        """Load an existing index from disk."""
        import faiss

        dir_path = Path(load_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Index directory not found: {load_dir}")

        index_file = dir_path / "index.faiss"
        chunks_file = dir_path / "chunks.json"
        config_file = dir_path / "config.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file missing: {index_file}")
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file missing: {chunks_file}")
        if not config_file.exists():
            raise FileNotFoundError(f"Config file missing: {config_file}")

        faiss_index = faiss.read_index(str(index_file))

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        chunks = [Chunk.from_dict(d) for d in chunks_data]

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(
            index=faiss_index,
            chunks=chunks,
            config=config,
            index_dir=dir_path,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Search top-k most similar chunks for a query vector.

        Args:
            query_vector: 1D or 2D float32 numpy array.
            top_k: Number of nearest chunks to retrieve.

        Returns:
            List of tuples (Chunk, similarity_score) sorted by descending score.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results
