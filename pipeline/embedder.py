"""Embedding generation module using Sentence Transformers."""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class Embedder:
    """Wrapper around Sentence Transformers for generating vector embeddings."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        """Initialize the Embedder.

        Args:
            model_name: HuggingFace model identifier for SentenceTransformer.
            device: Computing device ('cpu', 'cuda', etc.). If None, auto-detected.
            normalize_embeddings: If True, embeddings are L2 normalized (for cosine similarity).
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    @property
    def model(self):
        """Lazy-loaded SentenceTransformer model instance."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        """Generate vector embeddings for a list of texts.

        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to display progress bar.

        Returns:
            2D numpy array of shape (len(texts), embedding_dimension) with float32 dtype.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a single vector embedding for a query string.

        Args:
            query: Query string.

        Returns:
            1D numpy array of shape (embedding_dimension,) with float32 dtype.
        """
        emb = self.embed_texts([query], show_progress_bar=False)
        return emb[0]

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
