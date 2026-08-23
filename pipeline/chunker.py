"""Document chunking module for RAG Doctor.

Provides parameterized text and document chunking functions with metadata preservation.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Chunk:
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    doc_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """Split a string into overlapping chunks.
    
    Args:
        text: The text to chunk.
        chunk_size: The target character length of each chunk.
        overlap: The character overlap between consecutive chunks.
        doc_id: Identifier for the parent document.
        metadata: Base metadata to attach to all resulting chunks.
        
    Returns:
        A list of Chunk objects with position metadata.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")

    base_metadata = metadata.copy() if metadata else {}
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    total_len = len(cleaned_text)
    
    start = 0
    chunk_idx = 0

    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk_content = cleaned_text[start:end].strip()

        if chunk_content:
            chunk_metadata = {
                **base_metadata,
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "start_char": start,
                "end_char": end,
                "chunk_size": chunk_size,
                "overlap": overlap,
            }
            chunk_id = f"{doc_id}#chunk_{chunk_idx:04d}" if doc_id else f"chunk_{chunk_idx:04d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=chunk_content,
                    metadata=chunk_metadata,
                )
            )
            chunk_idx += 1

        if end >= total_len:
            break
        start += step

    return chunks


def chunk_file(
    file_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 100,
    base_dir: Optional[str | Path] = None,
) -> List[Chunk]:
    """Load and chunk a single file.
    
    Args:
        file_path: Path to the markdown or text file.
        chunk_size: Target character length per chunk.
        overlap: Character overlap.
        base_dir: Optional base directory to compute relative source path.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    rel_path = str(path.relative_to(base_dir)) if base_dir else path.name
    doc_id = rel_path.replace(os.sep, "/")

    metadata = {
        "source": str(path),
        "relative_path": rel_path,
        "filename": path.name,
    }

    return chunk_text(
        text=content,
        chunk_size=chunk_size,
        overlap=overlap,
        doc_id=doc_id,
        metadata=metadata,
    )


def chunk_directory(
    directory_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 100,
    extensions: tuple[str, ...] = (".md", ".txt"),
) -> List[Chunk]:
    """Recursively load and chunk all supported documents in a directory.
    
    Args:
        directory_path: Root directory to read documents from.
        chunk_size: Target character length per chunk.
        overlap: Character overlap between consecutive chunks.
        extensions: File extensions to include.
        
    Returns:
        A list of Chunk objects representing the entire corpus.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory_path}")

    all_chunks: List[Chunk] = []
    # Collect files deterministically sorted
    files = sorted([p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in extensions])

    for file_p in files:
        file_chunks = chunk_file(
            file_path=file_p,
            chunk_size=chunk_size,
            overlap=overlap,
            base_dir=dir_path,
        )
        all_chunks.extend(file_chunks)

    return all_chunks
