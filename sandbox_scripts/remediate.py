"""Deterministic remediation executor for RAG Doctor Fix workflow.

Executes candidate experiment specifications inside an isolated sandbox environment:
- Supports 'chunking' (chunk_size, overlap) and 'retrieval' (top_k) strategies.
- Operates on copies of the corpus and preserves active index immutability.
- Creates isolated FAISS index and configuration for the candidate version.
- Emits structured JSON results for downstream candidate evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.chunker import chunk_directory
from pipeline.embedder import Embedder
from pipeline.index import VectorIndex

logger = logging.getLogger(__name__)

SUPPORTED_STRATEGIES = ["chunking", "retrieval"]
FORBIDDEN_OVERWRITE_VERSIONS = {"v001"}


def get_active_version(indexes_dir: Path) -> str:
    """Read current active version from active.json pointer if present."""
    active_json = indexes_dir / "active.json"
    if active_json.exists():
        try:
            data = json.loads(active_json.read_text(encoding="utf-8"))
            return data.get("active_version", "v001")
        except Exception:
            return "v001"
    return "v001"


def execute_remediation(
    strategy: str,
    source_index: Optional[str] = None,
    output_version: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    top_k: Optional[int] = None,
    corpus_source_dir: str | Path = "corpus/active",
    corpus_working_dir: str | Path = "corpus/candidate",
    indexes_root_dir: str | Path = "indexes",
    embedding_model: str = "all-MiniLM-L6-v2",
    result_file: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute deterministic remediation to build an isolated candidate index.

    Args:
        strategy: 'chunking' or 'retrieval'.
        source_index: Version of starting index (defaults to active).
        output_version: Version tag for new candidate index.
        chunk_size: Chunk size in characters for chunking strategy.
        overlap: Chunk overlap in characters for chunking strategy.
        top_k: Default retrieval count for retrieval strategy.
        corpus_source_dir: Path to source corpus documents.
        corpus_working_dir: Path to working candidate corpus copy.
        indexes_root_dir: Path to indexes root folder.
        embedding_model: SentenceTransformer model name.
        result_file: Optional path to write machine-readable result JSON.

    Returns:
        Structured result dictionary describing the built candidate index.

    Raises:
        ValueError: On invalid parameters, geometry, or attempted overwrite of active/baseline.
        FileNotFoundError: If corpus source directory does not exist.
    """
    strat = str(strategy).strip().lower()
    if strat not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Invalid remediation strategy '{strategy}'. Supported strategies: {SUPPORTED_STRATEGIES}"
        )

    indexes_path = Path(indexes_root_dir)
    active_ver = get_active_version(indexes_path)
    src_ver = source_index or active_ver

    # 1. Load source index configuration if available
    source_config: Dict[str, Any] = {}
    src_config_file = indexes_path / src_ver / "config.json"
    if src_config_file.exists():
        try:
            source_config = json.loads(src_config_file.read_text(encoding="utf-8"))
        except Exception:
            source_config = {}

    # 2. Resolve and protect output version
    if output_version:
        cand_version = str(output_version).strip()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cand_version = f"candidate_{timestamp}"

    if cand_version in FORBIDDEN_OVERWRITE_VERSIONS:
        raise ValueError(
            f"Cannot overwrite baseline version '{cand_version}' with a candidate experiment."
        )

    if cand_version == active_ver:
        raise ValueError(
            f"Cannot overwrite currently active version '{active_ver}' with a candidate experiment. "
            f"Candidate indexes must have a distinct isolated version."
        )

    cand_dir = indexes_path / cand_version
    if cand_dir.exists() and cand_dir.resolve() == (indexes_path / src_ver).resolve():
        raise ValueError(
            f"Target candidate directory '{cand_dir}' conflicts with source index '{src_ver}'."
        )

    # 3. Resolve and validate strategy parameters
    if strat == "chunking":
        eff_chunk_size = chunk_size if chunk_size is not None else source_config.get("chunk_size", 500)
        eff_overlap = overlap if overlap is not None else source_config.get("overlap", 100)
        eff_top_k = top_k if top_k is not None else source_config.get("top_k", 3)

        if not isinstance(eff_chunk_size, int) or eff_chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {eff_chunk_size}")
        if not isinstance(eff_overlap, int) or eff_overlap < 0:
            raise ValueError(f"overlap must be a non-negative integer, got {eff_overlap}")
        if eff_overlap >= eff_chunk_size:
            raise ValueError(
                f"Invalid chunk geometry: overlap ({eff_overlap}) must be strictly less than chunk_size ({eff_chunk_size})."
            )
        if not isinstance(eff_top_k, int) or eff_top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {eff_top_k}")

    elif strat == "retrieval":
        eff_top_k = top_k if top_k is not None else source_config.get("top_k", 3)
        eff_chunk_size = chunk_size if chunk_size is not None else source_config.get("chunk_size", 500)
        eff_overlap = overlap if overlap is not None else source_config.get("overlap", 100)

        if not isinstance(eff_top_k, int) or eff_top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {eff_top_k}")
        if not isinstance(eff_chunk_size, int) or eff_chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {eff_chunk_size}")
        if not isinstance(eff_overlap, int) or eff_overlap < 0 or eff_overlap >= eff_chunk_size:
            raise ValueError(
                f"Invalid chunk geometry: overlap ({eff_overlap}) must be strictly less than chunk_size ({eff_chunk_size})."
            )
    else:
        raise ValueError(f"Unsupported strategy: {strat}")

    # 4. Copy corpus to candidate working directory (Requirement 4)
    src_corpus_path = Path(corpus_source_dir)
    work_corpus_path = Path(corpus_working_dir)

    if not src_corpus_path.exists():
        raise FileNotFoundError(f"Source corpus directory not found: {src_corpus_path}")

    if work_corpus_path.exists():
        shutil.rmtree(work_corpus_path)
    shutil.copytree(src_corpus_path, work_corpus_path)

    # 5. Chunk documents from working corpus
    chunks = chunk_directory(
        directory_path=work_corpus_path,
        chunk_size=eff_chunk_size,
        overlap=eff_overlap,
    )
    if not chunks:
        raise ValueError(f"No chunks generated from corpus at {work_corpus_path}")

    # 6. Initialize local embedder (no external API calls)
    embedder = Embedder(model_name=embedding_model)

    # 7. Build FAISS index in isolated candidate directory
    cand_dir.mkdir(parents=True, exist_ok=True)
    index = VectorIndex.build(
        chunks=chunks,
        embedder=embedder,
        save_dir=cand_dir,
        version=cand_version,
        chunk_size=eff_chunk_size,
        overlap=eff_overlap,
        top_k=eff_top_k,
        extra_config={
            "source_version": src_ver,
            "strategy": strat,
            "remediated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    # 8. Verify active.json immutability
    post_active_ver = get_active_version(indexes_path)
    if post_active_ver != active_ver:
        raise RuntimeError(
            f"Active index mutation detected! Active version changed from '{active_ver}' to '{post_active_ver}'."
        )

    # 9. Format machine-readable result payload
    result = {
        "status": "SUCCESS",
        "strategy": strat,
        "source_index": src_ver,
        "candidate_version": cand_version,
        "candidate_index_dir": str(cand_dir),
        "config": index.config,
        "total_chunks": len(chunks),
        "total_documents": index.config.get("total_documents", 0),
        "dimension": index.config.get("dimension", embedder.dimension),
    }

    # Save remediation result inside candidate folder
    res_path = cand_dir / "remediation_result.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if result_file:
        out_res = Path(result_file)
        out_res.parent.mkdir(parents=True, exist_ok=True)
        with open(out_res, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic remediation executor for RAG Doctor candidate index creation."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=SUPPORTED_STRATEGIES,
        help="Remediation strategy ('chunking' or 'retrieval')",
    )
    parser.add_argument(
        "--source-index",
        type=str,
        default=None,
        help="Source index version to base candidate on (defaults to active index)",
    )
    parser.add_argument(
        "--output-version",
        type=str,
        default=None,
        help="Unique candidate/experiment version tag (e.g. exp_001)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in characters (for chunking strategy)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Chunk overlap in characters (for chunking strategy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval top-k chunks count (for retrieval strategy)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="corpus/active",
        help="Source corpus directory",
    )
    parser.add_argument(
        "--candidate-corpus-dir",
        type=str,
        default="corpus/candidate",
        help="Working candidate corpus directory",
    )
    parser.add_argument(
        "--indexes-dir",
        type=str,
        default="indexes",
        help="Indexes root directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        help="Optional path to write machine-readable result JSON",
    )

    args = parser.parse_args()

    try:
        result = execute_remediation(
            strategy=args.strategy,
            source_index=args.source_index,
            output_version=args.output_version,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            top_k=args.top_k,
            corpus_source_dir=args.corpus_dir,
            corpus_working_dir=args.candidate_corpus_dir,
            indexes_root_dir=args.indexes_dir,
            embedding_model=args.model_name,
            result_file=args.result_file,
        )
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        error_payload = {
            "status": "FAILED",
            "error": str(e),
            "strategy": args.strategy,
            "source_index": args.source_index,
            "output_version": args.output_version,
        }
        print(json.dumps(error_payload, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
