"""Unit and integration tests for deterministic remediation executor (sandbox_scripts/remediate.py)."""

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from pipeline.chunker import Chunk
from pipeline.index import VectorIndex
from sandbox_scripts.remediate import execute_remediation, get_active_version


@pytest.fixture
def mock_corpus_and_indexes(tmp_path):
    """Fixture providing an isolated mock corpus and indexes root directory."""
    corpus_dir = tmp_path / "corpus" / "active"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "doc1.md").write_text("# Document 1\nFastAPI is a modern web framework for building APIs.", encoding="utf-8")
    (corpus_dir / "doc2.md").write_text("# Document 2\nBackground tasks run after returning a response.", encoding="utf-8")

    candidate_corpus = tmp_path / "corpus" / "candidate"

    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)

    # Establish baseline v001 index
    v001_dir = indexes_dir / "v001"
    v001_dir.mkdir()

    class MockEmbedder:
        model_name = "mock-model"
        dimension = 4
        def embed_texts(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)
        def embed_query(self, query):
            return np.ones((4,), dtype=np.float32)

    chunks = [
        Chunk(id="c1", content="Baseline chunk 1", metadata={"chunk_size": 500, "overlap": 100}),
        Chunk(id="c2", content="Baseline chunk 2", metadata={"chunk_size": 500, "overlap": 100}),
    ]
    VectorIndex.build(
        chunks=chunks,
        embedder=MockEmbedder(),
        save_dir=v001_dir,
        version="v001",
        chunk_size=500,
        overlap=100,
        top_k=3,
    )

    active_json = indexes_dir / "active.json"
    active_json.write_text(json.dumps({"active_version": "v001", "baseline_version": "v001"}))

    return {
        "corpus_dir": corpus_dir,
        "candidate_corpus": candidate_corpus,
        "indexes_dir": indexes_dir,
        "v001_dir": v001_dir,
        "active_json": active_json,
    }


def test_remediate_chunking_strategy_success(mock_corpus_and_indexes, monkeypatch):
    """Verify successful candidate index creation with chunking remediation strategy."""
    env = mock_corpus_and_indexes

    class MockEmbedder:
        model_name = "all-MiniLM-L6-v2"
        dimension = 4
        def __init__(self, model_name=None):
            self.model_name = model_name or "mock-model"
        def embed_texts(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)
        def embed_query(self, query):
            return np.ones((4,), dtype=np.float32)

    monkeypatch.setattr("sandbox_scripts.remediate.Embedder", MockEmbedder)

    result = execute_remediation(
        strategy="chunking",
        source_index="v001",
        output_version="exp_chunk_001",
        chunk_size=150,
        overlap=30,
        corpus_source_dir=env["corpus_dir"],
        corpus_working_dir=env["candidate_corpus"],
        indexes_root_dir=env["indexes_dir"],
    )

    assert result["status"] == "SUCCESS"
    assert result["strategy"] == "chunking"
    assert result["candidate_version"] == "exp_chunk_001"
    assert result["total_chunks"] > 0

    cand_dir = env["indexes_dir"] / "exp_chunk_001"
    assert (cand_dir / "index.faiss").exists()
    assert (cand_dir / "chunks.json").exists()
    assert (cand_dir / "config.json").exists()
    assert (cand_dir / "remediation_result.json").exists()

    loaded = VectorIndex.load(cand_dir)
    assert loaded.config["chunk_size"] == 150
    assert loaded.config["overlap"] == 30
    assert loaded.config["strategy"] == "chunking"


def test_remediate_retrieval_strategy_success(mock_corpus_and_indexes, monkeypatch):
    """Verify successful candidate index creation with retrieval remediation strategy."""
    env = mock_corpus_and_indexes

    class MockEmbedder:
        model_name = "all-MiniLM-L6-v2"
        dimension = 4
        def __init__(self, model_name=None):
            self.model_name = model_name or "mock-model"
        def embed_texts(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)
        def embed_query(self, query):
            return np.ones((4,), dtype=np.float32)

    monkeypatch.setattr("sandbox_scripts.remediate.Embedder", MockEmbedder)

    result = execute_remediation(
        strategy="retrieval",
        source_index="v001",
        output_version="exp_ret_001",
        top_k=5,
        corpus_source_dir=env["corpus_dir"],
        corpus_working_dir=env["candidate_corpus"],
        indexes_root_dir=env["indexes_dir"],
    )

    assert result["status"] == "SUCCESS"
    assert result["strategy"] == "retrieval"
    assert result["candidate_version"] == "exp_ret_001"

    cand_dir = env["indexes_dir"] / "exp_ret_001"
    loaded = VectorIndex.load(cand_dir)
    assert loaded.config["top_k"] == 5
    assert loaded.config["strategy"] == "retrieval"


def test_remediate_active_index_isolation_and_protection(mock_corpus_and_indexes, monkeypatch):
    """Verify that candidate creation never mutates active.json or baseline index v001."""
    env = mock_corpus_and_indexes

    class MockEmbedder:
        model_name = "all-MiniLM-L6-v2"
        dimension = 4
        def __init__(self, model_name=None):
            self.model_name = model_name or "mock-model"
        def embed_texts(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("sandbox_scripts.remediate.Embedder", MockEmbedder)

    v001_config_before = (env["v001_dir"] / "config.json").read_text(encoding="utf-8")
    v001_chunks_before = (env["v001_dir"] / "chunks.json").read_text(encoding="utf-8")
    v001_faiss_before = (env["v001_dir"] / "index.faiss").read_bytes()

    # 1. Attempting to overwrite baseline v001 must raise ValueError
    with pytest.raises(ValueError, match="Cannot overwrite baseline version 'v001'"):
        execute_remediation(
            strategy="chunking",
            output_version="v001",
            chunk_size=300,
            overlap=50,
            corpus_source_dir=env["corpus_dir"],
            corpus_working_dir=env["candidate_corpus"],
            indexes_root_dir=env["indexes_dir"],
        )

    # 2. Set active version to a custom version 'v002_custom' and test active overwrite protection
    env["active_json"].write_text(json.dumps({"active_version": "v002_custom", "baseline_version": "v001"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot overwrite currently active version 'v002_custom'"):
        execute_remediation(
            strategy="chunking",
            source_index="v001",
            output_version="v002_custom",
            chunk_size=300,
            overlap=50,
            corpus_source_dir=env["corpus_dir"],
            corpus_working_dir=env["candidate_corpus"],
            indexes_root_dir=env["indexes_dir"],
        )

    # 3. Valid candidate execution preserves active state exactly
    # Reset active to v001
    env["active_json"].write_text(json.dumps({"active_version": "v001", "baseline_version": "v001"}), encoding="utf-8")
    execute_remediation(
        strategy="chunking",
        output_version="exp_iso_001",
        chunk_size=300,
        overlap=50,
        corpus_source_dir=env["corpus_dir"],
        corpus_working_dir=env["candidate_corpus"],
        indexes_root_dir=env["indexes_dir"],
    )

    # Assert active.json and v001 contents are 100% untouched
    active_data = json.loads(env["active_json"].read_text(encoding="utf-8"))
    assert active_data["active_version"] == "v001"

    assert (env["v001_dir"] / "config.json").read_text(encoding="utf-8") == v001_config_before
    assert (env["v001_dir"] / "chunks.json").read_text(encoding="utf-8") == v001_chunks_before
    assert (env["v001_dir"] / "index.faiss").read_bytes() == v001_faiss_before


def test_remediate_invalid_parameters(mock_corpus_and_indexes):
    """Verify validation of illegal strategies and parameter values."""
    env = mock_corpus_and_indexes

    # Unsupported strategy
    with pytest.raises(ValueError, match="Invalid remediation strategy"):
        execute_remediation(
            strategy="unknown_strategy",
            output_version="exp_bad",
            corpus_source_dir=env["corpus_dir"],
            corpus_working_dir=env["candidate_corpus"],
            indexes_root_dir=env["indexes_dir"],
        )

    # Non-positive chunk size
    with pytest.raises(ValueError, match="positive integer"):
        execute_remediation(
            strategy="chunking",
            output_version="exp_bad",
            chunk_size=-50,
            overlap=10,
            corpus_source_dir=env["corpus_dir"],
            corpus_working_dir=env["candidate_corpus"],
            indexes_root_dir=env["indexes_dir"],
        )

    # Negative overlap
    with pytest.raises(ValueError, match="non-negative integer"):
        execute_remediation(
            strategy="chunking",
            output_version="exp_bad",
            chunk_size=300,
            overlap=-10,
            corpus_source_dir=env["corpus_dir"],
            corpus_working_dir=env["candidate_corpus"],
            indexes_root_dir=env["indexes_dir"],
        )

    # Invalid geometry (overlap >= chunk_size)
    with pytest.raises(ValueError, match="Invalid chunk geometry"):
        execute_remediation(
            strategy="chunking",
            output_version="exp_bad",
            chunk_size=100,
            overlap=100,
            corpus_source_dir=env["corpus_dir"],
            corpus_working_dir=env["candidate_corpus"],
            indexes_root_dir=env["indexes_dir"],
        )

    # Non-positive top_k
    with pytest.raises(ValueError, match="positive integer"):
        execute_remediation(
            strategy="retrieval",
            output_version="exp_bad",
            top_k=0,
            corpus_source_dir=env["corpus_dir"],
            corpus_working_dir=env["candidate_corpus"],
            indexes_root_dir=env["indexes_dir"],
        )


def test_remediate_cli_invocation(mock_corpus_and_indexes, monkeypatch):
    """Verify CLI invocation of sandbox_scripts/remediate.py via subprocess."""
    env = mock_corpus_and_indexes

    # Test CLI execution for chunking strategy
    cmd = [
        sys.executable,
        "sandbox_scripts/remediate.py",
        "--strategy", "chunking",
        "--source-index", "v001",
        "--output-version", "cli_cand_01",
        "--chunk-size", "200",
        "--overlap", "40",
        "--corpus-dir", str(env["corpus_dir"]),
        "--candidate-corpus-dir", str(env["candidate_corpus"]),
        "--indexes-dir", str(env["indexes_dir"]),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    payload = json.loads(res.stdout)
    assert payload["status"] == "SUCCESS"
    assert payload["candidate_version"] == "cli_cand_01"
    assert (env["indexes_dir"] / "cli_cand_01" / "index.faiss").exists()

    # Test CLI failure on invalid parameters
    bad_cmd = [
        sys.executable,
        "sandbox_scripts/remediate.py",
        "--strategy", "chunking",
        "--output-version", "v001",  # Overwrite attempt
        "--chunk-size", "200",
        "--overlap", "40",
        "--corpus-dir", str(env["corpus_dir"]),
        "--candidate-corpus-dir", str(env["candidate_corpus"]),
        "--indexes-dir", str(env["indexes_dir"]),
    ]
    bad_res = subprocess.run(bad_cmd, capture_output=True, text=True)
    assert bad_res.returncode == 1
    bad_payload = json.loads(bad_res.stderr)
    assert bad_payload["status"] == "FAILED"
    assert "Cannot overwrite baseline version" in bad_payload["error"]

