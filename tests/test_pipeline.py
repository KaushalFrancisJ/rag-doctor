"""Unit and integration tests for RAG pipeline components."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from pipeline.chunker import Chunk, chunk_directory, chunk_file, chunk_text
from pipeline.embedder import Embedder
from pipeline.generator import Generator
from pipeline.index import VectorIndex
from pipeline.retriever import Retriever


def test_chunk_text_basic():
    text = "Hello world this is a test of the chunking mechanism in RAG Doctor."
    chunks = chunk_text(text, chunk_size=30, overlap=10, doc_id="test_doc")
    assert len(chunks) > 1
    assert chunks[0].metadata["doc_id"] == "test_doc"
    assert chunks[0].metadata["chunk_size"] == 30
    assert chunks[0].metadata["overlap"] == 10
    assert chunks[0].id == "test_doc#chunk_0000"


def test_chunk_text_parameterization():
    text = "A" * 100
    chunks_small = chunk_text(text, chunk_size=20, overlap=5)
    chunks_large = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks_small) > len(chunks_large)


def test_chunk_text_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=50, overlap=-1)


def test_chunk_file_and_directory(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    doc1 = tmp_path / "doc1.md"
    doc2 = sub / "doc2.md"
    doc1.write_text("# Doc 1\nFastAPI is a modern, fast web framework for building APIs.", encoding="utf-8")
    doc2.write_text("# Doc 2\nBackground tasks can run after returning a response.", encoding="utf-8")

    chunks1 = chunk_file(doc1, chunk_size=50, overlap=10, base_dir=tmp_path)
    assert len(chunks1) >= 1
    assert chunks1[0].metadata["relative_path"] == "doc1.md"

    all_chunks = chunk_directory(tmp_path, chunk_size=50, overlap=10)
    assert len(all_chunks) >= 2
    relative_paths = {c.metadata["relative_path"] for c in all_chunks}
    assert "doc1.md" in relative_paths
    assert "sub/doc2.md" in relative_paths or "sub\\doc2.md" in relative_paths


def test_vector_index_build_save_load(tmp_path: Path):
    class MockEmbedder:
        model_name = "mock-model"
        def embed_texts(self, texts, **kwargs):
            # return deterministic normalized vectors
            vecs = np.zeros((len(texts), 4), dtype=np.float32)
            for i in range(len(texts)):
                vecs[i, i % 4] = 1.0
            return vecs
        def embed_query(self, query):
            v = np.zeros((4,), dtype=np.float32)
            v[0] = 1.0
            return v
        @property
        def dimension(self):
            return 4

    chunks = [
        Chunk(id="c1", content="FastAPI middleware handling", metadata={"source": "middleware.md"}),
        Chunk(id="c2", content="FastAPI dependency injection", metadata={"source": "deps.md"}),
    ]

    index_dir = tmp_path / "v001"
    index = VectorIndex.build(
        chunks=chunks,
        embedder=MockEmbedder(),
        save_dir=index_dir,
        version="v001",
        chunk_size=300,
        overlap=50,
    )

    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "config.json").exists()

    with open(index_dir / "config.json") as f:
        config = json.load(f)
    assert config["version"] == "v001"
    assert config["chunk_size"] == 300
    assert config["overlap"] == 50
    assert config["embedding_model"] == "mock-model"
    assert config["total_chunks"] == 2

    # Load back
    loaded_index = VectorIndex.load(index_dir)
    assert len(loaded_index.chunks) == 2
    assert loaded_index.config["version"] == "v001"

    # Search
    retriever = Retriever(index=loaded_index, embedder=MockEmbedder(), default_top_k=2)
    results = retriever.retrieve("middleware query", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"


def test_candidate_isolation(tmp_path: Path):
    """Verify that building a candidate index does not alter active index files."""
    active_dir = tmp_path / "indexes" / "active"
    candidate_dir = tmp_path / "indexes" / "candidate"

    class MockEmbedder:
        model_name = "mock-model"
        def embed_texts(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

    active_chunks = [Chunk(id="active_1", content="Active content", metadata={})]
    active_index = VectorIndex.build(
        chunks=active_chunks,
        embedder=MockEmbedder(),
        save_dir=active_dir,
        version="active_v001",
    )

    active_config_before = (active_dir / "config.json").read_text()

    candidate_chunks = [
        Chunk(id="candidate_1", content="Candidate content 1", metadata={}),
        Chunk(id="candidate_2", content="Candidate content 2", metadata={}),
    ]
    candidate_index = VectorIndex.build(
        chunks=candidate_chunks,
        embedder=MockEmbedder(),
        save_dir=candidate_dir,
        version="candidate_exp01",
    )

    active_config_after = (active_dir / "config.json").read_text()
    assert active_config_before == active_config_after
    assert len(VectorIndex.load(active_dir).chunks) == 1
    assert len(VectorIndex.load(candidate_dir).chunks) == 2


def test_generator_prompt_building():
    generator = Generator(api_key="mock_key")
    chunks = [
        {"content": "FastAPI is fast.", "metadata": {"relative_path": "features.md"}},
        {"content": "Dependencies use Depends.", "metadata": {"relative_path": "dependencies.md"}},
    ]
    prompt = generator.build_context_prompt("What is FastAPI?", chunks)
    assert "What is FastAPI?" in prompt
    assert "FastAPI is fast." in prompt
    assert "features.md" in prompt
    assert "dependencies.md" in prompt


def test_retriever_top_k_from_config(tmp_path: Path):
    class MockEmbedder:
        model_name = "mock-model"
        def embed_texts(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)
        def embed_query(self, query):
            return np.ones((4,), dtype=np.float32)
        @property
        def dimension(self):
            return 4

    chunks = [
        Chunk(id="c1", content="Chunk 1", metadata={}),
        Chunk(id="c2", content="Chunk 2", metadata={}),
        Chunk(id="c3", content="Chunk 3", metadata={}),
    ]

    index_dir = tmp_path / "top_k_test"
    index = VectorIndex.build(
        chunks=chunks,
        embedder=MockEmbedder(),
        save_dir=index_dir,
        version="v_test_k",
        top_k=1,
    )

    loaded_index = VectorIndex.load(index_dir)
    assert loaded_index.config["top_k"] == 1

    retriever = Retriever(index=loaded_index, embedder=MockEmbedder())
    assert retriever.default_top_k == 1

    results = retriever.retrieve("query")
    assert len(results) == 1


def test_mcp_tools_v003_retrieval_sick():
    from mcp_server import inspect_rag_health, get_pipeline_config, get_evaluation_results, get_failed_queries

    # Inspect health with explicit version
    health_raw = inspect_rag_health("v003_retrieval_sick")
    health = json.loads(health_raw)
    assert health["active_monitored_index"] == "v003_retrieval_sick"
    assert health["status"] == "DEGRADED"
    assert health["failing_queries_count"] > 0

    # Config
    cfg_raw = get_pipeline_config("v003_retrieval_sick")
    cfg = json.loads(cfg_raw)
    assert cfg["version"] == "v003_retrieval_sick"
    assert cfg["top_k"] == 1
    assert cfg["chunk_size"] == 500
    assert cfg["overlap"] == 100

    # Eval results
    eval_raw = get_evaluation_results("v003_retrieval_sick")
    eval_summary = json.loads(eval_raw)
    assert eval_summary["index_version"] == "v003_retrieval_sick"
    assert eval_summary["is_degraded"] is True

    # Failed queries
    failed_raw = get_failed_queries("v003_retrieval_sick")
    failed = json.loads(failed_raw)
    assert len(failed) > 0
    assert all(len(q["retrieved_chunks"]) == 1 for q in failed)


def test_mcp_generic_version_resolution(tmp_path: Path, monkeypatch):
    """Verify that arbitrary versions (e.g. 'exp_001') resolve generically without special-case branches."""
    from mcp_server import (
        inspect_rag_health,
        get_pipeline_config,
        get_evaluation_results,
        get_failed_queries,
        inspect_retrieval,
        get_active_version,
        get_baseline_version,
    )
    import mcp_server

    # Create dummy arbitrary index version
    exp_version = "exp_001"
    exp_index_dir = Path("indexes") / exp_version
    exp_index_dir.mkdir(parents=True, exist_ok=True)

    exp_eval_file = Path("eval") / f"{exp_version}_results.json"
    exp_active_file = Path("indexes") / "active.json"

    # Backup original active.json if exists
    original_active = exp_active_file.read_text(encoding="utf-8") if exp_active_file.exists() else None

    try:
        # 1. Create exp_001 config & index
        class MockEmbedder:
            model_name = "all-MiniLM-L6-v2"
            def embed_texts(self, texts, **kwargs):
                return np.ones((len(texts), 384), dtype=np.float32)
            def embed_query(self, query):
                return np.ones((384,), dtype=np.float32)
            @property
            def dimension(self):
                return 384

        chunks = [Chunk(id="exp_c1", content="Arbitrary experiment content", metadata={"source": "exp.md"})]
        VectorIndex.build(
            chunks=chunks,
            embedder=MockEmbedder(),
            save_dir=exp_index_dir,
            version=exp_version,
            chunk_size=420,
            overlap=42,
            top_k=2,
        )

        # 2. Create exp_001 eval results
        eval_payload = {
            "summary": {
                "index_version": exp_version,
                "evaluated_at": "2026-08-29T12:00:00Z",
                "total_queries": 1,
                "avg_faithfulness": 4.0,
                "avg_answer_relevancy": 4.0,
                "avg_overall_score": 4.0,
                "threshold": 3.5,
                "passing_count": 0,
                "failing_count": 1,
                "is_degraded": True,
            },
            "failing_queries": [
                {
                    "id": "exp_q1",
                    "question": "What is exp?",
                    "expected_answer": "Exp answer",
                    "generated_answer": "Generated exp",
                    "retrieved_chunks": [{"chunk_id": "exp_c1", "content": "Arbitrary experiment content", "metadata": {}}],
                    "faithfulness": 4,
                    "answer_relevancy": 4,
                    "overall_score": 4.0,
                    "passed": False,
                }
            ],
            "results": [],
        }
        with open(exp_eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2)

        # 3. Test generic resolution for get_pipeline_config
        cfg = json.loads(get_pipeline_config(exp_version))
        assert cfg["version"] == exp_version
        assert cfg["chunk_size"] == 420
        assert cfg["overlap"] == 42
        assert cfg["top_k"] == 2

        # 4. Test generic resolution for get_evaluation_results
        eval_res = json.loads(get_evaluation_results(exp_version))
        assert eval_res["index_version"] == exp_version
        assert eval_res["is_degraded"] is True

        # 5. Test generic resolution for get_failed_queries
        failed = json.loads(get_failed_queries(exp_version))
        assert len(failed) == 1
        assert failed[0]["id"] == "exp_q1"

        # 6. Test generic resolution for inspect_retrieval
        retrieved = json.loads(inspect_retrieval("test query", index_version=exp_version))
        assert len(retrieved) == 1
        assert retrieved[0]["chunk_id"] == "exp_c1"

        # 7. Test generic resolution for inspect_rag_health with explicit version
        health = json.loads(inspect_rag_health(exp_version))
        assert health["active_monitored_index"] == exp_version
        assert health["status"] == "DEGRADED"
        assert exp_version in health["available_index_versions"]

        # 8. Test active.json resolution when active_version is set to exp_version
        with open(exp_active_file, "w", encoding="utf-8") as f:
            json.dump({"active_version": exp_version, "baseline_version": "v001"}, f)

        assert get_active_version() == exp_version
        assert get_baseline_version() == "v001"

        # Default inspect_rag_health() should pick up active_version from active.json
        active_health = json.loads(inspect_rag_health())
        assert active_health["active_monitored_index"] == exp_version
        assert active_health["status"] == "DEGRADED"

    finally:
        # Cleanup arbitrary test artifacts
        if exp_eval_file.exists():
            exp_eval_file.unlink()
        if exp_index_dir.exists():
            import shutil
            shutil.rmtree(exp_index_dir)
        if original_active is not None:
            exp_active_file.write_text(original_active, encoding="utf-8")


