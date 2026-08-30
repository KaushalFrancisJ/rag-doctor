"""Unit and integration tests for candidate evaluation comparison and healing workflow."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from agent.orchestrator import run_healing_cycle
from eval.compare import (
    compare_evaluation_reports,
    evaluate_and_compare_candidate,
    identify_regressions,
    load_evaluation_report,
)
from eval.judge import Judge
from mcp_server import compare_experiments
from pipeline.chunker import Chunk
from pipeline.index import VectorIndex


def test_compare_evaluation_reports_exact_schema():
    """Verify that compare_evaluation_reports strictly returns the 7 required canonical keys."""
    candidate_report = {
        "summary": {
            "index_version": "cand_01",
            "avg_overall_score": 4.60,
            "overall_score": 5,
        },
        "results": [
            {"id": "q1", "overall_score": 5, "passed": True},
            {"id": "q2", "overall_score": 4, "passed": True},
        ],
    }

    current_report = {
        "summary": {
            "index_version": "v002_sick",
            "avg_overall_score": 3.80,
            "overall_score": 4,
        },
        "results": [
            {"id": "q1", "overall_score": 5, "passed": True},
            {"id": "q2", "overall_score": 2, "passed": False},
        ],
    }

    baseline_report = {
        "summary": {
            "index_version": "v001",
            "avg_overall_score": 4.50,
            "overall_score": 5,
        },
        "results": [
            {"id": "q1", "overall_score": 5, "passed": True},
            {"id": "q2", "overall_score": 4, "passed": True},
        ],
    }

    res = compare_evaluation_reports(
        candidate_report=candidate_report,
        current_report=current_report,
        baseline_report=baseline_report,
    )

    required_keys = {
        "baseline_score",
        "current_score",
        "candidate_score",
        "delta_vs_current",
        "delta_vs_baseline",
        "candidate_better",
        "regressions",
    }
    assert set(res.keys()) == required_keys

    assert res["baseline_score"] == 4.50
    assert res["current_score"] == 3.80
    assert res["candidate_score"] == 4.60
    assert res["delta_vs_current"] == 0.80  # 4.60 - 3.80
    assert res["delta_vs_baseline"] == 0.10  # 4.60 - 4.50
    assert res["candidate_better"] is True
    assert res["regressions"] == []


def test_compare_evaluation_reports_regressions_detection():
    """Verify regression detection when candidate scores lower than baseline or fails passing queries."""
    candidate_report = {
        "summary": {"avg_overall_score": 3.90},
        "results": [
            {"id": "q1", "overall_score": 5, "passed": True},
            {"id": "q2", "overall_score": 2, "passed": False},  # Regressed vs baseline q2 (was 4, passed)
            {"id": "q3", "overall_score": 3, "passed": False},  # Regressed vs baseline q3 (was 5, passed)
        ],
    }

    current_report = {
        "summary": {"avg_overall_score": 3.50},
        "results": [
            {"id": "q1", "overall_score": 5, "passed": True},
            {"id": "q2", "overall_score": 2, "passed": False},
            {"id": "q3", "overall_score": 4, "passed": True},
        ],
    }

    baseline_report = {
        "summary": {"avg_overall_score": 4.50},
        "results": [
            {"id": "q1", "overall_score": 5, "passed": True},
            {"id": "q2", "overall_score": 4, "passed": True},
            {"id": "q3", "overall_score": 5, "passed": True},
        ],
    }

    res = compare_evaluation_reports(
        candidate_report=candidate_report,
        current_report=current_report,
        baseline_report=baseline_report,
    )

    assert res["candidate_better"] is True  # 3.90 > 3.50
    assert res["delta_vs_current"] == 0.40
    assert res["delta_vs_baseline"] == -0.60
    assert res["regressions"] == ["q2", "q3"]


def test_compare_evaluation_candidate_not_better():
    """Verify candidate_better is False when candidate score <= current degraded score."""
    candidate_report = {
        "summary": {"avg_overall_score": 3.20},
        "results": [{"id": "q1", "overall_score": 3, "passed": False}],
    }
    current_report = {
        "summary": {"avg_overall_score": 3.50},
        "results": [{"id": "q1", "overall_score": 4, "passed": True}],
    }

    res = compare_evaluation_reports(candidate_report=candidate_report, current_report=current_report)
    assert res["candidate_better"] is False
    assert res["delta_vs_current"] == -0.30
    assert res["regressions"] == ["q1"]


def test_load_evaluation_report_resolution(tmp_path: Path):
    """Verify loading evaluation reports via dict, file path, version convention, and error cases."""
    # 1. Direct dict
    d = {"summary": {"avg_overall_score": 4.0}}
    assert load_evaluation_report(d) == d

    # 2. File path
    custom_file = tmp_path / "custom_results.json"
    custom_file.write_text(json.dumps(d), encoding="utf-8")
    loaded_file = load_evaluation_report(custom_file)
    assert loaded_file["summary"]["avg_overall_score"] == 4.0

    # 3. Version tag convention
    loaded_v001 = load_evaluation_report("v001")
    assert loaded_v001["summary"]["index_version"] == "v001"

    loaded_sick = load_evaluation_report("v002_sick")
    assert loaded_sick["summary"]["index_version"] == "v002_sick"

    # 4. Non-existent report raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Could not locate evaluation report"):
        load_evaluation_report("non_existent_version_999")


def test_evaluate_and_compare_candidate_mocked(tmp_path: Path):
    """Verify evaluate_and_compare_candidate evaluates candidate index and produces structured comparison."""
    cand_dir = tmp_path / "cand_index"
    cand_dir.mkdir()

    class MockEmbedder:
        model_name = "mock-embedder"
        dimension = 4
        def embed_texts(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)
        def embed_query(self, query):
            return np.ones((4,), dtype=np.float32)

    chunks = [Chunk(id="c1", content="FastAPI background task content", metadata={"chunk_size": 500})]
    VectorIndex.build(
        chunks=chunks,
        embedder=MockEmbedder(),
        save_dir=cand_dir,
        version="cand_test",
        top_k=3,
    )

    eval_set_file = tmp_path / "eval_set.json"
    eval_set_file.write_text(
        json.dumps([
            {
                "id": "eval_001",
                "question": "How to schedule background task?",
                "expected_answer": "Use BackgroundTasks parameter.",
                "source_doc": "background-tasks.md",
            }
        ]),
        encoding="utf-8",
    )

    judge = Judge(api_key="mock_key")
    def mock_eval_rag(retriever, generator, eval_set_path, threshold=3.5, output_path=None, top_k=None):
        return {
            "summary": {
                "index_version": "cand_test",
                "avg_overall_score": 4.80,
                "overall_score": 5,
                "is_degraded": False,
            },
            "results": [
                {
                    "id": "eval_001",
                    "overall_score": 5,
                    "passed": True,
                }
            ],
            "failing_queries": [],
        }

    judge.evaluate_rag = mock_eval_rag

    out_file = tmp_path / "cand_eval_out.json"
    res = evaluate_and_compare_candidate(
        candidate_index_dir=cand_dir,
        current_index_version="v002_sick",
        baseline_index_version="v001",
        eval_set_path=eval_set_file,
        output_report_path=out_file,
        judge=judge,
    )

    assert res["candidate_score"] == 4.80
    assert res["current_score"] == 3.83  # v002_sick score
    assert res["baseline_score"] == 4.58  # v001 baseline score
    assert res["delta_vs_current"] == round(4.80 - 3.83, 2)
    assert res["delta_vs_baseline"] == round(4.80 - 4.58, 2)
    assert res["candidate_better"] is True
    assert res["regressions"] == []


def test_mcp_compare_experiments_tool():
    """Verify that compare_experiments MCP tool returns structured comparison JSON."""
    res_raw = compare_experiments(
        candidate_version="v001",  # Candidate with good baseline score
        current_version="v003_retrieval_sick",  # Degraded current
        baseline_version="v001",
    )
    res = json.loads(res_raw)

    assert "candidate_score" in res
    assert "current_score" in res
    assert "baseline_score" in res
    assert "delta_vs_current" in res
    assert "delta_vs_baseline" in res
    assert "candidate_better" in res
    assert "regressions" in res
    assert res["candidate_better"] is True
    assert res["delta_vs_current"] > 0


def test_healing_workflow_full_cycle_immutability(tmp_path: Path, monkeypatch):
    """Verify complete healing loop: degraded -> diagnosis -> experiment -> sandbox -> candidate -> evaluation -> comparison.

    Asserts that active index is never modified and candidate is not automatically promoted.
    """
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)

    # Establish baseline v001
    v001_dir = indexes_dir / "v001"
    v001_dir.mkdir()

    class MockEmbedder:
        model_name = "mock-embedder"
        dimension = 4
        def __init__(self, model_name=None):
            self.model_name = model_name or "mock-model"
            self.dimension = 4
        def embed_texts(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)
        def embed_query(self, query):
            return np.ones((4,), dtype=np.float32)

    monkeypatch.setattr("sandbox_scripts.remediate.Embedder", MockEmbedder)

    chunks = [Chunk(id="c1", content="Baseline chunk content", metadata={"chunk_size": 500, "overlap": 100})]
    VectorIndex.build(
        chunks=chunks,
        embedder=MockEmbedder(),
        save_dir=v001_dir,
        version="v001",
        chunk_size=500,
        overlap=100,
        top_k=3,
    )

    # Establish active degraded v002_sick
    v002_dir = indexes_dir / "v002_sick"
    v002_dir.mkdir()
    sick_chunks = [Chunk(id="sc1", content="Sick chunk", metadata={"chunk_size": 70, "overlap": 0})]
    VectorIndex.build(
        chunks=sick_chunks,
        embedder=MockEmbedder(),
        save_dir=v002_dir,
        version="v002_sick",
        chunk_size=70,
        overlap=0,
        top_k=3,
    )

    active_json = indexes_dir / "active.json"
    active_json.write_text(json.dumps({"active_version": "v002_sick", "baseline_version": "v001"}), encoding="utf-8")

    active_json_snapshot = active_json.read_text(encoding="utf-8")
    v002_config_snapshot = (v002_dir / "config.json").read_text(encoding="utf-8")

    # Set up corpus
    corpus_dir = tmp_path / "corpus" / "active"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "doc.md").write_text("# Doc\nFastAPI is a modern web framework.", encoding="utf-8")
    cand_corpus = tmp_path / "corpus" / "candidate"

    # Mock judge
    judge = Judge(api_key="mock_key")
    def mock_eval_rag(retriever, generator, eval_set_path, threshold=3.5, output_path=None, top_k=None):
        return {
            "summary": {
                "index_version": retriever.index.config.get("version"),
                "avg_overall_score": 4.70,
                "overall_score": 5,
                "is_degraded": False,
            },
            "results": [{"id": "eval_001", "overall_score": 5, "passed": True}],
            "failing_queries": [],
        }
    judge.evaluate_rag = mock_eval_rag

    # Run full healing cycle with explicit overrides for deterministic test
    diagnosis_payload = {
        "suspected_cause": "chunking problem",
        "evidence": "Observed small chunk_size of 70 cutting off context.",
        "confidence": 0.95,
        "hypothesis": "Small chunks fragment text.",
        "recommended_experiment": "increase chunk_size to 500",
    }
    experiment_payload = {
        "hypothesis": "Restoring chunk size 500 recovers context.",
        "strategy": "chunking",
        "changes": {"chunk_size": 500, "overlap": 100},
        "expected_effect": "Improve answer grounding.",
        "reasoning": "Chunks were too small.",
    }

    result = run_healing_cycle(
        index_version="v002_sick",
        candidate_version="exp_healed_01",
        diagnosis_override=diagnosis_payload,
        experiment_override=experiment_payload,
        judge=judge,
        indexes_dir=indexes_dir,
        corpus_dir=corpus_dir,
        candidate_corpus_dir=cand_corpus,
    )

    # 1. Validate structured comparison output
    assert result["candidate_score"] == 4.70
    assert result["current_score"] == 3.83
    assert result["baseline_score"] == 4.58
    assert result["delta_vs_current"] == round(4.70 - 3.83, 2)
    assert result["delta_vs_baseline"] == round(4.70 - 4.58, 2)
    assert result["candidate_better"] is True
    assert result["regressions"] == []

    # 2. Validate candidate index created in isolated directory
    cand_index_dir = indexes_dir / "exp_healed_01"
    assert cand_index_dir.exists()
    assert (cand_index_dir / "index.faiss").exists()
    assert (cand_index_dir / "config.json").exists()

    cand_cfg = json.loads((cand_index_dir / "config.json").read_text(encoding="utf-8"))
    assert cand_cfg["chunk_size"] == 500
    assert cand_cfg["overlap"] == 100

    # 3. Assert Active Index Immutability (Candidate is NEVER promoted automatically)
    assert active_json.read_text(encoding="utf-8") == active_json_snapshot
    assert (v002_dir / "config.json").read_text(encoding="utf-8") == v002_config_snapshot
