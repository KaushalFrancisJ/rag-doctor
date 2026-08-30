"""Unit and integration tests for human approval gate, promotion, discard, and rollback."""

import json
from pathlib import Path
import pytest
import numpy as np

from agent.orchestrator import (
    discard_candidate,
    format_approval_presentation,
    promote_candidate,
    request_human_approval,
    rollback_candidate,
    run_healing_cycle,
)
from eval.judge import Judge
from mcp_server import create_experiment, discard_experiment, promote_experiment, rollback_experiment
from pipeline.chunker import Chunk
from pipeline.index import VectorIndex


@pytest.fixture
def mock_environment(tmp_path: Path):
    """Fixture providing isolated indexes and corpus directory."""
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)

    corpus_dir = tmp_path / "corpus" / "active"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "doc.md").write_text("# FastAPI\nFastAPI is modern.", encoding="utf-8")
    cand_corpus = tmp_path / "corpus" / "candidate"

    class MockEmbedder:
        model_name = "mock-model"
        dimension = 4
        def __init__(self, model_name=None):
            self.model_name = model_name or "mock-model"
            self.dimension = 4
        def embed_texts(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)
        def embed_query(self, query):
            return np.ones((4,), dtype=np.float32)

    # 1. Baseline index v001
    v001_dir = indexes_dir / "v001"
    v001_dir.mkdir()
    chunks1 = [Chunk(id="c1", content="Chunk 1", metadata={"chunk_size": 500, "overlap": 100})]
    VectorIndex.build(
        chunks=chunks1,
        embedder=MockEmbedder(),
        save_dir=v001_dir,
        version="v001",
        chunk_size=500,
        overlap=100,
        top_k=3,
    )

    # 2. Degraded index v002_sick
    v002_dir = indexes_dir / "v002_sick"
    v002_dir.mkdir()
    chunks2 = [Chunk(id="c2", content="Chunk 2", metadata={"chunk_size": 70, "overlap": 0})]
    VectorIndex.build(
        chunks=chunks2,
        embedder=MockEmbedder(),
        save_dir=v002_dir,
        version="v002_sick",
        chunk_size=70,
        overlap=0,
        top_k=3,
    )

    # 3. Candidate index exp_001
    cand_dir = indexes_dir / "exp_001"
    cand_dir.mkdir()
    chunks_cand = [Chunk(id="c3", content="Cand Chunk", metadata={"chunk_size": 500, "overlap": 100})]
    VectorIndex.build(
        chunks=chunks_cand,
        embedder=MockEmbedder(),
        save_dir=cand_dir,
        version="exp_001",
        chunk_size=500,
        overlap=100,
        top_k=3,
    )

    # active.json pointing to v002_sick
    active_file = indexes_dir / "active.json"
    active_data = {"active_version": "v002_sick", "baseline_version": "v001"}
    active_file.write_text(json.dumps(active_data, indent=2), encoding="utf-8")

    return {
        "indexes_dir": indexes_dir,
        "corpus_dir": corpus_dir,
        "cand_corpus": cand_corpus,
        "active_file": active_file,
        "v001_dir": v001_dir,
        "v002_dir": v002_dir,
        "cand_dir": cand_dir,
        "MockEmbedder": MockEmbedder,
    }


def test_format_approval_presentation():
    """Verify format_approval_presentation includes all 8 required fields."""
    comparison = {
        "baseline_score": 4.58,
        "current_score": 3.83,
        "candidate_score": 4.52,
        "delta_vs_current": 0.69,
        "delta_vs_baseline": -0.06,
        "candidate_better": True,
        "regressions": [],
    }

    workflow_details = {
        "candidate_version": "exp_001",
        "diagnosis": {
            "suspected_cause": "chunking problem",
            "evidence": "Small chunk size of 70 cut off context.",
        },
        "experiment": {
            "strategy": "chunking",
            "changes": {"chunk_size": 500, "overlap": 100},
        },
    }

    text = format_approval_presentation(comparison, workflow_details)

    # 1. Header
    assert "RAG Doctor found a candidate fix." in text

    # 2. Suspected cause & evidence
    assert "Diagnosis:" in text
    assert "Chunking problem" in text
    assert "Small chunk size of 70 cut off context." in text

    # 3. Experiment
    assert "Experiment:" in text
    assert "500 chars / 100 overlap" in text

    # 4. Scores & delta
    assert "Score:" in text
    assert "3.83 → 4.52" in text

    # 5. Regressions
    assert "Regression checks:" in text
    assert "No regressions detected" in text

    # 6. Candidate version & approval question
    assert "Promote exp_001 to active?" in text


def test_format_approval_presentation_with_regressions():
    """Verify format_approval_presentation displays regressed query IDs when present."""
    comparison = {
        "current_score": 3.50,
        "candidate_score": 4.00,
        "regressions": ["eval_003", "eval_007"],
    }
    workflow_details = {
        "candidate_version": "exp_retrieval_02",
        "diagnosis": {"suspected_cause": "retrieval problem", "evidence": "Low top_k = 1."},
        "experiment": {"strategy": "retrieval", "changes": {"top_k": 5}},
    }

    text = format_approval_presentation(comparison, workflow_details)
    assert "top_k = 5 chunks" in text
    assert "Regressions detected in 2 queries: eval_003, eval_007" in text
    assert "Promote exp_retrieval_02 to active?" in text


def test_promote_candidate_lifecycle(mock_environment):
    """Verify promoting a candidate updates active.json, preserves previous version, and writes metadata."""
    env = mock_environment
    indexes_dir = env["indexes_dir"]
    active_file = env["active_file"]

    v002_config_before = (env["v002_dir"] / "config.json").read_text(encoding="utf-8")

    meta = {"candidate_score": 4.52, "delta_vs_current": 0.69}
    promo_result = promote_candidate("exp_001", indexes_dir=indexes_dir, metadata=meta)

    assert promo_result["status"] == "PROMOTED"
    assert promo_result["candidate_version"] == "exp_001"
    assert promo_result["previous_active_version"] == "v002_sick"
    assert promo_result["baseline_version"] == "v001"

    # Verify active.json updated
    active_data = json.loads(active_file.read_text(encoding="utf-8"))
    assert active_data["active_version"] == "exp_001"
    assert active_data["previous_active_version"] == "v002_sick"
    assert active_data["baseline_version"] == "v001"
    assert len(active_data["history"]) == 1
    assert active_data["history"][0]["active_version"] == "exp_001"

    # Verify previous active version (v002_sick) is preserved
    assert env["v002_dir"].exists()
    assert (env["v002_dir"] / "config.json").read_text(encoding="utf-8") == v002_config_before

    # Verify promotion metadata file written in candidate directory
    promo_file = env["cand_dir"] / "promotion_metadata.json"
    assert promo_file.exists()
    file_meta = json.loads(promo_file.read_text(encoding="utf-8"))
    assert file_meta["status"] == "PROMOTED"
    assert file_meta["candidate_version"] == "exp_001"


def test_promote_candidate_auto_rebuilds_when_missing(mock_environment, monkeypatch):
    """Verify promote_candidate automatically builds and promotes candidate index when not on disk."""
    env = mock_environment
    monkeypatch.setattr("sandbox_scripts.remediate.Embedder", env["MockEmbedder"])

    # Attempt promoting candidate that was created in sandbox (not yet on disk)
    meta = '{"strategy": "chunking", "changes": {"chunk_size": 150, "overlap": 20}}'
    promo_res = promote_candidate("v002_sick_modified", indexes_dir=env["indexes_dir"], metadata=meta)

    assert promo_res["status"] == "PROMOTED"
    assert promo_res["candidate_version"] == "v002_sick_modified"
    assert (env["indexes_dir"] / "v002_sick_modified" / "index.faiss").exists()
    assert (env["indexes_dir"] / "v002_sick_modified" / "config.json").exists()

    cfg = json.loads((env["indexes_dir"] / "v002_sick_modified" / "config.json").read_text(encoding="utf-8"))
    assert cfg["chunk_size"] == 150
    assert cfg["overlap"] == 20

    active_data = json.loads(env["active_file"].read_text(encoding="utf-8"))
    assert active_data["active_version"] == "v002_sick_modified"


def test_discard_candidate_leaves_active_untouched(mock_environment):
    """Verify discard_candidate leaves active index untouched and writes discard metadata."""
    env = mock_environment
    active_file = env["active_file"]
    active_content_before = active_file.read_text(encoding="utf-8")

    discard_res = discard_candidate("exp_001", indexes_dir=env["indexes_dir"], reason="Quality score insufficient")

    assert discard_res["status"] == "DISCARDED"
    assert discard_res["active_untouched"] is True
    assert discard_res["active_version"] == "v002_sick"

    # Verify active.json is completely untouched
    assert active_file.read_text(encoding="utf-8") == active_content_before

    # Verify discard metadata file in candidate directory
    discard_file = env["cand_dir"] / "discard_metadata.json"
    assert discard_file.exists()
    file_data = json.loads(discard_file.read_text(encoding="utf-8"))
    assert file_data["status"] == "DISCARDED"
    assert file_data["reason"] == "Quality score insufficient"


def test_rollback_candidate(mock_environment):
    """Verify rollback_candidate restores previous active index version."""
    env = mock_environment
    indexes_dir = env["indexes_dir"]

    # 1. Promote exp_001
    promote_candidate("exp_001", indexes_dir=indexes_dir)
    assert json.loads(env["active_file"].read_text(encoding="utf-8"))["active_version"] == "exp_001"

    # 2. Rollback to previous version (v002_sick)
    rollback_res = rollback_candidate(indexes_dir=indexes_dir)
    assert rollback_res["status"] == "ROLLED_BACK"
    assert rollback_res["active_version"] == "v002_sick"
    assert rollback_res["previous_active_version"] == "exp_001"

    active_data = json.loads(env["active_file"].read_text(encoding="utf-8"))
    assert active_data["active_version"] == "v002_sick"
    assert active_data["previous_active_version"] == "exp_001"


def test_request_human_approval_behavior():
    """Verify request_human_approval honors simulated response, callbacks, and safe defaults."""
    # 1. Simulated response True
    assert request_human_approval("summary", "exp_001", simulated_response=True) is True

    # 2. Simulated response False
    assert request_human_approval("summary", "exp_001", simulated_response=False) is False

    # 3. Callback returning True
    assert request_human_approval("summary", "exp_001", approval_callback=lambda s: "summary" in s) is True

    # 4. Callback returning False
    assert request_human_approval("summary", "exp_001", approval_callback=lambda s: False) is False

    # 5. Non-interactive default (sys.stdin not a tty in test environment) -> False (safe)
    assert request_human_approval("summary", "exp_001") is False


def test_mcp_promotion_and_rollback_tools(mock_environment, monkeypatch):
    """Verify promote_experiment, rollback_experiment, and discard_experiment MCP tools."""
    env = mock_environment
    import mcp_server
    monkeypatch.setattr(mcp_server, "PROJECT_ROOT", env["indexes_dir"].parent)

    # 1. Promote via MCP
    promo_raw = promote_experiment("exp_001", metadata='{"tested_by": "operator"}')
    promo_data = json.loads(promo_raw)
    assert promo_data["status"] == "PROMOTED"
    assert promo_data["candidate_version"] == "exp_001"

    active_data = json.loads(env["active_file"].read_text(encoding="utf-8"))
    assert active_data["active_version"] == "exp_001"

    # 2. Rollback via MCP
    roll_raw = rollback_experiment()
    roll_data = json.loads(roll_raw)
    assert roll_data["status"] == "ROLLED_BACK"
    assert roll_data["active_version"] == "v002_sick"

    # 3. Discard via MCP
    discard_raw = discard_experiment("exp_001", reason="Testing discard")
    discard_data = json.loads(discard_raw)
    assert discard_data["status"] == "DISCARDED"

    # 4. Create experiment via MCP
    monkeypatch.setattr("sandbox_scripts.remediate.Embedder", env["MockEmbedder"])
    eval_set_path = env["indexes_dir"].parent / "eval" / "eval_set.json"
    eval_set_path.parent.mkdir(parents=True, exist_ok=True)
    eval_set_path.write_text(json.dumps([{"id": "q1", "question": "test question", "expected_answer": "test", "source_doc": "doc.md"}]), encoding="utf-8")

    # Mock judge evaluate_index
    def mock_eval_index(index_dir, eval_set_path, threshold=3.5, output_path=None, top_k=None, **kwargs):
        report = {
            "summary": {
                "index_version": Path(index_dir).name,
                "avg_overall_score": 4.65,
                "overall_score": 5,
                "is_degraded": False,
            },
            "results": [{"id": "q1", "overall_score": 5, "passed": True}],
            "failing_queries": [],
        }
        if output_path:
            Path(output_path).write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr("eval.compare.evaluate_index", mock_eval_index)

    create_raw = create_experiment(strategy="chunking", output_version="exp_mcp_01", chunk_size=500, overlap=100)
    create_data = json.loads(create_raw)
    assert create_data["candidate_score"] == 4.65
    assert (env["indexes_dir"] / "exp_mcp_01" / "index.faiss").exists()


def test_run_healing_cycle_with_approval_flow(mock_environment, monkeypatch):
    """Verify run_healing_cycle promotes on approval and discards on rejection."""
    env = mock_environment
    monkeypatch.setattr("sandbox_scripts.remediate.Embedder", env["MockEmbedder"])

    judge = Judge(api_key="mock_key")
    def mock_eval_rag(retriever, generator, eval_set_path, threshold=3.5, output_path=None, top_k=None):
        return {
            "summary": {
                "index_version": retriever.index.config.get("version"),
                "avg_overall_score": 4.60,
                "overall_score": 5,
                "is_degraded": False,
            },
            "results": [{"id": "eval_001", "overall_score": 5, "passed": True}],
            "failing_queries": [],
        }
    judge.evaluate_rag = mock_eval_rag

    diag_override = {
        "suspected_cause": "chunking problem",
        "evidence": "Observed small chunk size 70 cutting off sentences.",
        "confidence": 0.95,
        "hypothesis": "Small chunks fragment text.",
        "recommended_experiment": "increase chunk_size to 500",
    }
    exp_override = {
        "hypothesis": "Increase chunk size to 500.",
        "strategy": "chunking",
        "changes": {"chunk_size": 500, "overlap": 100},
        "expected_effect": "Improve grounding.",
        "reasoning": "Restore context.",
    }

    # 1. Run with explicit rejection (approval_response=False)
    res_rejected = run_healing_cycle(
        index_version="v002_sick",
        candidate_version="exp_rejected_01",
        diagnosis_override=diag_override,
        experiment_override=exp_override,
        judge=judge,
        indexes_dir=env["indexes_dir"],
        corpus_dir=env["corpus_dir"],
        candidate_corpus_dir=env["cand_corpus"],
        require_human_approval=True,
        approval_response=False,
    )

    assert res_rejected["approval_status"] == "REJECTED"
    assert res_rejected["discard_result"]["status"] == "DISCARDED"
    # Active index remains untouched (v002_sick)
    assert json.loads(env["active_file"].read_text(encoding="utf-8"))["active_version"] == "v002_sick"

    # 2. Run with explicit approval (approval_response=True)
    res_approved = run_healing_cycle(
        index_version="v002_sick",
        candidate_version="exp_approved_01",
        diagnosis_override=diag_override,
        experiment_override=exp_override,
        judge=judge,
        indexes_dir=env["indexes_dir"],
        corpus_dir=env["corpus_dir"],
        candidate_corpus_dir=env["cand_corpus"],
        require_human_approval=True,
        approval_response=True,
    )

    assert res_approved["approval_status"] == "APPROVED"
    assert res_approved["promotion_result"]["status"] == "PROMOTED"
    assert res_approved["promotion_result"]["candidate_version"] == "exp_approved_01"

    # Active index updated to exp_approved_01
    active_now = json.loads(env["active_file"].read_text(encoding="utf-8"))
    assert active_now["active_version"] == "exp_approved_01"
    assert active_now["previous_active_version"] == "v002_sick"
