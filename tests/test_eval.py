"""Tests for eval set and LLM-as-judge evaluation harness."""

import json
from pathlib import Path
import pytest

from eval.judge import Judge, JUDGE_SYSTEM_PROMPT


def test_eval_set_structure():
    eval_set_path = Path("eval/eval_set.json")
    assert eval_set_path.exists(), "eval/eval_set.json must exist"

    with open(eval_set_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    assert 15 <= len(items) <= 25, f"Expected 15-25 items, got {len(items)}"

    ids = set()
    for item in items:
        assert "id" in item
        assert "question" in item
        assert "expected_answer" in item
        assert "source_doc" in item
        assert len(item["question"].strip()) > 10
        assert len(item["expected_answer"].strip()) > 10

        assert item["id"] not in ids, f"Duplicate ID: {item['id']}"
        ids.add(item["id"])

        # Check source document exists in active corpus
        doc_path = Path("corpus/active") / item["source_doc"]
        assert doc_path.exists(), f"Source doc does not exist: {doc_path}"


def test_judge_parse_response_valid():
    judge = Judge(api_key="mock_key")
    raw_json = json.dumps({
        "faithfulness": 5,
        "faithfulness_reasoning": "Every fact is supported by the context.",
        "answer_relevancy": 4,
        "relevancy_reasoning": "Answers directly but slightly concise.",
    })

    result = judge._parse_judge_response(raw_json)
    assert result["faithfulness"] == 5
    assert result["answer_relevancy"] == 4
    assert result["overall_score"] == 4
    assert "supported" in result["faithfulness_reasoning"]
    assert "concise" in result["relevancy_reasoning"]


def test_judge_parse_response_clamping_and_fallback():
    judge = Judge(api_key="mock_key")

    # Clamping out-of-bounds scores
    raw_out_of_bounds = json.dumps({
        "faithfulness": 10,
        "answer_relevancy": -2,
    })
    result = judge._parse_judge_response(raw_out_of_bounds)
    assert result["faithfulness"] == 5
    assert result["answer_relevancy"] == 1
    assert result["overall_score"] == 3

    # Markdown codeblock wrapping
    raw_wrapped = "```json\n{\"faithfulness\": 4, \"answer_relevancy\": 4, \"faithfulness_reasoning\": \"ok\"}\n```"
    result = judge._parse_judge_response(raw_wrapped)
    assert result["faithfulness"] == 4
    assert result["answer_relevancy"] == 4
    assert result["overall_score"] == 4

    # Invalid JSON
    raw_invalid = "This is not json at all."
    result = judge._parse_judge_response(raw_invalid)
    assert result["faithfulness"] == 1
    assert result["answer_relevancy"] == 1
    assert result["overall_score"] == 1


def test_judge_prompt_construction():
    judge = Judge(api_key="mock_key")
    chunks = [
        {"content": "FastAPI uses Pydantic.", "metadata": {"relative_path": "features.md"}},
    ]
    prompt = judge.build_eval_prompt(
        query="What library does FastAPI use for models?",
        retrieved_chunks=chunks,
        answer="FastAPI uses Pydantic for data validation.",
        expected_answer="Pydantic is used by FastAPI.",
    )

    assert "What library does FastAPI use for models?" in prompt
    assert "FastAPI uses Pydantic." in prompt
    assert "features.md" in prompt
    assert "FastAPI uses Pydantic for data validation." in prompt
    assert "Expected Ground Truth" in prompt
    assert "Pydantic is used by FastAPI." in prompt


def test_judge_evaluate_rag_mocked(tmp_path: Path):
    judge = Judge(api_key="mock_key")

    # Create dummy eval set
    dummy_eval_set = [
        {
            "id": "q1",
            "topic": "test",
            "question": "What is A?",
            "expected_answer": "A is alpha",
            "source_doc": "a.md",
        },
        {
            "id": "q2",
            "topic": "test",
            "question": "What is B?",
            "expected_answer": "B is beta",
            "source_doc": "b.md",
        }
    ]
    eval_set_file = tmp_path / "eval_set.json"
    with open(eval_set_file, "w") as f:
        json.dump(dummy_eval_set, f)

    class MockIndex:
        config = {"version": "v_test"}

    class MockRetriever:
        index = MockIndex()
        def retrieve(self, query, top_k=3):
            return [{"chunk_id": "c1", "content": f"Content for {query}", "metadata": {}}]

    class MockGenerator:
        model_name = "mock_gen"
        def generate(self, query, chunks):
            return {"answer": f"Answer for {query}"}

    # Mock judge evaluate_single
    call_count = 0
    def mock_eval_single(query, retrieved_chunks, answer, expected_answer=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "faithfulness": 5,
                "faithfulness_reasoning": "Perfect",
                "answer_relevancy": 5,
                "relevancy_reasoning": "Direct",
                "overall_score": 5.0,
            }
        else:
            return {
                "faithfulness": 2,
                "faithfulness_reasoning": "Unfaithful",
                "answer_relevancy": 2,
                "relevancy_reasoning": "Off-topic",
                "overall_score": 2.0,
            }

    judge.evaluate_single = mock_eval_single

    output_report = tmp_path / "report.json"
    report = judge.evaluate_rag(
        retriever=MockRetriever(),
        generator=MockGenerator(),
        eval_set_path=eval_set_file,
        threshold=3.5,
        output_path=output_report,
    )

    summary = report["summary"]
    assert summary["total_queries"] == 2
    assert summary["avg_faithfulness"] == 3.5  # (5 + 2) / 2
    assert summary["avg_answer_relevancy"] == 3.5
    assert summary["avg_overall_score"] == 3.5
    assert summary["passing_count"] == 1
    assert summary["failing_count"] == 1
    assert summary["is_degraded"] is True
    assert len(report["failing_queries"]) == 1
    assert report["failing_queries"][0]["id"] == "q2"

    assert output_report.exists()


def test_v002_sick_index_isolation():
    v001_path = Path("indexes/v001")
    v002_sick_path = Path("indexes/v002_sick")

    assert v001_path.exists(), "indexes/v001 must exist"
    assert v002_sick_path.exists(), "indexes/v002_sick must exist"

    with open(v001_path / "config.json") as f:
        v001_cfg = json.load(f)

    with open(v002_sick_path / "config.json") as f:
        v002_cfg = json.load(f)

    assert v001_cfg["version"] == "v001"
    assert v001_cfg["chunk_size"] == 500
    assert v001_cfg["overlap"] == 100

    assert v002_cfg["version"] == "v002_sick"
    assert v002_cfg["chunk_size"] == 70
    assert v002_cfg["overlap"] == 0
    assert v002_cfg["total_chunks"] > v001_cfg["total_chunks"]


def test_v003_retrieval_sick_index_structure_and_isolation():
    v001_path = Path("indexes/v001")
    v002_sick_path = Path("indexes/v002_sick")
    v003_path = Path("indexes/v003_retrieval_sick")

    assert v001_path.exists(), "indexes/v001 must exist"
    assert v002_sick_path.exists(), "indexes/v002_sick must exist"
    assert v003_path.exists(), "indexes/v003_retrieval_sick must exist"

    with open(v001_path / "config.json") as f:
        v001_cfg = json.load(f)

    with open(v002_sick_path / "config.json") as f:
        v002_cfg = json.load(f)

    with open(v003_path / "config.json") as f:
        v003_cfg = json.load(f)

    # v003_retrieval_sick preserves baseline chunking & corpus
    assert v003_cfg["version"] == "v003_retrieval_sick"
    assert v003_cfg["chunk_size"] == v001_cfg["chunk_size"] == 500
    assert v003_cfg["overlap"] == v001_cfg["overlap"] == 100
    assert v003_cfg["total_chunks"] == v001_cfg["total_chunks"] == 294
    assert v003_cfg["total_documents"] == v001_cfg["total_documents"] == 13
    assert v003_cfg["embedding_model"] == v001_cfg["embedding_model"]

    # v003_retrieval_sick has retrieval degradation (top_k = 1)
    assert v003_cfg["top_k"] == 1

    # v001 and v002_sick remain untouched
    assert v001_cfg["chunk_size"] == 500
    assert v002_cfg["chunk_size"] == 70
    assert v002_cfg["overlap"] == 0

    # Verify evaluation results file exists and records degradation
    v003_results_path = Path("eval/v003_retrieval_sick_results.json")
    assert v003_results_path.exists(), "eval/v003_retrieval_sick_results.json must exist"

    with open(v003_results_path, "r", encoding="utf-8") as f:
        v003_results = json.load(f)

    summary = v003_results["summary"]
    assert summary["index_version"] == "v003_retrieval_sick"
    assert summary["is_degraded"] is True
    assert summary["failing_count"] > 0
    assert len(v003_results["failing_queries"]) == summary["failing_count"]


def test_variant_failure_distinction():
    """Verify that v002_sick and v003_retrieval_sick represent distinct failure modes."""
    with open("eval/sick_results.json") as f:
        v002_results = json.load(f)

    with open("eval/v003_retrieval_sick_results.json") as f:
        v003_results = json.load(f)

    # v002_sick has chunking failure: retrieved chunks have chunk_size=70
    v002_first_fail = v002_results["failing_queries"][0]
    v002_chunks = v002_first_fail["retrieved_chunks"]
    assert len(v002_chunks) == 3
    assert all(c["metadata"]["chunk_size"] == 70 for c in v002_chunks)
    assert all(c["metadata"]["overlap"] == 0 for c in v002_chunks)

    # v003_retrieval_sick has retrieval failure: only 1 chunk retrieved, but chunk_size=500
    v003_first_fail = v003_results["failing_queries"][0]
    v003_chunks = v003_first_fail["retrieved_chunks"]
    assert len(v003_chunks) == 1
    assert v003_chunks[0]["metadata"]["chunk_size"] == 500
    assert v003_chunks[0]["metadata"]["overlap"] == 100


def test_validate_diagnosis_valid_and_categories():
    from agent.diagnose import validate_diagnosis, DIAGNOSIS_CATEGORIES

    assert len(DIAGNOSIS_CATEGORIES) == 7
    assert "chunking problem" in DIAGNOSIS_CATEGORIES
    assert "retrieval problem" in DIAGNOSIS_CATEGORIES

    valid_payload = {
        "suspected_cause": "chunking problem",
        "evidence": "Chunks cut off mid-sentence with size 70.",
        "confidence": 0.95,
        "hypothesis": "Small chunk size fragments context preventing proper grounding.",
        "recommended_experiment": "increase chunk_size to 500",
    }
    validated = validate_diagnosis(valid_payload)
    assert validated["suspected_cause"] == "chunking problem"
    assert validated["confidence"] == 0.95
    assert validated["hypothesis"] == "Small chunk size fragments context preventing proper grounding."


def test_validate_diagnosis_invalid():
    from agent.diagnose import validate_diagnosis

    # Invalid cause
    with pytest.raises(ValueError, match="Invalid suspected_cause"):
        validate_diagnosis({
            "suspected_cause": "unknown mystery problem",
            "evidence": "Chunks cut off mid-sentence with size 70.",
            "confidence": 0.5,
            "hypothesis": "unknown",
            "recommended_experiment": "try something",
        })

    # Missing keys
    with pytest.raises(ValueError, match="missing required keys"):
        validate_diagnosis({"suspected_cause": "chunking problem"})

    # Evidence too brief
    with pytest.raises(ValueError, match="too brief"):
        validate_diagnosis({
            "suspected_cause": "retrieval problem",
            "evidence": "short",
            "confidence": 0.5,
            "hypothesis": "test",
            "recommended_experiment": "test",
        })

    # Evidence unsubstantiated
    with pytest.raises(ValueError, match="substantiated with concrete data references"):
        validate_diagnosis({
            "suspected_cause": "retrieval problem",
            "evidence": "something seems very wrong here in the system",
            "confidence": 0.5,
            "hypothesis": "test",
            "recommended_experiment": "test",
        })

    # Out of range confidence
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_diagnosis({
            "suspected_cause": "retrieval problem",
            "evidence": "Too low top_k with only 1 chunk retrieved from index.",
            "confidence": 1.5,
            "hypothesis": "insufficient chunks retrieved",
            "recommended_experiment": "increase top_k",
        })


def test_fetch_diagnostic_context_and_prompt_formatting():
    from agent.diagnose import fetch_diagnostic_context, format_diagnostic_prompt

    context = fetch_diagnostic_context("v003_retrieval_sick")
    assert context["active_version"] == "v003_retrieval_sick"
    assert "health" in context
    assert "baseline_config" in context
    assert "active_config" in context
    assert len(context["failing_queries"]) > 0

    prompt = format_diagnostic_prompt(context)
    assert "Pipeline Configurations" in prompt
    assert "v003_retrieval_sick" in prompt
    assert "Failing Queries Analysis" in prompt


def test_validate_fix_experiment_valid():
    from agent.fix import validate_fix_experiment

    # Valid chunking proposal
    chunking_payload = {
        "hypothesis": "Increasing chunk size restores context.",
        "strategy": "chunking",
        "changes": {"chunk_size": 400, "overlap": 80},
        "expected_effect": "Improve grounding.",
        "reasoning": "Chunks were too small.",
    }
    val_chunk = validate_fix_experiment(chunking_payload)
    assert val_chunk["strategy"] == "chunking"
    assert val_chunk["changes"]["chunk_size"] == 400

    # Valid retrieval proposal
    retrieval_payload = {
        "hypothesis": "Increasing top_k fetches more chunks.",
        "strategy": "retrieval",
        "changes": {"top_k": 3},
        "expected_effect": "Improve answer coverage.",
        "reasoning": "Only 1 chunk retrieved.",
    }
    val_ret = validate_fix_experiment(retrieval_payload)
    assert val_ret["strategy"] == "retrieval"
    assert val_ret["changes"]["top_k"] == 3


def test_validate_fix_experiment_invalid():
    from agent.fix import validate_fix_experiment

    # Invalid strategy
    with pytest.raises(ValueError, match="Invalid strategy"):
        validate_fix_experiment({
            "hypothesis": "Test",
            "strategy": "magic_fix",
            "changes": {"top_k": 3},
            "expected_effect": "Nothing",
            "reasoning": "Test",
        })

    # Disallowed parameter for retrieval
    with pytest.raises(ValueError, match="Parameter 'chunk_size' is not allowed for strategy 'retrieval'"):
        validate_fix_experiment({
            "hypothesis": "Test",
            "strategy": "retrieval",
            "changes": {"chunk_size": 300},
            "expected_effect": "Nothing",
            "reasoning": "Test",
        })

    # Non-positive chunk size
    with pytest.raises(ValueError, match="positive integer"):
        validate_fix_experiment({
            "hypothesis": "Test",
            "strategy": "chunking",
            "changes": {"chunk_size": -10},
            "expected_effect": "Nothing",
            "reasoning": "Test",
        })



