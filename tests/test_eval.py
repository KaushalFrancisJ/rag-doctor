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
    assert result["overall_score"] == 4.5
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
    assert result["overall_score"] == 3.0

    # Markdown codeblock wrapping
    raw_wrapped = "```json\n{\"faithfulness\": 4, \"answer_relevancy\": 4, \"faithfulness_reasoning\": \"ok\"}\n```"
    result = judge._parse_judge_response(raw_wrapped)
    assert result["faithfulness"] == 4
    assert result["answer_relevancy"] == 4
    assert result["overall_score"] == 4.0

    # Invalid JSON
    raw_invalid = "This is not json at all."
    result = judge._parse_judge_response(raw_invalid)
    assert result["faithfulness"] == 1
    assert result["answer_relevancy"] == 1
    assert result["overall_score"] == 1.0


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
