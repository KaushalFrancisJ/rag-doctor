"""LLM-as-a-judge evaluation harness for RAG Doctor.

Evaluates generated answers on Faithfulness and Answer Relevancy (1-5 scale)
using Groq structured JSON outputs, referenced against Ragas principles.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from pipeline.generator import Generator
from pipeline.index import VectorIndex
from pipeline.retriever import Retriever
from pipeline.llm_client import DEFAULT_MODEL as DEFAULT_JUDGE_MODEL

load_dotenv()

DEFAULT_THRESHOLD = 3.5

JUDGE_SYSTEM_PROMPT = """You are an expert impartial evaluation judge for Retrieval-Augmented Generation (RAG) systems.
Your task is to evaluate the quality of a generated answer given a user question and the retrieved context documents.

You will score two distinct dimensions on a 1 to 5 integer scale:

1. FAITHFULNESS (1 to 5):
   Measures whether the claims in the generated answer are grounded in and supported by the retrieved context.
   - 5: Completely faithful. Every factual claim is directly supported by the context. No hallucinations.
   - 4: Mostly faithful. The core claims are supported; minor general statements do not contradict the context.
   - 3: Partially faithful. Some claims are supported, but contains notable statements not present in the context.
   - 2: Mostly unfaithful. Major claims contradict or have no basis in the retrieved context.
   - 1: Completely unfaithful / hallucinated. The answer is fabricated or directly contradicts the context.

2. ANSWER RELEVANCY (1 to 5):
   Measures how directly, completely, and appropriately the answer addresses the user's question.
   - 5: Highly relevant and complete. Directly answers the question thoroughly and concisely.
   - 4: Relevant. Answers the question well, with minor omissions or slight verbosity.
   - 3: Partially relevant. Addresses part of the question but misses key aspects or includes tangential details.
   - 2: Poor relevance. Barely addresses the question, mostly off-topic or evasive.
   - 1: Irrelevant / non-responsive. Fails completely to address the user's prompt.

You MUST return a valid JSON object strictly matching this schema:
{
  "faithfulness": <integer 1-5>,
  "faithfulness_reasoning": "<concise explanation of grounding against context>",
  "answer_relevancy": <integer 1-5>,
  "relevancy_reasoning": "<concise explanation of completeness and directness for the question>"
}"""


class Judge:
    """LLM-as-judge scoring harness for RAG pipeline evaluation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("GROQ_MODEL", DEFAULT_JUDGE_MODEL)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy-loaded Groq client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY environment variable is not set. "
                    "Please set GROQ_API_KEY or provide it to Judge(api_key=...)."
                )
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def build_eval_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        answer: str,
        expected_answer: Optional[str] = None,
    ) -> str:
        """Format evaluation inputs into structured evaluation prompt."""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("metadata", {}).get("relative_path", "unknown")
            content = chunk.get("content", "").strip()
            context_parts.append(f"[Chunk {i} | Source: {source}]\n{content}")

        context_str = "\n\n".join(context_parts) if context_parts else "No context retrieved."

        expected_section = f"\nExpected Ground Truth (for reference):\n{expected_answer}\n" if expected_answer else ""

        return (
            f"User Question:\n{query}\n\n"
            f"Retrieved Context:\n{context_str}\n"
            f"{expected_section}\n"
            f"Generated Answer To Evaluate:\n{answer}\n\n"
            f"Provide your JSON evaluation with 'faithfulness' (1-5), 'faithfulness_reasoning', "
            f"'answer_relevancy' (1-5), and 'relevancy_reasoning'."
        )

    def evaluate_single(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        answer: str,
        expected_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single query and answer.

        Returns:
            Dict with keys: faithfulness, faithfulness_reasoning,
            answer_relevancy, relevancy_reasoning, overall_score.
        """
        prompt = self.build_eval_prompt(query, retrieved_chunks, answer, expected_answer)

        import time
        max_retries = 6
        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                if ("rate_limit" in err_str or "429" in err_str or "retry" in err_str) and attempt < max_retries - 1:
                    wait_sec = 4.0 * (attempt + 1)
                    m_min = re.search(r"try again in (\d+)m([\d\.]+)s", str(e), re.I)
                    m_sec = re.search(r"try again in ([\d\.]+)s", str(e), re.I)
                    m_ms = re.search(r"try again in ([\d\.]+)ms", str(e), re.I)
                    if m_min:
                        wait_sec = int(m_min.group(1)) * 60 + float(m_min.group(2)) + 1.0
                    elif m_sec:
                        wait_sec = float(m_sec.group(1)) + 1.0
                    elif m_ms:
                        wait_sec = float(m_ms.group(1)) / 1000.0 + 1.0
                    time.sleep(wait_sec)
                else:
                    raise

        content = response.choices[0].message.content or "{}"
        parsed = self._parse_judge_response(content)
        return parsed

    def _parse_judge_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse and validate JSON response from the judge LLM."""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback regex extraction if raw JSON has surrounding text
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = {}

        # Extract and clamp scores 1-5
        faithfulness = int(data.get("faithfulness", 1))
        faithfulness = max(1, min(5, faithfulness))

        answer_relevancy = int(data.get("answer_relevancy", 1))
        answer_relevancy = max(1, min(5, answer_relevancy))

        faith_reason = str(data.get("faithfulness_reasoning", "No reasoning provided.")).strip()
        rel_reason = str(data.get("relevancy_reasoning", "No reasoning provided.")).strip()

        overall_score = int(round((faithfulness + answer_relevancy) / 2.0))
        overall_score = max(1, min(5, overall_score))

        return {
            "faithfulness": faithfulness,
            "faithfulness_reasoning": faith_reason,
            "answer_relevancy": answer_relevancy,
            "relevancy_reasoning": rel_reason,
            "overall_score": overall_score,
        }

    def evaluate_rag(
        self,
        retriever: Retriever,
        generator: Generator,
        eval_set_path: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        output_path: Optional[str | Path] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation suite over an evaluation dataset.

        Args:
            retriever: Retriever instance.
            generator: Generator instance.
            eval_set_path: Path to eval_set.json.
            threshold: Minimum acceptable overall score.
            output_path: Optional file path to persist evaluation results JSON.
            top_k: Number of chunks to retrieve per query (defaults to retriever.default_top_k).

        Returns:
            Structured dictionary with summary metrics and per-query results.
        """
        eval_path = Path(eval_set_path)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_set = json.load(f)

        if not eval_set:
            raise ValueError("Evaluation dataset is empty. Cannot evaluate RAG pipeline without evaluation queries.")

        effective_k = top_k if top_k is not None else getattr(retriever, "default_top_k", 3)
        results: List[Dict[str, Any]] = []
        total_faithfulness = 0
        total_relevancy = 0
        total_overall = 0.0
        failing_queries: List[Dict[str, Any]] = []

        for item in eval_set:
            query_id = item.get("id", "unknown")
            topic = item.get("topic", "general")
            query = item["question"]
            expected = item.get("expected_answer")

            # 1. Retrieve
            chunks = retriever.retrieve(query, top_k=effective_k)

            # 2. Generate
            gen_result = generator.generate(query, chunks)
            generated_answer = gen_result["answer"]

            # 3. Judge
            judge_result = self.evaluate_single(
                query=query,
                retrieved_chunks=chunks,
                answer=generated_answer,
                expected_answer=expected,
            )

            passed = (
                judge_result["overall_score"] >= threshold
                and judge_result["faithfulness"] >= 3
                and judge_result["answer_relevancy"] >= 3
            )

            query_result = {
                "id": query_id,
                "topic": topic,
                "question": query,
                "expected_answer": expected,
                "generated_answer": generated_answer,
                "retrieved_chunks": chunks,
                "faithfulness": judge_result["faithfulness"],
                "faithfulness_reasoning": judge_result["faithfulness_reasoning"],
                "answer_relevancy": judge_result["answer_relevancy"],
                "relevancy_reasoning": judge_result["relevancy_reasoning"],
                "overall_score": judge_result["overall_score"],
                "passed": passed,
            }

            results.append(query_result)
            total_faithfulness += judge_result["faithfulness"]
            total_relevancy += judge_result["answer_relevancy"]
            total_overall += judge_result["overall_score"]

            if not passed:
                failing_queries.append(query_result)

            import time
            time.sleep(0.5)

        count = len(results)
        avg_faithfulness = round(total_faithfulness / count, 2) if count else 0.0
        avg_relevancy = round(total_relevancy / count, 2) if count else 0.0
        avg_overall = round(total_overall / count, 2) if count else 0.0

        is_degraded = avg_overall < threshold or len(failing_queries) > 0

        index_version = retriever.index.config.get("version", "unknown")

        eval_report = {
            "summary": {
                "index_version": index_version,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "judge_model": self.model_name,
                "generator_model": generator.model_name,
                "total_queries": count,
                "overall_score": int(round(avg_overall)) if count else 0,
                "avg_faithfulness": avg_faithfulness,
                "avg_answer_relevancy": avg_relevancy,
                "avg_overall_score": avg_overall,
                "threshold": threshold,
                "passing_count": count - len(failing_queries),
                "failing_count": len(failing_queries),
                "is_degraded": is_degraded,
            },
            "failing_queries": failing_queries,
            "results": results,
        }

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2, ensure_ascii=False)

        return eval_report


def evaluate_index(
    index_dir: str | Path,
    eval_set_path: str | Path = "eval/eval_set.json",
    threshold: float = DEFAULT_THRESHOLD,
    output_path: Optional[str | Path] = None,
    top_k: Optional[int] = None,
    generator_model: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to evaluate an index directory against an evaluation set."""
    index = VectorIndex.load(index_dir)
    effective_top_k = top_k if top_k is not None else int(index.config.get("top_k", 3))
    retriever = Retriever(index=index, default_top_k=effective_top_k)
    generator = Generator(model_name=generator_model)
    judge = Judge(model_name=judge_model)

    return judge.evaluate_rag(
        retriever=retriever,
        generator=generator,
        eval_set_path=eval_set_path,
        threshold=threshold,
        output_path=output_path,
        top_k=effective_top_k,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline index using LLM-as-judge.")
    parser.add_argument("--index", default="indexes/v001", help="Path to index directory")
    parser.add_argument("--eval-set", default="eval/eval_set.json", help="Path to evaluation dataset")
    parser.add_argument("--output", default=None, help="Path to save evaluation report JSON")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Degradation score threshold (1-5)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve (default: from index config or 3)")
    parser.add_argument("--generator-model", default=None, help="Groq model for generator (default: from env or DEFAULT_GROQ_MODEL)")
    parser.add_argument("--judge-model", default=None, help="Groq model for judge (default: from env or DEFAULT_JUDGE_MODEL)")
    args = parser.parse_args()

    print(f"=== Evaluating RAG Index: {args.index} ===")
    report = evaluate_index(
        index_dir=args.index,
        eval_set_path=args.eval_set,
        threshold=args.threshold,
        output_path=args.output,
        top_k=args.top_k,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
    )

    summary = report["summary"]
    print("\n=== Evaluation Summary ===")
    print(f"Index Version:        {summary['index_version']}")
    print(f"Total Queries:        {summary['total_queries']}")
    print(f"Avg Faithfulness:     {summary['avg_faithfulness']:.2f} / 5.0")
    print(f"Avg Answer Relevancy: {summary['avg_answer_relevancy']:.2f} / 5.0")
    print(f"Avg Overall Score:    {summary['avg_overall_score']:.2f} / 5.0")
    print(f"Threshold:            {summary['threshold']:.2f}")
    print(f"Passing Queries:      {summary['passing_count']} / {summary['total_queries']}")
    print(f"Failing Queries:      {summary['failing_count']}")
    print(f"Degraded Status:      {'DEGRADED' if summary['is_degraded'] else 'HEALTHY'}")

    if args.output:
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
