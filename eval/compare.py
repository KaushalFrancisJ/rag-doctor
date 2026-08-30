"""Candidate evaluation and comparison harness for RAG Doctor.

Connects candidate evaluation to the healing workflow:
1. Runs the existing evaluation harness (eval/judge.py) against candidate indexes.
2. Compares degraded current score, baseline score, and candidate score.
3. Produces structured comparison payloads and identifies query regressions.
4. Preserves active index immutability and enforces candidate isolation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.judge import Judge, evaluate_index


def _extract_score(report: Dict[str, Any]) -> float:
    """Extract avg_overall_score or overall_score as a float from an evaluation report."""
    summary = report.get("summary", {})
    if "avg_overall_score" in summary and summary["avg_overall_score"] is not None:
        return round(float(summary["avg_overall_score"]), 2)
    if "overall_score" in summary and summary["overall_score"] is not None:
        return round(float(summary["overall_score"]), 2)
    return 0.0


def load_evaluation_report(
    source: Union[str, Path, Dict[str, Any]],
    eval_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load an evaluation report dictionary from a dict, file path, or index version tag.

    Args:
        source: A report dictionary, a direct file Path/str, or a version string (e.g. 'v001').
        eval_dir: Optional base directory to search for '<version>_results.json'.

    Returns:
        Loaded evaluation report dictionary.

    Raises:
        FileNotFoundError: If the report file cannot be found.
        ValueError: If the file contains invalid JSON.
    """
    if isinstance(source, dict):
        return source

    base_eval_dir = Path(eval_dir) if eval_dir else PROJECT_ROOT / "eval"
    src_str = str(source).strip()

    # 1. Direct path check
    direct_path = Path(src_str)
    if direct_path.exists() and direct_path.is_file():
        with open(direct_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Check relative to base_eval_dir
    rel_in_eval = base_eval_dir / src_str
    if rel_in_eval.exists() and rel_in_eval.is_file():
        with open(rel_in_eval, "r", encoding="utf-8") as f:
            return json.load(f)

    # 3. Check version naming convention: <version>_results.json
    convention_path = base_eval_dir / f"{src_str}_results.json"
    if convention_path.exists() and convention_path.is_file():
        with open(convention_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 4. Check special alias for baseline / sick if applicable
    if src_str in ("v002_sick", "sick"):
        alias_sick = base_eval_dir / "sick_results.json"
        if alias_sick.exists():
            with open(alias_sick, "r", encoding="utf-8") as f:
                return json.load(f)

    if src_str in ("v001", "baseline"):
        alias_base = base_eval_dir / "baseline_results.json"
        if alias_base.exists():
            with open(alias_base, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Could not locate evaluation report for '{source}'. Checked: "
        f"'{direct_path}', '{rel_in_eval}', '{convention_path}'"
    )


def identify_regressions(
    candidate_results: List[Dict[str, Any]],
    baseline_results: Optional[List[Dict[str, Any]]] = None,
    current_results: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Identify query IDs that regressed in the candidate evaluation.

    A regression occurs when:
    - A query scored lower in the candidate than in the baseline.
    - A query passed in baseline/current but failed in the candidate.
    - A query scored lower in candidate than in current degraded.

    Returns:
        Sorted list of unique query IDs representing regressions.
    """
    baseline_map = {r["id"]: r for r in (baseline_results or []) if "id" in r}
    current_map = {r["id"]: r for r in (current_results or []) if "id" in r}

    regressed_ids: set[str] = set()

    for cand_q in candidate_results:
        qid = cand_q.get("id")
        if not qid:
            continue

        cand_score = cand_q.get("overall_score", 0)
        cand_passed = cand_q.get("passed", cand_score >= 3.5)

        # Compare against baseline
        if qid in baseline_map:
            base_q = baseline_map[qid]
            base_score = base_q.get("overall_score", 0)
            base_passed = base_q.get("passed", base_score >= 3.5)

            if cand_score < base_score or (base_passed and not cand_passed):
                regressed_ids.add(qid)

        # Compare against current degraded
        if qid in current_map:
            curr_q = current_map[qid]
            curr_score = curr_q.get("overall_score", 0)
            curr_passed = curr_q.get("passed", curr_score >= 3.5)

            if curr_passed and not cand_passed:
                regressed_ids.add(qid)

    return sorted(list(regressed_ids))


def compare_evaluation_reports(
    candidate_report: Dict[str, Any],
    current_report: Dict[str, Any],
    baseline_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce a structured comparison between candidate, degraded current, and baseline evaluations.

    Args:
        candidate_report: Evaluation report dictionary for the candidate index.
        current_report: Evaluation report dictionary for the currently active/degraded index.
        baseline_report: Optional baseline evaluation report dictionary (defaults to current_report if None).

    Returns:
        Structured dictionary matching canonical schema:
        {
            "baseline_score": float,
            "current_score": float,
            "candidate_score": float,
            "delta_vs_current": float,
            "delta_vs_baseline": float,
            "candidate_better": bool,
            "regressions": List[str]
        }
    """
    effective_base = baseline_report if baseline_report is not None else current_report

    cand_score = _extract_score(candidate_report)
    curr_score = _extract_score(current_report)
    base_score = _extract_score(effective_base)

    delta_vs_current = round(cand_score - curr_score, 2)
    delta_vs_baseline = round(cand_score - base_score, 2)

    cand_results = candidate_report.get("results", [])
    base_results = effective_base.get("results", [])
    curr_results = current_report.get("results", [])

    regressions = identify_regressions(
        candidate_results=cand_results,
        baseline_results=base_results,
        current_results=curr_results,
    )

    # Candidate is better if it achieves a strictly higher overall score than current degraded
    # and improves quality.
    candidate_better = bool(cand_score > curr_score)

    return {
        "baseline_score": base_score,
        "current_score": curr_score,
        "candidate_score": cand_score,
        "delta_vs_current": delta_vs_current,
        "delta_vs_baseline": delta_vs_baseline,
        "candidate_better": candidate_better,
        "regressions": regressions,
    }


def evaluate_and_compare_candidate(
    candidate_index_dir: Union[str, Path],
    current_index_version: Optional[str] = None,
    baseline_index_version: Optional[str] = None,
    eval_set_path: Union[str, Path] = "eval/eval_set.json",
    threshold: float = 3.5,
    output_report_path: Optional[Union[str, Path]] = None,
    top_k: Optional[int] = None,
    eval_limit: Optional[int] = None,
    generator_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge: Optional[Judge] = None,
) -> Dict[str, Any]:
    """Execute evaluation harness against a candidate index and return structured comparison.

    Uses the existing eval/judge.py harness (no duplication of evaluation logic).
    Preserves active index immutability — does not promote the candidate.

    Args:
        candidate_index_dir: Directory path of the candidate FAISS index.
        current_index_version: Version tag of degraded current index (defaults to active.json).
        baseline_index_version: Version tag of baseline index (defaults to active.json).
        eval_set_path: Path to held-out evaluation dataset.
        threshold: Score threshold for degradation checks.
        output_report_path: Optional path to persist candidate evaluation report JSON.
        top_k: Optional top_k override.
        eval_limit: Optional limit on evaluation query count for fast trial runs.
        generator_model: Groq model name for generation.
        judge_model: Groq model name for judging.
        judge: Optional pre-configured Judge instance (useful for unit testing / mocking).

    Returns:
        Structured comparison dictionary.
    """
    cand_path = Path(candidate_index_dir)
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate index directory not found: {cand_path}")

    # 1. Resolve current and baseline versions
    active_json = PROJECT_ROOT / "indexes" / "active.json"
    active_state = {}
    if active_json.exists():
        try:
            with open(active_json, "r", encoding="utf-8") as f:
                active_state = json.load(f)
        except Exception:
            active_state = {}

    curr_ver = current_index_version or active_state.get("active_version", "v001")
    base_ver = baseline_index_version or active_state.get("baseline_version", "v001")

    # 2. Run existing evaluation harness against candidate index
    if judge is not None:
        from pipeline.generator import Generator
        from pipeline.index import VectorIndex
        from pipeline.retriever import Retriever

        cand_index = VectorIndex.load(cand_path)
        eff_k = top_k if top_k is not None else int(cand_index.config.get("top_k", 3))
        retriever = Retriever(index=cand_index, default_top_k=eff_k)
        generator = Generator(model_name=generator_model)
        cand_report = judge.evaluate_rag(
            retriever=retriever,
            generator=generator,
            eval_set_path=eval_set_path,
            threshold=threshold,
            output_path=output_report_path,
            top_k=eff_k,
            eval_limit=eval_limit,
        )
    else:
        cand_report = evaluate_index(
            index_dir=cand_path,
            eval_set_path=eval_set_path,
            threshold=threshold,
            output_path=output_report_path,
            top_k=top_k,
            eval_limit=eval_limit,
            generator_model=generator_model,
            judge_model=judge_model,
        )

    # 3. Load current degraded and baseline evaluation reports
    current_report = load_evaluation_report(curr_ver)
    baseline_report = load_evaluation_report(base_ver)

    # 4. Produce structured comparison
    comparison = compare_evaluation_reports(
        candidate_report=cand_report,
        current_report=current_report,
        baseline_report=baseline_report,
    )

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate candidate index and compare against baseline/current.")
    parser.add_argument("--candidate-index", required=True, help="Path to candidate index directory")
    parser.add_argument("--current-version", default=None, help="Current active/degraded index version (default: from active.json)")
    parser.add_argument("--baseline-version", default=None, help="Baseline index version (default: from active.json)")
    parser.add_argument("--eval-set", default="eval/eval_set.json", help="Path to evaluation dataset")
    parser.add_argument("--output", default=None, help="Optional path to save candidate evaluation results JSON")
    parser.add_argument("--threshold", type=float, default=3.5, help="Score threshold (1-5)")
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval top_k chunks")
    args = parser.parse_args()

    comparison = evaluate_and_compare_candidate(
        candidate_index_dir=args.candidate_index,
        current_index_version=args.current_version,
        baseline_index_version=args.baseline_version,
        eval_set_path=args.eval_set,
        threshold=args.threshold,
        output_report_path=args.output,
        top_k=args.top_k,
    )

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
