"""Orchestrator for RAG Doctor self-healing workflow.

Executes the complete self-healing loop:
degraded
  ↓
diagnosis
  ↓
experiment
  ↓
sandbox
  ↓
candidate
  ↓
evaluation
  ↓
comparison

Enforces active index immutability and candidate isolation.
Reuses existing eval/judge.py and eval/compare.py implementations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.diagnose import fetch_diagnostic_context, run_local_diagnose, validate_diagnosis
from agent.fix import run_local_fix, validate_fix_experiment
from eval.compare import (
    compare_evaluation_reports,
    evaluate_and_compare_candidate,
    load_evaluation_report,
)
from eval.judge import Judge
from mcp_server import get_active_version, get_baseline_version, inspect_rag_health
from sandbox_scripts.remediate import execute_remediation


def run_healing_cycle(
    index_version: Optional[str] = None,
    candidate_version: Optional[str] = None,
    diagnosis_override: Optional[Dict[str, Any]] = None,
    experiment_override: Optional[Dict[str, Any]] = None,
    judge: Optional[Judge] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    eval_set_path: str | Path = "eval/eval_set.json",
    indexes_dir: str | Path = "indexes",
    corpus_dir: str | Path = "corpus/active",
    candidate_corpus_dir: str | Path = "corpus/candidate",
) -> Dict[str, Any]:
    """Run end-to-end self-healing pipeline from degradation detection to candidate comparison.

    Workflow:
    1. Health check on degraded index.
    2. Diagnosis subagent analyzes failing queries & configs.
    3. Fix subagent designs isolated experiment proposal.
    4. Sandbox executes deterministic remediation to build isolated candidate index.
    5. Evaluation harness scores the candidate index.
    6. Produces structured comparison against degraded current and baseline.

    Args:
        index_version: Target degraded index version (defaults to active index).
        candidate_version: Optional unique candidate version name.
        diagnosis_override: Optional pre-computed diagnosis dictionary (e.g. for offline testing).
        experiment_override: Optional pre-computed fix experiment dictionary.
        judge: Optional Judge instance (useful for mocking LLM in tests).
        api_key: Optional Groq API key for diagnosis and fix.
        model_name: Optional Groq model name for agents.
        eval_set_path: Path to evaluation dataset.
        indexes_dir: Root directory of vector indexes.
        corpus_dir: Source corpus directory.
        candidate_corpus_dir: Working candidate corpus directory.

    Returns:
        Structured comparison dictionary:
        {
            "baseline_score": float,
            "current_score": float,
            "candidate_score": float,
            "delta_vs_current": float,
            "delta_vs_baseline": float,
            "candidate_better": bool,
            "regressions": List[str],
            "workflow_details": Dict[str, Any]
        }
    """
    indexes_path = Path(indexes_dir)
    active_ver = index_version or get_active_version()
    baseline_ver = get_baseline_version()

    # Capture initial active.json content to guarantee immutability
    active_json_file = indexes_path / "active.json"
    initial_active_state = active_json_file.read_text(encoding="utf-8") if active_json_file.exists() else "{}"

    # 1. Inspect Health
    health_raw = inspect_rag_health(active_ver)
    health_data = json.loads(health_raw)

    # 2. Diagnose Root Cause
    if diagnosis_override:
        diagnosis = validate_diagnosis(diagnosis_override)
    else:
        diagnosis = run_local_diagnose(
            index_version=active_ver,
            api_key=api_key,
            model_name=model_name,
        )
        if "suspected_cause" in diagnosis:
            diagnosis = validate_diagnosis(diagnosis)

    # 3. Formulate Candidate Experiment (Fix)
    if experiment_override:
        active_cfg = json.loads(indexes_path.joinpath(active_ver, "config.json").read_text(encoding="utf-8")) if (indexes_path / active_ver / "config.json").exists() else {}
        experiment = validate_fix_experiment(experiment_override, active_config=active_cfg)
    else:
        experiment = run_local_fix(
            diagnosis=diagnosis,
            index_version=active_ver,
            api_key=api_key,
            model_name=model_name,
        )
        if "strategy" in experiment:
            active_cfg = json.loads(indexes_path.joinpath(active_ver, "config.json").read_text(encoding="utf-8")) if (indexes_path / active_ver / "config.json").exists() else {}
            experiment = validate_fix_experiment(experiment, active_config=active_cfg)

    # 4. Sandbox Remediation Execution (Isolated candidate index creation)
    cand_tag = candidate_version or f"candidate_{active_ver}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    strategy = experiment.get("strategy", "chunking")
    changes = experiment.get("changes", {})

    remediation_res = execute_remediation(
        strategy=strategy,
        source_index=active_ver,
        output_version=cand_tag,
        chunk_size=changes.get("chunk_size"),
        overlap=changes.get("overlap"),
        top_k=changes.get("top_k"),
        corpus_source_dir=corpus_dir,
        corpus_working_dir=candidate_corpus_dir,
        indexes_root_dir=indexes_path,
    )

    cand_dir = indexes_path / cand_tag

    # 5. Evaluate Candidate Index using existing eval harness
    cand_eval_output = PROJECT_ROOT / "eval" / f"{cand_tag}_results.json"
    comparison = evaluate_and_compare_candidate(
        candidate_index_dir=cand_dir,
        current_index_version=active_ver,
        baseline_index_version=baseline_ver,
        eval_set_path=eval_set_path,
        output_report_path=cand_eval_output,
        top_k=changes.get("top_k"),
        judge=judge,
    )

    # 6. Verify Active Index Immutability (Candidate is NEVER automatically promoted)
    post_active_state = active_json_file.read_text(encoding="utf-8") if active_json_file.exists() else "{}"
    if post_active_state != initial_active_state:
        raise RuntimeError(
            f"Active index violation: active.json was modified during candidate workflow! "
            f"Expected '{initial_active_state}', got '{post_active_state}'."
        )

    # Attach workflow metadata for visibility
    comparison["workflow_details"] = {
        "active_version": active_ver,
        "baseline_version": baseline_ver,
        "candidate_version": cand_tag,
        "candidate_dir": str(cand_dir),
        "diagnosis": diagnosis,
        "experiment": experiment,
        "remediation": remediation_res,
        "evaluation_file": str(cand_eval_output),
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(description="RAG Doctor end-to-end self-healing workflow orchestrator.")
    parser.add_argument("--index", default=None, help="Target degraded index version (default: active index)")
    parser.add_argument("--candidate-version", default=None, help="Version tag for candidate index")
    parser.add_argument("--eval-set", default="eval/eval_set.json", help="Path to evaluation dataset")
    args = parser.parse_args()

    result = run_healing_cycle(
        index_version=args.index,
        candidate_version=args.candidate_version,
        eval_set_path=args.eval_set,
    )

    print("\n=== Healing Workflow Candidate Comparison ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
