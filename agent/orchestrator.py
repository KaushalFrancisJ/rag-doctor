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


def format_approval_presentation(
    comparison: Dict[str, Any],
    workflow_details: Optional[Dict[str, Any]] = None,
) -> str:
    """Format structured diagnosis, experiment, score delta, and regressions for human approval."""
    details = workflow_details or comparison.get("workflow_details", {})
    diagnosis = details.get("diagnosis", {})
    experiment = details.get("experiment", {})
    cand_version = details.get("candidate_version", "candidate")

    cause = diagnosis.get("suspected_cause", "Unknown root cause").capitalize()
    evidence = diagnosis.get("evidence", "")
    strategy = experiment.get("strategy", "")
    changes = experiment.get("changes", {})

    if strategy == "chunking":
        exp_desc = f"{changes.get('chunk_size', 'N/A')} chars / {changes.get('overlap', 'N/A')} overlap"
    elif strategy == "retrieval":
        exp_desc = f"top_k = {changes.get('top_k', 'N/A')} chunks"
    else:
        exp_desc = json.dumps(changes)

    curr_score = comparison.get("current_score", 0.0)
    cand_score = comparison.get("candidate_score", 0.0)
    regressions = comparison.get("regressions", [])

    if not regressions:
        reg_text = "No regressions detected"
    else:
        reg_text = f"Regressions detected in {len(regressions)} queries: {', '.join(regressions)}"

    evidence_text = f"\nEvidence:\n{evidence}" if evidence else ""

    return (
        f"RAG Doctor found a candidate fix.\n\n"
        f"Diagnosis:\n{cause}{evidence_text}\n\n"
        f"Experiment:\n{exp_desc}\n\n"
        f"Score:\n{curr_score:.2f} → {cand_score:.2f}\n\n"
        f"Regression checks:\n{reg_text}\n\n"
        f"Promote {cand_version} to active?"
    )


def _extract_experiment_params(
    metadata: Optional[Union[Dict[str, Any], str]] = None,
    candidate_version: str = "",
    current_active_version: str = "v001",
) -> Dict[str, Any]:
    """Extract or infer remediation strategy and parameters for candidate index reconstruction."""
    strategy = "chunking"
    chunk_size = None
    overlap = None
    top_k = None

    meta_dict: Dict[str, Any] = {}
    if isinstance(metadata, dict):
        meta_dict = metadata
    elif isinstance(metadata, str) and metadata.strip():
        try:
            meta_dict = json.loads(metadata)
        except Exception:
            import re
            m_cs = re.search(r"(?:chunk_size|size|chars)\s*[:=]?\s*(\d+)", metadata, re.IGNORECASE)
            if m_cs:
                chunk_size = int(m_cs.group(1))
            m_ov = re.search(r"overlap\s*[:=]?\s*(\d+)", metadata, re.IGNORECASE)
            if m_ov:
                overlap = int(m_ov.group(1))
            m_tk = re.search(r"top_k\s*[:=]?\s*(\d+)", metadata, re.IGNORECASE)
            if m_tk:
                top_k = int(m_tk.group(1))
                strategy = "retrieval"

    if meta_dict:
        exp = meta_dict.get("experiment", meta_dict.get("proposed_remediation", meta_dict))
        if isinstance(exp, dict):
            strategy = exp.get("strategy", strategy)
            changes = exp.get("changes", exp)
            if isinstance(changes, dict):
                chunk_size = changes.get("chunk_size", chunk_size)
                overlap = changes.get("overlap", overlap)
                top_k = changes.get("top_k", top_k)

    cand_lower = candidate_version.lower()
    if top_k is None and ("retrieval" in cand_lower or "top_k" in cand_lower or "v003" in current_active_version):
        strategy = "retrieval"
        top_k = 3

    if strategy == "chunking":
        if chunk_size is None:
            import re
            m_num = re.search(r"(\d{2,4})", candidate_version)
            if m_num and int(m_num.group(1)) not in (1, 2, 3):
                chunk_size = int(m_num.group(1))
            else:
                chunk_size = 500
        if overlap is None:
            overlap = min(100, max(0, int(chunk_size * 0.2)))

    return {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k or 3,
    }


def promote_candidate(
    candidate_version: str,
    indexes_dir: str | Path = "indexes",
    metadata: Optional[Union[Dict[str, Any], str]] = None,
) -> Dict[str, Any]:
    """Promote candidate index to active status after explicit human approval.

    - Updates indexes/active.json with new active_version, preserving previous_active_version.
    - If the candidate index directory does not exist on the machine (e.g. built in an isolated sandbox),
      automatically builds/recreates it on the machine filesystem.
    - Appends promotion record to history.
    - Preserves previous active index directory on disk (never deletes previous versions).
    - Records promotion metadata in indexes/<candidate_version>/promotion_metadata.json.
    """
    indexes_path = Path(indexes_dir)
    cand_dir = indexes_path / candidate_version

    active_file = indexes_path / "active.json"
    active_data: Dict[str, Any] = {}
    if active_file.exists():
        try:
            with open(active_file, "r", encoding="utf-8") as f:
                active_data = json.load(f)
        except Exception:
            active_data = {}

    prev_active = active_data.get("active_version", "v001")
    baseline_ver = active_data.get("baseline_version", "v001")

    # If candidate directory is not present on the host/container filesystem, build it on the machine
    if not cand_dir.exists():
        params = _extract_experiment_params(
            metadata=metadata,
            candidate_version=candidate_version,
            current_active_version=prev_active,
        )

        corpus_src = indexes_path.parent / "corpus" / "active"
        if not corpus_src.exists():
            corpus_src = PROJECT_ROOT / "corpus" / "active"

        corpus_work = indexes_path.parent / "corpus" / "candidate"
        if not corpus_work.parent.exists():
            corpus_work = PROJECT_ROOT / "corpus" / "candidate"

        execute_remediation(
            strategy=params["strategy"],
            source_index=prev_active,
            output_version=candidate_version,
            chunk_size=params.get("chunk_size"),
            overlap=params.get("overlap"),
            top_k=params.get("top_k"),
            corpus_source_dir=corpus_src,
            corpus_working_dir=corpus_work,
            indexes_root_dir=indexes_path,
        )

    now_iso = datetime.now(timezone.utc).isoformat()

    history = active_data.get("history", [])
    history.append({
        "active_version": candidate_version,
        "previous_active_version": prev_active,
        "baseline_version": baseline_ver,
        "promoted_at": now_iso,
    })

    new_active_data = {
        "active_version": candidate_version,
        "previous_active_version": prev_active,
        "baseline_version": baseline_ver,
        "last_promoted_at": now_iso,
        "history": history,
    }

    with open(active_file, "w", encoding="utf-8") as f:
        json.dump(new_active_data, f, indent=2)

    # Write promotion metadata in candidate directory
    promo_meta_file = cand_dir / "promotion_metadata.json"
    promotion_record = {
        "status": "PROMOTED",
        "candidate_version": candidate_version,
        "previous_active_version": prev_active,
        "baseline_version": baseline_ver,
        "promoted_at": now_iso,
        "metadata": metadata or {},
    }
    with open(promo_meta_file, "w", encoding="utf-8") as f:
        json.dump(promotion_record, f, indent=2)

    return promotion_record


def discard_candidate(
    candidate_version: str,
    indexes_dir: str | Path = "indexes",
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Discard a candidate index when human rejects promotion.

    Leaves active index untouched and records discard metadata.
    """
    indexes_path = Path(indexes_dir)
    active_file = indexes_path / "active.json"
    active_ver = "v001"
    if active_file.exists():
        try:
            with open(active_file, "r", encoding="utf-8") as f:
                active_ver = json.load(f).get("active_version", "v001")
        except Exception:
            pass

    now_iso = datetime.now(timezone.utc).isoformat()
    cand_dir = indexes_path / candidate_version

    discard_record = {
        "status": "DISCARDED",
        "candidate_version": candidate_version,
        "active_version": active_ver,
        "active_untouched": True,
        "discarded_at": now_iso,
        "reason": reason or "Rejected by human operator",
    }

    if cand_dir.exists():
        discard_file = cand_dir / "discard_metadata.json"
        try:
            with open(discard_file, "w", encoding="utf-8") as f:
                json.dump(discard_record, f, indent=2)
        except Exception:
            pass

    return discard_record


def rollback_candidate(
    target_version: Optional[str] = None,
    indexes_dir: str | Path = "indexes",
) -> Dict[str, Any]:
    """Roll back active index to previous active version or specified target version."""
    indexes_path = Path(indexes_dir)
    active_file = indexes_path / "active.json"
    active_data: Dict[str, Any] = {}
    if active_file.exists():
        try:
            with open(active_file, "r", encoding="utf-8") as f:
                active_data = json.load(f)
        except Exception:
            active_data = {}

    current_active = active_data.get("active_version", "v001")
    target = target_version or active_data.get("previous_active_version") or active_data.get("baseline_version", "v001")

    target_dir = indexes_path / target
    if not target_dir.exists():
        raise FileNotFoundError(f"Target rollback index version '{target}' not found at '{target_dir}'.")

    now_iso = datetime.now(timezone.utc).isoformat()
    history = active_data.get("history", [])
    history.append({
        "action": "ROLLBACK",
        "from_version": current_active,
        "to_version": target,
        "rolled_back_at": now_iso,
    })

    new_active_data = {
        "active_version": target,
        "previous_active_version": current_active,
        "baseline_version": active_data.get("baseline_version", "v001"),
        "last_rolled_back_at": now_iso,
        "history": history,
    }

    with open(active_file, "w", encoding="utf-8") as f:
        json.dump(new_active_data, f, indent=2)

    return {
        "status": "ROLLED_BACK",
        "active_version": target,
        "previous_active_version": current_active,
        "rolled_back_at": now_iso,
    }


def request_human_approval(
    summary_text: str,
    candidate_version: str,
    approval_callback: Optional[Any] = None,
    simulated_response: Optional[bool] = None,
) -> bool:
    """Pause workflow for explicit human approval.

    - Never automatically promotes.
    - Uses approval_callback if provided.
    - Uses simulated_response if provided.
    - If interactive terminal, prompts the human operator.
    - Defaults to False (safe, never auto-promotes).
    """
    if simulated_response is not None:
        return bool(simulated_response)

    if approval_callback is not None:
        return bool(approval_callback(summary_text))

    if sys.stdin.isatty():
        print("\n" + "=" * 50)
        print(summary_text)
        print("=" * 50)
        try:
            resp = input(f"\nApprove promotion of '{candidate_version}' to active? [y/N]: ").strip().lower()
            return resp in ("y", "yes", "approve")
        except (EOFError, KeyboardInterrupt):
            return False

    return False


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
    require_human_approval: bool = False,
    approval_response: Optional[bool] = None,
    approval_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run end-to-end self-healing pipeline from degradation detection to candidate comparison and approval.

    Workflow:
    1. Health check on degraded index.
    2. Diagnosis subagent analyzes failing queries & configs.
    3. Fix subagent designs isolated experiment proposal.
    4. Sandbox executes deterministic remediation to build isolated candidate index.
    5. Evaluation harness scores the candidate index.
    6. Produces structured comparison against degraded current and baseline.
    7. Pauses for human approval checkpoint if require_human_approval=True.
    8. Promotes candidate if approved, or discards candidate if rejected.

    Args:
        index_version: Target degraded index version (defaults to active index).
        candidate_version: Optional unique candidate version name.
        diagnosis_override: Optional pre-computed diagnosis dictionary.
        experiment_override: Optional pre-computed fix experiment dictionary.
        judge: Optional Judge instance (useful for mocking in tests).
        api_key: Optional Groq API key for diagnosis and fix.
        model_name: Optional Groq model name for agents.
        eval_set_path: Path to evaluation dataset.
        indexes_dir: Root directory of vector indexes.
        corpus_dir: Source corpus directory.
        candidate_corpus_dir: Working candidate corpus directory.
        require_human_approval: Whether to trigger human approval gate.
        approval_response: Optional simulated approval response (True=Approve, False=Reject).
        approval_callback: Optional approval callback function.

    Returns:
        Structured comparison dictionary with workflow details and approval result.
    """
    indexes_path = Path(indexes_dir)
    active_ver = index_version or get_active_version()
    baseline_ver = get_baseline_version()

    # Capture initial active.json content to verify candidate isolation
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

    # Verify Active Index was NOT modified prior to approval
    pre_approval_active_state = active_json_file.read_text(encoding="utf-8") if active_json_file.exists() else "{}"
    if pre_approval_active_state != initial_active_state:
        raise RuntimeError(
            f"Active index violation: active.json was modified prior to human approval! "
            f"Expected '{initial_active_state}', got '{pre_approval_active_state}'."
        )

    # Attach workflow metadata
    workflow_details = {
        "active_version": active_ver,
        "baseline_version": baseline_ver,
        "candidate_version": cand_tag,
        "candidate_dir": str(cand_dir),
        "diagnosis": diagnosis,
        "experiment": experiment,
        "remediation": remediation_res,
        "evaluation_file": str(cand_eval_output),
    }
    comparison["workflow_details"] = workflow_details

    # Format Human Approval Presentation
    presentation_text = format_approval_presentation(comparison, workflow_details)
    comparison["approval_summary"] = presentation_text

    # 6. Human Approval Checkpoint
    if require_human_approval:
        is_approved = request_human_approval(
            summary_text=presentation_text,
            candidate_version=cand_tag,
            approval_callback=approval_callback,
            simulated_response=approval_response,
        )

        if is_approved:
            promo_res = promote_candidate(cand_tag, indexes_dir=indexes_path, metadata=comparison)
            comparison["approval_status"] = "APPROVED"
            comparison["promotion_result"] = promo_res
        else:
            discard_res = discard_candidate(cand_tag, indexes_dir=indexes_path)
            comparison["approval_status"] = "REJECTED"
            comparison["discard_result"] = discard_res
    else:
        comparison["approval_status"] = "PENDING_APPROVAL"

    return comparison


def main():
    parser = argparse.ArgumentParser(description="RAG Doctor end-to-end self-healing workflow orchestrator.")
    parser.add_argument("--index", default=None, help="Target degraded index version (default: active index)")
    parser.add_argument("--candidate-version", default=None, help="Version tag for candidate index")
    parser.add_argument("--eval-set", default="eval/eval_set.json", help="Path to evaluation dataset")
    parser.add_argument("--interactive", action="store_true", help="Prompt human operator interactively for approval")
    args = parser.parse_args()

    result = run_healing_cycle(
        index_version=args.index,
        candidate_version=args.candidate_version,
        eval_set_path=args.eval_set,
        require_human_approval=args.interactive,
    )

    print("\n=== Healing Workflow Summary ===")
    print(result.get("approval_summary", ""))
    print("\n=== Structured Output ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
