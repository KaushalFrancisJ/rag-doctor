"""Diagnostic client and validation utilities for RAG Doctor.

This module provides helper utilities to:
1. Gather pipeline health and diagnostic context via MCP.
2. Format diagnostic context into prompt strings.
3. Validate structured diagnosis responses against the canonical schema.
4. Provide a lightweight local debug / smoke-test utility for development.

Note: Production agent orchestration, subagent loops, model calls, sandbox execution,
and human approval gates are handled by TrueForge.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Valid diagnostic root-cause categories
DIAGNOSIS_CATEGORIES = [
    "chunking problem",
    "retrieval problem",
    "embedding problem",
    "indexing problem",
    "metadata/context problem",
    "generation problem",
    "evaluation/data problem",
]

INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "diagnose_instructions.md"

if INSTRUCTIONS_PATH.exists():
    with open(INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
        DIAGNOSIS_SYSTEM_PROMPT = f.read().strip()
else:
    DIAGNOSIS_SYSTEM_PROMPT = """You are an expert AI Diagnostic Agent for a Retrieval-Augmented Generation (RAG) pipeline.
Your task is to analyze failing evaluation queries and pipeline configuration to determine the root cause of the failure.

Possible diagnosis categories:
- chunking problem
- retrieval problem
- embedding problem
- indexing problem
- metadata/context problem
- generation problem
- evaluation/data problem

You MUST return a valid JSON object strictly matching this schema:
{
  "suspected_cause": "<one of the categories above>",
  "evidence": "<concise explanation of why this cause is suspected based on the chunks and configs>",
  "confidence": <float between 0.0 and 1.0>,
  "recommended_experiment": "<concise recommendation for how to fix the issue, e.g. 'increase top_k to 3-5'>"
}"""


def validate_diagnosis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a diagnosis output dictionary matches the strict JSON schema.

    Args:
        data: Parsed dictionary from diagnostic LLM output.

    Returns:
        Validated dictionary with correctly typed fields.

    Raises:
        ValueError: If required keys are missing, values are out of bounds, or cause is invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected diagnosis data to be a dictionary, got {type(data).__name__}")

    required_keys = {"suspected_cause", "evidence", "confidence", "hypothesis", "recommended_experiment"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Diagnosis missing required keys: {missing}")

    cause = str(data["suspected_cause"]).strip()
    if cause not in DIAGNOSIS_CATEGORIES:
        # Check case-insensitive match
        matched = next((c for c in DIAGNOSIS_CATEGORIES if c.lower() == cause.lower()), None)
        if matched:
            cause = matched
        else:
            raise ValueError(
                f"Invalid suspected_cause '{cause}'. Must be one of: {DIAGNOSIS_CATEGORIES}"
            )

    evidence = str(data["evidence"]).strip()
    if not evidence:
        raise ValueError("Diagnosis 'evidence' must not be empty.")
    if len(evidence) < 15:
        raise ValueError("Diagnosis 'evidence' is too brief; must provide substantiated reasoning.")

    # Check for substantiated data references
    substantiating_terms = {
        "chunk", "overlap", "top_k", "retriev", "query", "queries", "score", "config",
        "context", "token", "passage", "size", "ground", "sentence", "document", "embed",
        "index", "faithfulness", "relevan"
    }
    evidence_lower = evidence.lower()
    if not any(term in evidence_lower for term in substantiating_terms):
        raise ValueError(
            "Diagnosis 'evidence' must be substantiated with concrete data references "
            "(e.g., chunk size, overlap, top_k, query context, or observed scores)."
        )

    hypothesis = str(data.get("hypothesis", "")).strip()
    if not hypothesis:
        raise ValueError("Diagnosis 'hypothesis' must not be empty.")

    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        raise ValueError(f"Diagnosis 'confidence' must be a float, got {data['confidence']}")

    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"Diagnosis 'confidence' must be between 0.0 and 1.0, got {confidence}")

    recommended_exp = str(data["recommended_experiment"]).strip()
    if not recommended_exp:
        raise ValueError("Diagnosis 'recommended_experiment' must not be empty.")

    return {
        "suspected_cause": cause,
        "evidence": evidence,
        "confidence": confidence,
        "hypothesis": hypothesis,
        "recommended_experiment": recommended_exp,
    }


def fetch_diagnostic_context(index_version: Optional[str] = None) -> Dict[str, Any]:
    """Gather health status, baseline/active configs, and failing queries for diagnosis.

    Uses the MCP tool functions directly in-process for deterministic data retrieval.
    """
    from mcp_server import (
        get_active_version,
        get_baseline_version,
        get_failed_queries,
        get_pipeline_config,
        inspect_rag_health,
    )

    active_ver = index_version.strip() if index_version and index_version.strip() else get_active_version()
    baseline_ver = get_baseline_version()

    # 1. Health inspection
    health_raw = inspect_rag_health(active_ver)
    try:
        health_data = json.loads(health_raw)
    except Exception:
        health_data = {"raw": health_raw}

    # 2. Configs
    cfg_raw = get_pipeline_config(active_ver)
    try:
        active_config = json.loads(cfg_raw)
    except Exception:
        active_config = {"error": cfg_raw}

    base_raw = get_pipeline_config(baseline_ver)
    try:
        baseline_config = json.loads(base_raw)
    except Exception:
        baseline_config = {"error": base_raw}

    # 3. Failing queries
    failed_raw = get_failed_queries(active_ver)
    try:
        failing_queries = json.loads(failed_raw)
    except Exception:
        failing_queries = []

    return {
        "active_version": active_ver,
        "baseline_version": baseline_ver,
        "health": health_data,
        "baseline_config": baseline_config,
        "active_config": active_config,
        "failing_queries": failing_queries if isinstance(failing_queries, list) else [],
    }


def format_diagnostic_prompt(context: Dict[str, Any]) -> str:
    """Format structured diagnostic context into a clear analysis prompt."""
    baseline_ver = context.get("baseline_version", "v001")
    active_ver = context.get("active_version", "active")
    baseline_cfg = context.get("baseline_config", {})
    active_cfg = context.get("active_config", {})
    failing_queries = context.get("failing_queries", [])

    prompt_parts = [
        "Pipeline Configurations:",
        f"Baseline Config ({baseline_ver}): {json.dumps(baseline_cfg, indent=2)}",
        f"Active Degraded Config ({active_ver}): {json.dumps(active_cfg, indent=2)}\n",
        f"Failing Queries Analysis ({len(failing_queries)} failing):\n",
    ]

    for q in failing_queries:
        prompt_parts.append(f"Query ID: {q.get('id')}")
        prompt_parts.append(f"Question: {q.get('question')}")
        prompt_parts.append(f"Expected Answer: {q.get('expected_answer')}")
        prompt_parts.append(f"Generated Answer: {q.get('generated_answer')}")
        prompt_parts.append(f"Retrieved Chunks (Total: {len(q.get('retrieved_chunks', []))}):")
        for i, chunk in enumerate(q.get("retrieved_chunks", [])):
            metadata = chunk.get("metadata", {})
            prompt_parts.append(
                f"  - Chunk {i+1} (Size: {metadata.get('chunk_size')}, Overlap: {metadata.get('overlap')}): {chunk.get('content')}"
            )
        prompt_parts.append(f"Faithfulness Score: {q.get('faithfulness')} - {q.get('faithfulness_reasoning')}")
        prompt_parts.append(f"Answer Relevancy Score: {q.get('answer_relevancy')} - {q.get('relevancy_reasoning')}")
        prompt_parts.append("-" * 40)

    prompt_parts.append(
        "\nBased on the configurations and the failing queries (note differences in chunk size, overlap, "
        "top_k / number of retrieved chunks, and the content grounding), diagnose the root cause and output the JSON response."
    )

    return "\n".join(prompt_parts)


def run_local_diagnose(
    index_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Lightweight local smoke-test and debugging runner for diagnosis.

    Note: TrueForge handles production diagnosis loops. This function is for
    local validation and quick testing without running container infrastructure.
    """
    context = fetch_diagnostic_context(index_version)
    failing = context.get("failing_queries", [])

    if not failing:
        return {"status": "healthy", "message": f"No failing queries found for {context['active_version']}."}

    prompt = format_diagnostic_prompt(context)

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return {
            "status": "context_only",
            "context": context,
            "prompt": prompt,
            "note": "GROQ_API_KEY not set. Returning extracted context without LLM inference.",
        }

    from pipeline.llm_client import LLMClient

    llm = LLMClient(api_key=key, model_name=model_name)
    messages = [
        {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    parsed = llm.complete_json(messages, temperature=0.0)
    return validate_diagnosis(parsed)

    return {}


async def run_diagnose_subagent(index_version: Optional[str] = None) -> Dict[str, Any]:
    """Backward-compatible async entrypoint for legacy scripts."""
    return run_local_diagnose(index_version)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    target_index = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_local_diagnose(target_index)
    print("\n=== Diagnosis Result ===")
    print(json.dumps(result, indent=2))

