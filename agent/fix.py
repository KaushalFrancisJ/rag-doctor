"""Fix client and experiment validation utilities for RAG Doctor.

This module provides helper utilities to:
1. Validate structured fix experiment proposals against the MVP schema.
2. Format diagnosis and pipeline configuration into targeted fix prompts.
3. Provide a lightweight local test/smoke-test runner for experiment generation.

Note: Production agent orchestration, subagent loops, model calls, Daytona sandbox
execution, and human approval gates are handled by TrueForge.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_STRATEGIES = ["chunking", "retrieval"]

INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "fix_instructions.md"

if INSTRUCTIONS_PATH.exists():
    with open(INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
        FIX_SYSTEM_PROMPT = f.read().strip()
else:
    FIX_SYSTEM_PROMPT = """You are the Fix Subagent for RAG Doctor.
Your mission is to turn a root-cause diagnosis into a concrete, isolated candidate experiment specification.

Supported strategies:
- chunking (chunk_size, overlap)
- retrieval (top_k)

Output JSON schema:
{
  "hypothesis": "...",
  "strategy": "chunking | retrieval",
  "changes": { ... },
  "expected_effect": "...",
  "reasoning": "..."
}"""


def validate_fix_experiment(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a fix experiment output dictionary matches the strict MVP schema.

    Args:
        data: Parsed dictionary from fix LLM output.

    Returns:
        Validated dictionary with correctly typed fields.

    Raises:
        ValueError: If required keys are missing, strategy is invalid, or changes are malformed.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected fix experiment data to be a dictionary, got {type(data).__name__}")

    required_keys = {"hypothesis", "strategy", "changes", "expected_effect", "reasoning"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Fix experiment missing required keys: {missing}")

    hypothesis = str(data["hypothesis"]).strip()
    if not hypothesis:
        raise ValueError("Fix experiment 'hypothesis' must not be empty.")

    strategy = str(data["strategy"]).strip().lower()
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {SUPPORTED_STRATEGIES}")

    changes = data.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise ValueError("Fix experiment 'changes' must be a non-empty dictionary.")

    # Strategy-specific parameter validation
    allowed_params = {
        "chunking": {"chunk_size", "overlap"},
        "retrieval": {"top_k"},
    }
    valid_keys = allowed_params[strategy]
    for key, val in changes.items():
        if key not in valid_keys:
            raise ValueError(
                f"Parameter '{key}' is not allowed for strategy '{strategy}'. Allowed: {valid_keys}"
            )
        try:
            int_val = int(val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Parameter '{key}' value must be an integer, got {val}") from e

        if int_val <= 0 and key != "overlap":
            raise ValueError(f"Parameter '{key}' value must be a positive integer, got {val}")
        if key == "overlap" and int_val < 0:
            raise ValueError(f"Parameter 'overlap' must be non-negative, got {val}")

    expected_effect = str(data["expected_effect"]).strip()
    if not expected_effect:
        raise ValueError("Fix experiment 'expected_effect' must not be empty.")

    reasoning = str(data["reasoning"]).strip()
    if not reasoning:
        raise ValueError("Fix experiment 'reasoning' must not be empty.")

    return {
        "hypothesis": hypothesis,
        "strategy": strategy,
        "changes": {k: int(v) for k, v in changes.items()},
        "expected_effect": expected_effect,
        "reasoning": reasoning,
    }


def format_fix_prompt(
    diagnosis: Dict[str, Any],
    baseline_config: Dict[str, Any],
    active_config: Dict[str, Any],
) -> str:
    """Format structured diagnosis and configs into a fix reasoning prompt."""
    prompt_parts = [
        "Diagnosed Pipeline Degradation:",
        f"- Suspected Cause: {diagnosis.get('suspected_cause', 'Unknown')}",
        f"- Evidence: {diagnosis.get('evidence', 'None')}",
        f"- Confidence: {diagnosis.get('confidence', 0.0)}",
        f"- Diagnostic Hypothesis: {diagnosis.get('hypothesis', 'None')}",
        f"- Recommended Experiment: {diagnosis.get('recommended_experiment', 'None')}",
        "\nConfigurations:",
        f"Baseline Config: {json.dumps(baseline_config, indent=2)}",
        f"Active Config: {json.dumps(active_config, indent=2)}",
        "\nTask:",
        "Propose an isolated candidate experiment specification to remediate this degradation.",
        "Choose either 'chunking' or 'retrieval' strategy, make the minimal necessary parameter changes, and return strict JSON.",
    ]
    return "\n".join(prompt_parts)


def run_local_fix(
    diagnosis: Dict[str, Any],
    baseline_config: Optional[Dict[str, Any]] = None,
    active_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Lightweight local smoke-test runner for fix agent experiment formulation."""
    from mcp_server import get_active_version, get_baseline_version, get_pipeline_config

    if baseline_config is None:
        base_ver = get_baseline_version()
        try:
            baseline_config = json.loads(get_pipeline_config(base_ver))
        except Exception:
            baseline_config = {"chunk_size": 500, "overlap": 100, "top_k": 3}

    if active_config is None:
        act_ver = get_active_version()
        try:
            active_config = json.loads(get_pipeline_config(act_ver))
        except Exception:
            active_config = {}

    prompt = format_fix_prompt(diagnosis, baseline_config, active_config)

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return {
            "status": "context_only",
            "prompt": prompt,
            "note": "GROQ_API_KEY not set. Returning prompt without LLM inference.",
        }

    from groq import Groq

    model = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=key)

    max_retries = 5
    response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FIX_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            break
        except Exception as e:
            err_str = str(e).lower()
            if ("rate_limit" in err_str or "429" in err_str or "retry" in err_str) and attempt < max_retries - 1:
                wait_sec = 4.0 * (attempt + 1)
                time.sleep(wait_sec)
            else:
                raise

    if response:
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return validate_fix_experiment(parsed)

    return {}


if __name__ == "__main__":
    from dotenv import load_dotenv
    from agent.diagnose import run_local_diagnose

    load_dotenv()

    target_index = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"--- Running Diagnose on {target_index or 'active'} ---")
    diag = run_local_diagnose(target_index)
    print(json.dumps(diag, indent=2))

    if diag.get("suspected_cause"):
        print(f"\n--- Running Fix Subagent on {diag['suspected_cause']} ---")
        fix_result = run_local_fix(diag)
        print(json.dumps(fix_result, indent=2))
