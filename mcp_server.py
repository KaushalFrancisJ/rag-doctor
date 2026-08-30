import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pipeline.index import VectorIndex
from pipeline.retriever import Retriever

# Initialize MCPServer
mcp = MCPServer("rag-doctor")

def get_active_state() -> dict:
    """Read active and baseline index configuration from indexes/active.json."""
    active_file = PROJECT_ROOT / "indexes" / "active.json"
    if active_file.exists():
        try:
            with open(active_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"active_version": "v001", "baseline_version": "v001"}

def get_active_version() -> str:
    """Read the active index version from indexes/active.json, fallback to v001."""
    return get_active_state().get("active_version", "v001")

def get_baseline_version() -> str:
    """Read the baseline index version from indexes/active.json, fallback to v001."""
    return get_active_state().get("baseline_version", "v001")

def _get_results_file(index_version: str) -> Path:
    """Helper to locate evaluation results for an index version by convention."""
    return PROJECT_ROOT / "eval" / f"{index_version}_results.json"

@mcp.tool()
def inspect_rag_health(index_version: str = "") -> str:
    """
    Inspect the overall health and status of the RAG pipeline.
    If index_version is omitted, inspects the current ACTIVE index.
    Returns current active version, baseline comparison, degradation status,
    failing query counts, and available index versions.
    """
    active_ver = index_version.strip() if index_version.strip() else get_active_version()
    baseline_ver = get_baseline_version()

    indexes_dir = PROJECT_ROOT / "indexes"
    available_indexes = [d.name for d in indexes_dir.iterdir() if d.is_dir()] if indexes_dir.exists() else []

    # Check baseline evaluation
    baseline_eval = _get_results_file(baseline_ver)
    baseline_score = None
    if baseline_eval.exists():
        try:
            with open(baseline_eval, "r", encoding="utf-8") as f:
                b_data = json.load(f)
                baseline_score = b_data.get("summary", {}).get("avg_overall_score")
        except Exception:
            pass

    # Check active / specified index evaluation
    active_eval = _get_results_file(active_ver)
    active_summary = {}
    if active_eval.exists():
        try:
            with open(active_eval, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                active_summary = c_data.get("summary", {})
        except Exception:
            pass

    report = {
        "status": "DEGRADED" if active_summary.get("is_degraded") else "HEALTHY",
        "active_monitored_index": active_ver,
        "baseline_index": baseline_ver,
        "baseline_score": baseline_score,
        "current_score": active_summary.get("avg_overall_score"),
        "score_threshold": active_summary.get("threshold", 3.5),
        "failing_queries_count": active_summary.get("failing_count", 0),
        "total_queries_evaluated": active_summary.get("total_queries", 0),
        "available_index_versions": sorted(available_indexes),
        "instruction": f"Call get_failed_queries('{active_ver}') and compare get_pipeline_config('{active_ver}') with get_pipeline_config('{baseline_ver}') to diagnose the root cause."
    }
    return json.dumps(report, indent=2)

@mcp.tool()
def get_evaluation_results(index_version: str = "") -> str:
    """
    Get the summary evaluation results for a specific index version.
    Defaults to the currently active index version if not specified.
    """
    ver = index_version.strip() if index_version.strip() else get_active_version()
    results_file = _get_results_file(ver)
    if not results_file.exists():
        indexes_dir = PROJECT_ROOT / "indexes"
        available = [d.name for d in indexes_dir.iterdir() if d.is_dir()] if indexes_dir.exists() else []
        return f"Error: Evaluation results for '{ver}' not found at {results_file}. Available index versions: {available}"

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data.get("summary", {}), indent=2)
    except Exception as e:
        return f"Error reading results: {e}"

@mcp.tool()
def get_failed_queries(index_version: str = "") -> str:
    """
    Get the details of failing queries for a specific index version.
    Includes the query, expected answer, generated answer, and retrieved chunks.
    Defaults to the currently active index version if not specified.
    """
    ver = index_version.strip() if index_version.strip() else get_active_version()
    results_file = _get_results_file(ver)
    if not results_file.exists():
        indexes_dir = PROJECT_ROOT / "indexes"
        available = [d.name for d in indexes_dir.iterdir() if d.is_dir()] if indexes_dir.exists() else []
        return f"Error: Evaluation results for '{ver}' not found at {results_file}. Available index versions: {available}"

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data.get("failing_queries", []), indent=2)
    except Exception as e:
        return f"Error reading failing queries: {e}"

@mcp.tool()
def inspect_retrieval(query: str, index_version: str = "", top_k: int = 0) -> str:
    """
    Run retrieval for a specific query against a specific index version.
    Defaults to the currently active index version if not specified.
    """
    ver = index_version.strip() if index_version.strip() else get_active_version()
    index_dir = PROJECT_ROOT / "indexes" / ver
    if not index_dir.exists():
        indexes_dir = PROJECT_ROOT / "indexes"
        available = [d.name for d in indexes_dir.iterdir() if d.is_dir()] if indexes_dir.exists() else []
        return f"Error: Index version '{ver}' not found at {index_dir}. Available index versions: {available}"

    try:
        index = VectorIndex.load(index_dir)
        effective_k = top_k if top_k > 0 else int(index.config.get("top_k", 3))
        retriever = Retriever(index=index, default_top_k=effective_k)
        chunks = retriever.retrieve(query, top_k=effective_k)
        return json.dumps(chunks, indent=2)
    except Exception as e:
        return f"Error inspecting retrieval: {e}"


@mcp.tool()
def query_rag_pipeline(query: str, index_version: str = "", top_k: int = 0) -> str:
    """
    Ask a question directly to the Patient RAG pipeline.
    Retrieves context from the active (or specified) vector index and generates an answer using Groq.
    Returns the generated answer, the index version used, and retrieved chunk references.
    """
    from pipeline.generator import Generator

    ver = index_version.strip() if index_version.strip() else get_active_version()
    index_dir = PROJECT_ROOT / "indexes" / ver
    if not index_dir.exists():
        indexes_dir = PROJECT_ROOT / "indexes"
        available = [d.name for d in indexes_dir.iterdir() if d.is_dir()] if indexes_dir.exists() else []
        return f"Error: Index version '{ver}' not found. Available versions: {available}"

    try:
        index = VectorIndex.load(index_dir)
        effective_k = top_k if top_k > 0 else int(index.config.get("top_k", 3))
        retriever = Retriever(index=index, default_top_k=effective_k)
        chunks = retriever.retrieve(query, top_k=effective_k)

        generator = Generator()
        res = generator.generate(query=query, retrieved_chunks=chunks)

        return json.dumps({
            "index_version": ver,
            "query": query,
            "answer": res.get("answer", ""),
            "retrieved_chunks_count": len(chunks),
            "sources": [c.get("metadata", {}).get("relative_path", "unknown") for c in chunks],
            "chunks": chunks,
        }, indent=2)
    except Exception as e:
        return f"Error querying RAG pipeline: {e}"

@mcp.tool()
def compare_experiments(candidate_version: str, current_version: str = "", baseline_version: str = "") -> str:
    """
    Compare a candidate experiment's evaluation results against the current active
    (or specified) degraded index and the baseline index.
    Returns structured JSON with baseline_score, current_score, candidate_score,
    delta_vs_current, delta_vs_baseline, candidate_better, and regressions.
    """
    from eval.compare import compare_evaluation_reports, load_evaluation_report

    cand_ver = candidate_version.strip()
    curr_ver = current_version.strip() if current_version.strip() else get_active_version()
    base_ver = baseline_version.strip() if baseline_version.strip() else get_baseline_version()

    try:
        cand_report = load_evaluation_report(cand_ver)
    except Exception as e:
        return f"Error loading candidate evaluation for '{cand_ver}': {e}"

    try:
        curr_report = load_evaluation_report(curr_ver)
    except Exception as e:
        return f"Error loading current evaluation for '{curr_ver}': {e}"

    try:
        base_report = load_evaluation_report(base_ver)
    except Exception:
        base_report = curr_report

    comparison = compare_evaluation_reports(
        candidate_report=cand_report,
        current_report=curr_report,
        baseline_report=base_report,
    )
    return json.dumps(comparison, indent=2)


@mcp.tool()
def create_experiment(
    strategy: str,
    output_version: str = "",
    chunk_size: int = 0,
    overlap: int = 0,
    top_k: int = 0,
    source_index: str = "",
) -> str:
    """
    Execute sandbox remediation to create an isolated candidate index, evaluate it against
    the evaluation dataset, and return the structured comparison against current degraded and baseline.
    Supported strategies: 'chunking', 'retrieval'.
    Never modifies the active index.
    """
    from sandbox_scripts.remediate import execute_remediation
    from eval.compare import evaluate_and_compare_candidate

    strat = strategy.strip().lower()
    if strat not in ("chunking", "retrieval"):
        return f"Error: Invalid strategy '{strategy}'. Supported strategies: ['chunking', 'retrieval']"

    active_ver = get_active_version()
    src_ver = source_index.strip() if source_index.strip() else active_ver
    out_ver = output_version.strip() if output_version.strip() else None

    kwargs: dict = {
        "strategy": strat,
        "source_index": src_ver,
        "output_version": out_ver,
        "corpus_source_dir": PROJECT_ROOT / "corpus" / "active",
        "corpus_working_dir": PROJECT_ROOT / "corpus" / "candidate",
        "indexes_root_dir": PROJECT_ROOT / "indexes",
    }
    if chunk_size > 0:
        kwargs["chunk_size"] = chunk_size
    if overlap > 0:
        kwargs["overlap"] = overlap
    if top_k > 0:
        kwargs["top_k"] = top_k

    try:
        remed_res = execute_remediation(**kwargs)
        cand_version = remed_res["candidate_version"]
        cand_dir = Path(remed_res["candidate_index_dir"])

        cand_eval_output = PROJECT_ROOT / "eval" / f"{cand_version}_results.json"
        comparison = evaluate_and_compare_candidate(
            candidate_index_dir=cand_dir,
            current_index_version=src_ver,
            baseline_index_version=get_baseline_version(),
            eval_set_path=PROJECT_ROOT / "eval" / "eval_set.json",
            output_report_path=cand_eval_output,
            top_k=top_k if top_k > 0 else None,
            eval_limit=6,
        )
        comparison["remediation_result"] = remed_res
        return json.dumps(comparison, indent=2)
    except Exception as e:
        return f"Error creating candidate experiment: {e}"


@mcp.tool()
def get_pipeline_config(index_version: str = "") -> str:
    """
    Get the pipeline configuration (chunk size, overlap, embedding model, etc.)
    for a specific index version. Defaults to currently active index if not specified.
    """
    ver = index_version.strip() if index_version.strip() else get_active_version()
    config_file = PROJECT_ROOT / "indexes" / ver / "config.json"
    if not config_file.exists():
        indexes_dir = PROJECT_ROOT / "indexes"
        available = [d.name for d in indexes_dir.iterdir() if d.is_dir()] if indexes_dir.exists() else []
        return f"Error: Config for '{ver}' not found at {config_file}. Available index versions: {available}"

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return json.dumps(config, indent=2)
    except Exception as e:
        return f"Error reading config: {e}"


@mcp.tool()
def promote_experiment(candidate_version: str, metadata: str = "") -> str:
    """
    Promote a candidate experiment index to active status after explicit human approval.
    Updates indexes/active.json, preserves previous active version,
    and records promotion metadata. Never promote automatically.
    """
    from agent.orchestrator import promote_candidate

    cand_ver = candidate_version.strip()
    if not cand_ver:
        return "Error: candidate_version cannot be empty."

    meta_dict = None
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except Exception:
            meta_dict = {"raw_metadata": metadata}

    try:
        res = promote_candidate(cand_ver, indexes_dir=PROJECT_ROOT / "indexes", metadata=meta_dict)
        return json.dumps(res, indent=2)
    except Exception as e:
        return f"Error promoting candidate '{cand_ver}': {e}"


@mcp.tool()
def rollback_experiment(target_version: str = "") -> str:
    """
    Roll back the active index to the previous active version or a specified target version.
    Preserves version history and logs rollback metadata.
    """
    from agent.orchestrator import rollback_candidate

    try:
        res = rollback_candidate(target_version=target_version.strip() or None, indexes_dir=PROJECT_ROOT / "indexes")
        return json.dumps(res, indent=2)
    except Exception as e:
        return f"Error rolling back index: {e}"


@mcp.tool()
def discard_experiment(candidate_version: str, reason: str = "") -> str:
    """
    Discard a rejected candidate experiment index without modifying active index.
    """
    from agent.orchestrator import discard_candidate

    cand_ver = candidate_version.strip()
    if not cand_ver:
        return "Error: candidate_version cannot be empty."

    try:
        res = discard_candidate(cand_ver, indexes_dir=PROJECT_ROOT / "indexes", reason=reason.strip() or None)
        return json.dumps(res, indent=2)
    except Exception as e:
        return f"Error discarding candidate '{cand_ver}': {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Doctor MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE/HTTP transport (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE/HTTP transport (default: 8000)")
    parser.add_argument("--sse-path", default="/sse", help="SSE endpoint path (default: /sse, use / if client expects root)")
    args = parser.parse_args()

    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port, sse_path=args.sse_path, transport_security=security)
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port, transport_security=security)
