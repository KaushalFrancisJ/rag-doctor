import json
import sys
from pathlib import Path
from mcp.server.mcpserver import MCPServer

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.index import VectorIndex
from pipeline.retriever import Retriever

# Initialize MCPServer
mcp = MCPServer("rag-doctor")

def _get_results_file(index_version: str) -> Path:
    """Helper to locate the results file based on index version."""
    if index_version == "v001":
        return PROJECT_ROOT / "eval" / "baseline_results.json"
    elif index_version == "v002_sick":
        return PROJECT_ROOT / "eval" / "sick_results.json"
    else:
        # Fallback or generic path
        return PROJECT_ROOT / "eval" / f"{index_version}_results.json"

@mcp.tool()
def get_evaluation_results(index_version: str) -> str:
    """
    Get the summary evaluation results for a specific index version.
    Returns faithfulness, relevancy, overall scores, and pass/fail counts.
    """
    results_file = _get_results_file(index_version)
    if not results_file.exists():
        return f"Error: Evaluation results for {index_version} not found at {results_file}."
    
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
        return json.dumps(data.get("summary", {}), indent=2)
    except Exception as e:
        return f"Error reading results: {e}"

@mcp.tool()
def get_failed_queries(index_version: str) -> str:
    """
    Get the details of failing queries for a specific index version.
    Includes the query, expected answer, generated answer, and retrieved chunks.
    """
    results_file = _get_results_file(index_version)
    if not results_file.exists():
        return f"Error: Evaluation results for {index_version} not found at {results_file}."
    
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
        return json.dumps(data.get("failing_queries", []), indent=2)
    except Exception as e:
        return f"Error reading failing queries: {e}"

@mcp.tool()
def inspect_retrieval(query: str, index_version: str, top_k: int = 3) -> str:
    """
    Run retrieval for a specific query against a specific index version.
    Useful for ad-hoc debugging of what the index returns.
    """
    index_dir = PROJECT_ROOT / "indexes" / index_version
    if not index_dir.exists():
        return f"Error: Index version {index_version} not found at {index_dir}."
    
    try:
        index = VectorIndex.load(index_dir)
        retriever = Retriever(index=index, default_top_k=top_k)
        chunks = retriever.retrieve(query, top_k=top_k)
        return json.dumps(chunks, indent=2)
    except Exception as e:
        return f"Error inspecting retrieval: {e}"

@mcp.tool()
def get_pipeline_config(index_version: str) -> str:
    """
    Get the pipeline configuration (chunk size, overlap, embedding model, etc.)
    used to create a specific index version.
    """
    config_file = PROJECT_ROOT / "indexes" / index_version / "config.json"
    if not config_file.exists():
        return f"Error: Config for {index_version} not found at {config_file}."
    
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        return json.dumps(config, indent=2)
    except Exception as e:
        return f"Error reading config: {e}"

import argparse

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
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)

