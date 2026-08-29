import asyncio
import json
import os
import sys
import re
import time
from typing import Dict, Any, Optional

from groq import Groq
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

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
  "recommended_experiment": "<concise recommendation for how to fix the issue, e.g., 'change chunk size to 500 and overlap to 100'>"
}"""

async def run_diagnose_subagent(index_version: Optional[str] = None) -> Dict[str, Any]:
    """
    Connects to the RAG Doctor MCP server to inspect the active RAG pipeline health,
    gathers failing queries and configuration diffs, then diagnoses the root cause.
    """
    server_params = StdioServerParameters(
        command=os.environ.get("PYTHON_EXE", "python"),
        args=["mcp_server.py"]
    )
    
    # Using MCP to fetch data
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Inspect RAG health to discover active and baseline index versions
            health_params = {"index_version": index_version} if index_version else {}
            health_result = await session.call_tool("inspect_rag_health", health_params)
            health_data = json.loads(health_result.content[0].text)

            active_ver = health_data.get("active_monitored_index")
            baseline_ver = health_data.get("baseline_index", "v001")

            print(f"Health Status: {health_data.get('status')}")
            print(f"Active Index: {active_ver} (Score: {health_data.get('current_score')})")
            print(f"Baseline Index: {baseline_ver} (Score: {health_data.get('baseline_score')})")

            # 2. Get baseline and active configurations
            try:
                b_res = await session.call_tool("get_pipeline_config", {"index_version": baseline_ver})
                b_text = b_res.content[0].text
                baseline_config = json.loads(b_text) if not b_text.startswith("Error:") else {"error": b_text}
            except Exception as e:
                baseline_config = {"error": str(e)}

            try:
                a_res = await session.call_tool("get_pipeline_config", {"index_version": active_ver})
                a_text = a_res.content[0].text
                active_config = json.loads(a_text) if not a_text.startswith("Error:") else {"error": a_text}
            except Exception as e:
                active_config = {"error": str(e)}

            # 3. Get failing queries for active index
            try:
                failed_res = await session.call_tool("get_failed_queries", {"index_version": active_ver})
                f_text = failed_res.content[0].text
                if f_text.startswith("Error:"):
                    raise ValueError(f_text)
                failing_queries = json.loads(f_text)
            except Exception as e:
                print(f"Error fetching failed queries: {e}")
                return {"error": "Failed to fetch failing queries", "details": str(e)}

    if not failing_queries:
        print("No failing queries found. System is healthy!")
        return {"status": "healthy"}

    # 4. Construct the prompt
    prompt = f"Pipeline Configurations:\n"
    prompt += f"Baseline Config ({baseline_ver}): {json.dumps(baseline_config, indent=2)}\n"
    prompt += f"Active Degraded Config ({active_ver}): {json.dumps(active_config, indent=2)}\n\n"
    prompt += f"Failing Queries Analysis ({len(failing_queries)} failing):\n"
    
    for q in failing_queries:
        prompt += f"Query ID: {q.get('id')}\n"
        prompt += f"Question: {q.get('question')}\n"
        prompt += f"Expected Answer: {q.get('expected_answer')}\n"
        prompt += f"Generated Answer: {q.get('generated_answer')}\n"
        prompt += "Retrieved Chunks:\n"
        for i, chunk in enumerate(q.get("retrieved_chunks", [])):
            metadata = chunk.get("metadata", {})
            prompt += f"  - Chunk {i+1} (Size: {metadata.get('chunk_size')}, Overlap: {metadata.get('overlap')}): {chunk.get('content')}\n"
        prompt += f"Faithfulness Score: {q.get('faithfulness')} - {q.get('faithfulness_reasoning')}\n"
        prompt += f"Answer Relevancy Score: {q.get('answer_relevancy')} - {q.get('relevancy_reasoning')}\n"
    prompt += "\nBased on the configurations and the failing queries (note differences in chunk size, overlap, top_k / number of retrieved chunks, and the content grounding), diagnose the root cause and output the JSON response."

    # 5. Make LLM call
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")
        
    client = Groq(api_key=api_key)
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    print("Submitting diagnosis request to Groq...")
    max_retries = 6
    response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
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
                m_min = re.search(r"try again in (\d+)m([\d\.]+)s", str(e), re.I)
                m_sec = re.search(r"try again in ([\d\.]+)s", str(e), re.I)
                m_ms = re.search(r"try again in ([\d\.]+)ms", str(e), re.I)
                if m_min:
                    wait_sec = int(m_min.group(1)) * 60 + float(m_min.group(2)) + 1.0
                elif m_sec:
                    wait_sec = float(m_sec.group(1)) + 1.0
                elif m_ms:
                    wait_sec = float(m_ms.group(1)) / 1000.0 + 1.0
                print(f"Rate limited, sleeping for {wait_sec} seconds...")
                time.sleep(wait_sec)
            else:
                raise

    if response:
        content = response.choices[0].message.content
        diagnosis = json.loads(content)
        return diagnosis
    return {}

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Using python from virtualenv for MCP Server
    os.environ["PYTHON_EXE"] = sys.executable if hasattr(sys, "executable") else "python"
    
    # Accept index version from CLI args if provided, else let MCP dynamically resolve active index
    target_index = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run the Diagnose Subagent
    diagnosis_result = asyncio.run(run_diagnose_subagent(target_index))
    if diagnosis_result:
        print("\n=== Diagnose Subagent Result ===")
        print(json.dumps(diagnosis_result, indent=2))
