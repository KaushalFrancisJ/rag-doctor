import asyncio
import json
import os
import sys
import re
import time
from typing import Dict, Any

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

async def run_diagnose_subagent(index_version: str) -> Dict[str, Any]:
    """
    Connects to the RAG Doctor MCP server to gather pipeline context and failing queries,
    then uses Groq to diagnose the problem.
    """
    server_params = StdioServerParameters(
        command=os.environ.get("PYTHON_EXE", "python"),
        args=["mcp_server.py"]
    )
    
    # Using MCP to fetch data
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 1. Get baseline config (v001) and current config (index_version)
            try:
                v001_result = await session.call_tool("get_pipeline_config", {"index_version": "v001"})
                text = v001_result.content[0].text
                if text.startswith("Error:"):
                    raise ValueError(text)
                v001_config = json.loads(text)
            except Exception as e:
                v001_config = {"error": str(e)}

            try:
                current_result = await session.call_tool("get_pipeline_config", {"index_version": index_version})
                text = current_result.content[0].text
                if text.startswith("Error:"):
                    raise ValueError(text)
                current_config = json.loads(text)
            except Exception as e:
                current_config = {"error": str(e)}
            
            # 2. Get failing queries
            try:
                failed_result = await session.call_tool("get_failed_queries", {"index_version": index_version})
                text = failed_result.content[0].text
                if text.startswith("Error:"):
                    raise ValueError(text)
                failing_queries = json.loads(text)
            except Exception as e:
                print(f"Error fetching failed queries: {e}")
                return {"error": "Failed to fetch failing queries", "details": str(e)}

    if not failing_queries:
        print("No failing queries found. System is healthy!")
        return {"status": "healthy"}

    # 3. Construct the prompt
    prompt = f"Pipeline Configurations:\n"
    prompt += f"Baseline Config (v001): {json.dumps(v001_config, indent=2)}\n"
    prompt += f"Current Sick Config ({index_version}): {json.dumps(current_config, indent=2)}\n\n"
    prompt += f"Failing Queries Analysis:\n"
    
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
        prompt += "-" * 40 + "\n"
        
    prompt += "\nBased on the configurations and the failing queries (especially note the chunk sizes, overlaps, and the fragmented content), diagnose the root cause and output the JSON response."

    # 4. Make LLM call
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
    
    # Accept index version from CLI args if provided, else default to v002_sick
    target_index = sys.argv[1] if len(sys.argv) > 1 else "v002_sick"
    
    # Run the Diagnose Subagent
    print(f"Running Diagnose subagent against index version: {target_index}")
    diagnosis_result = asyncio.run(run_diagnose_subagent(target_index))
    if diagnosis_result:
        print("\n=== Diagnose Subagent Result ===")
        print(json.dumps(diagnosis_result, indent=2))
