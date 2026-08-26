import json
import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

def prototype_diagnosis():
    # 1. Load sick results
    sick_results_path = PROJECT_ROOT / "eval" / "sick_results.json"
    with open(sick_results_path, "r", encoding="utf-8") as f:
        sick_results = json.load(f)
    
    failing_queries = sick_results.get("failing_queries", [])
    if not failing_queries:
        print("No failing queries found.")
        return
    
    # 2. Load configs
    try:
        with open(PROJECT_ROOT / "indexes" / "v001" / "config.json", "r") as f:
            v001_config = json.load(f)
    except Exception as e:
        v001_config = {"error": str(e)}

    try:
        with open(PROJECT_ROOT / "indexes" / "v002_sick" / "config.json", "r") as f:
            v002_sick_config = json.load(f)
    except Exception as e:
        v002_sick_config = {"error": str(e)}
        
    # 3. Construct prompt
    prompt = f"Pipeline Configurations:\n"
    prompt += f"Baseline Config (v001): {json.dumps(v001_config, indent=2)}\n"
    prompt += f"Current Sick Config (v002_sick): {json.dumps(v002_sick_config, indent=2)}\n\n"
    prompt += f"Failing Queries Analysis:\n"
    
    for q in failing_queries:
        prompt += f"Query ID: {q.get('id')}\n"
        prompt += f"Question: {q.get('question')}\n"
        prompt += f"Expected Answer: {q.get('expected_answer')}\n"
        prompt += f"Generated Answer: {q.get('generated_answer')}\n"
        prompt += "Retrieved Chunks:\n"
        for i, chunk in enumerate(q.get("retrieved_chunks", [])):
            prompt += f"  - Chunk {i+1} (Size: {chunk.get('metadata', {}).get('chunk_size')}, Overlap: {chunk.get('metadata', {}).get('overlap')}): {chunk.get('content')}\n"
        prompt += f"Faithfulness Score: {q.get('faithfulness')} - {q.get('faithfulness_reasoning')}\n"
        prompt += f"Answer Relevancy Score: {q.get('answer_relevancy')} - {q.get('relevancy_reasoning')}\n"
        prompt += "-" * 40 + "\n"
        
    prompt += "\nBased on the configurations and the failing queries (especially note the chunk sizes, overlaps, and the fragmented content), diagnose the root cause and output the JSON response."

    # 4. Make LLM call
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set.")
        return
        
    client = Groq(api_key=api_key)
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    print("Sending prompt to LLM...")
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
        print("\n=== Diagnosis Result ===")
        print(content)
        
if __name__ == "__main__":
    prototype_diagnosis()
