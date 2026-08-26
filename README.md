# RAG Doctor 🩺

Self-healing RAG pipeline agent built for the TrueForge Hackathon.

RAG Doctor continuously monitors a RAG pipeline's answer quality using an LLM-as-judge, diagnoses root causes when quality degrades, tests fixes inside an isolated sandbox, and promotes verified candidate indices to active production only upon human approval.

---

## Project Structure

```
rag-doctor/
├── corpus/
│   ├── active/               # Current production documents (FastAPI docs subset)
│   └── candidate/            # Working copy used for candidate experiments
├── indexes/
│   ├── active.json           # Active index pointer (tracks current active & baseline)
│   ├── v001/                 # Baseline production index (chunk_size: 500, overlap: 100)
│   └── v002_sick/            # Deliberately degraded index (chunk_size: 70, overlap: 0)
├── eval/
│   ├── eval_set.json         # 20 held-out Q&A pairs covering the FastAPI corpus
│   ├── judge.py              # LLM-as-judge evaluator (Faithfulness & Answer Relevancy)
│   ├── baseline_results.json # Baseline evaluation report (v001: 4.58 / 5.0)
│   └── sick_results.json     # Degraded evaluation report (v002_sick: 3.83 / 5.0)
├── pipeline/
│   ├── __init__.py           # Unified exports and RAGPipeline wrapper
│   ├── chunker.py            # Parameterized chunking (chunk_size, overlap)
│   ├── embedder.py           # Sentence Transformers embedding module
│   ├── index.py              # FAISS wrapper with versioning & persistence
│   ├── retriever.py          # Top-k similarity retrieval
│   └── generator.py          # Groq LLM answer generation with retry backoff
├── agent/                    # TrueForge agent orchestrator & subagents
│   └── diagnose.py           # Diagnose subagent (MCP client + Groq reasoning)
├── mcp_server.py             # Local MCP server (stdio & SSE transport)
├── sandbox_scripts/          # Sandbox remediation scripts
├── scripts/
│   ├── build_index.py        # CLI script to build and version indices
│   ├── prototype_diagnosis.py# Prototype LLM diagnosis reasoning
│   ├── query.py              # Interactive / CLI query tool
│   └── smoke_test.py         # End-to-end pipeline smoke test
├── tests/
│   ├── test_pipeline.py      # Pipeline unit & integration tests
│   └── test_eval.py          # Eval harness & index isolation tests
├── requirements.txt
├── .env.example
├── AGENTS.md
└── ARCHITECTURE.md
```

---

## Setup & Quickstart

### 1. Environment Setup

Using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Configure your Groq API key in `.env`:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

### 3. Active Index State (`indexes/active.json`)

RAG Doctor uses a dynamic state pointer file (`indexes/active.json`) to track which index is currently monitored and which is the known-good baseline:

```json
{
  "active_version": "v002_sick",
  "baseline_version": "v001"
}
```

- **For the Demo (Degraded State)**: Set `"active_version": "v002_sick"` so the Doctor detects degradation and diagnoses the issue.
- **For Healthy Baseline**: Set `"active_version": "v001"`.

---

## TrueForge MCP Integration

RAG Doctor exposes a custom Model Context Protocol (MCP) server that TrueForge connects to over **SSE (Server-Sent Events)** or **stdio**.

### 1. Start the MCP Server

Start the server in SSE mode on port 8000:

```bash
.venv/bin/python mcp_server.py --transport sse --port 8000
```

### 2. Connect TrueForge to MCP

In the TrueForge Agent / MCP configuration:
- **Transport**: SSE / HTTP URL
- **URL**: `http://127.0.0.1:8000/sse` *(or `http://localhost:8000/sse`)*

*(If TrueForge is running in the cloud, expose port 8000 via a tunnel like `ngrok http 8000` and use `https://your-subdomain.ngrok-free.app/sse`).*

### 3. Exposed MCP Tools

| MCP Tool | Description |
|---|---|
| `inspect_rag_health()` | Returns health status (`DEGRADED`/`HEALTHY`), active vs baseline score, failing query counts, and available indices. |
| `get_pipeline_config(index_version)` | Fetches chunk size, overlap, and embedding settings (defaults to active index). |
| `get_failed_queries(index_version)` | Returns failing queries with expected vs generated answers and retrieved chunks (defaults to active index). |
| `get_evaluation_results(index_version)`| Returns summary metrics (faithfulness, relevancy, overall score). |
| `inspect_retrieval(query, index_version)` | Ad-hoc similarity retrieval for debugging chunk relevance. |

---

## Usage & Development Commands

### 1. Build an Index

```bash
# Build baseline index (v001)
python scripts/build_index.py --corpus-dir corpus/active --output-dir indexes/v001 --chunk-size 500 --overlap 100 --version v001

# Build a sick / degraded index variant for testing (v002_sick)
python scripts/build_index.py --corpus-dir corpus/active --output-dir indexes/v002_sick --chunk-size 70 --overlap 0 --version v002_sick
```

### 2. Run Evaluation (LLM-as-a-Judge)

The evaluation harness evaluates answers against 20 held-out Q&A pairs on two dimensions (1–5 integer scale):
- **Faithfulness (1–5)**: Checks if claims are grounded in retrieved context chunks without hallucination.
- **Answer Relevancy (1–5)**: Checks if the response directly and completely answers the question.

```bash
# Evaluate baseline index
python eval/judge.py --index indexes/v001 --output eval/baseline_results.json

# Evaluate sick index
python eval/judge.py --index indexes/v002_sick --output eval/sick_results.json
```

### 3. Compare Healthy vs Sick Index (Controlled: `openai/gpt-oss-120b`)

| Metric | `v001` (Healthy Baseline) | `v002_sick` (Degraded) | Delta |
|---|---|---|---|
| **Model** | `openai/gpt-oss-120b` | `openai/gpt-oss-120b` | *Controlled* |
| **Chunk Size / Overlap** | 500 chars / 100 chars | 70 chars / 0 chars | Pathology |
| **Total Chunks** | 294 | 1666 | +1372 |
| **Avg Faithfulness** | **4.85 / 5.0** | **4.55 / 5.0** | -0.30 |
| **Avg Answer Relevancy** | **4.30 / 5.0** | **3.10 / 5.0** | **-1.20** |
| **Avg Overall Score** | **4.58 / 5.0** | **3.83 / 5.0** | **-0.75 (-16.4%)** |
| **Passing / Failing Queries** | 20 passed / 0 failed | 18 passed / 2 failed | - |
| **Degraded Status** | `HEALTHY` | `DEGRADED` (>15% drop) | - |

### 4. Run the Diagnose Subagent Locally

The Diagnose subagent queries the MCP server for pipeline state and failing queries, then prompts an LLM to identify the pathology category, evidence, confidence, and recommended experiment:

```bash
# Run Diagnose subagent dynamically against the active index
python agent/diagnose.py
```

Example Diagnosis output:
```json
{
  "suspected_cause": "chunking problem",
  "evidence": "The degraded config uses a chunk size of 70 with no overlap, resulting in extremely fragmented excerpts that omit critical details...",
  "confidence": 0.94,
  "recommended_experiment": "increase chunk size to 400-500 and set overlap to 50-100 to capture whole statements and preserve context"
}
```

### 5. Interactive Query Tool

```bash
python scripts/query.py --index indexes/v001 --query "How do background tasks work in FastAPI?"
```

### 6. Run Test Suite

```bash
pytest
```


