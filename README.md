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
│   ├── v001/                 # Baseline production index (chunk_size: 500, overlap: 100)
│   └── v002_sick/            # Deliberately degraded index (chunk_size: 70, overlap: 0)
├── eval/
│   ├── eval_set.json         # 20 held-out Q&A pairs covering the FastAPI corpus
│   ├── judge.py              # LLM-as-judge evaluator (Faithfulness & Answer Relevancy)
│   ├── baseline_results.json # Baseline evaluation report (v001: 4.58 / 5.0)
│   └── sick_results.json     # Degraded evaluation report (v002_sick: 3.58 / 5.0)
├── pipeline/
│   ├── __init__.py           # Unified exports and RAGPipeline wrapper
│   ├── chunker.py            # Parameterized chunking (chunk_size, overlap)
│   ├── embedder.py           # Sentence Transformers embedding module
│   ├── index.py              # FAISS wrapper with versioning & persistence
│   ├── retriever.py          # Top-k similarity retrieval
│   └── generator.py          # Groq LLM answer generation with retry backoff
├── agent/                    # TrueForge agent orchestrator & subagents
├── sandbox_scripts/          # Sandbox remediation scripts
├── scripts/
│   ├── build_index.py        # CLI script to build and version indices
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

### 1. Environment

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

Configure your `GROQ_API_KEY`:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

---

## Usage

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

### 4. Interactive Query Tool

```bash
python scripts/query.py --index indexes/v001 --query "How do background tasks work in FastAPI?"
```

### 5. Run Test Suite

```bash
pytest
```
