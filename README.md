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
│   └── v001/                 # Initial production FAISS index & metadata
│       ├── index.faiss       # FAISS vector index binary
│       ├── chunks.json       # Document chunks with position metadata
│       └── config.json       # Config (chunk_size, overlap, embedding_model)
├── pipeline/
│   ├── __init__.py           # Unified exports and RAGPipeline wrapper
│   ├── chunker.py            # Parameterized chunking (chunk_size, overlap)
│   ├── embedder.py           # Sentence Transformers embedding module
│   ├── index.py              # FAISS wrapper with versioning & persistence
│   ├── retriever.py          # Top-k similarity retrieval
│   └── generator.py          # Groq LLM answer generation
├── eval/                     # Evaluation harness & held-out eval set
├── agent/                    # TrueForge agent orchestrator & subagents
├── sandbox_scripts/          # Sandbox remediation scripts
├── scripts/
│   ├── build_index.py        # CLI script to build and version indices
│   ├── query.py              # Interactive / CLI query tool
│   └── smoke_test.py         # End-to-end pipeline smoke test
├── tests/
│   └── test_pipeline.py      # Unit & integration tests
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

Set your `GROQ_API_KEY`:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

### 3. Build Initial Index (`v001`)

```bash
python scripts/build_index.py --corpus-dir corpus/active --output-dir indexes/v001 --chunk-size 500 --overlap 100
```

### 4. Run Smoke Test

```bash
python scripts/smoke_test.py
```

### 5. Run Tests

```bash
pytest
```
