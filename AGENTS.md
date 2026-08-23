# RAG Doctor — Coding Agent Instructions

## Source of Truth

The architecture in `ARCHITECTURE.md` is authoritative.

Do not redesign, expand, or replace the architecture unless explicitly
requested.

The goal is to implement exactly:

    health check
        ↓
    evaluation
        ↓
    detect degradation
        ↓
    diagnose
        ↓
    design remediation
        ↓
    execute in TrueForge sandbox
        ↓
    evaluate candidate
        ↓
    human approval
        ↓
    promote or discard

---

# Project Goal

RAG Doctor is a self-healing RAG pipeline agent for the TrueForge Hackathon.

The "Patient" is a small fixed RAG pipeline.

The "Doctor" is a TrueForge agent that:

1. runs a health check
2. detects quality degradation
3. investigates failing examples
4. forms a diagnosis
5. creates a remediation
6. executes the remediation inside the TrueForge sandbox
7. evaluates the candidate
8. presents before/after results
9. waits for human approval
10. promotes or discards the candidate

The active index must never be modified directly during experimentation.

---

# Architecture

## Patient

The patient consists of:

- fixed document corpus
- chunker
- Sentence Transformers embedder
- FAISS index
- retriever
- Groq-based generator

Keep the patient deliberately simple.

Do not add:
- PostgreSQL
- Redis
- external vector databases
- LangChain
- LlamaIndex
- additional retrieval frameworks

unless explicitly requested.

---

# Evaluation

The evaluation harness contains:

- held-out evaluation queries
- expected answers
- LLM-as-judge

The judge evaluates:

- faithfulness
- answer relevancy

Scores use a 1–5 scale and structured JSON.

Do not fabricate scores.

Do not change the evaluation set merely to improve results.

The evaluation set must remain separate from the normal corpus.

---

# Health Check

A health check is manually triggered.

Real-time monitoring is NOT part of the MVP.

The health check must:

1. evaluate the active index
2. calculate scores
3. compare the result against the configured threshold
4. return failing examples when degradation is detected

Suggested degradation rules:

- average score < 3.5 / 5
OR
- 15%+ drop from the baseline

Keep thresholds configurable.

---

# Diagnosis Agent

The Diagnose subagent receives:

- evaluation results
- failing queries
- retrieved chunks
- relevant pipeline information

It should determine a likely cause.

Possible diagnosis categories include:

- chunking problem
- retrieval problem
- embedding problem
- indexing problem
- metadata/context problem
- generation problem
- evaluation/data problem

The diagnosis must produce:

- suspected cause
- evidence
- confidence
- recommended experiment

Do not guess without evidence.

---

# Fix Agent

The Fix subagent receives the diagnosis.

Its job is to turn the diagnosis into a concrete remediation.

Examples:

- change chunk size
- change chunk overlap
- change chunking strategy
- re-embed
- rebuild the FAISS index
- adjust retrieval parameters

Do not modify the active index.

The remediation must operate against a copy of the corpus/index.

---

# Sandbox

All remediation code must execute inside the TrueForge sandbox.

The sandbox is responsible for running:

- re-chunking
- re-embedding
- index creation
- candidate evaluation scripts

Do not replace the TrueForge sandbox with another execution mechanism.

Do not expose unrestricted shell execution through the application.

---

# Candidate vs Active

There must always be a clear distinction between:

ACTIVE
and
CANDIDATE

Experiments operate on candidate state.

Candidate state must never overwrite active state.

After evaluation:

    candidate better
        ↓
    approval required
        ↓
    promote candidate

If rejected:

    discard candidate

The active index remains unchanged.

---

# Promotion

Promotion is the only operation allowed to change the active version.

Promotion requires explicit human approval.

Before asking for approval, present:

- diagnosis
- proposed remediation
- baseline score
- candidate score
- score delta
- any regressions
- affected evaluation examples

Never automatically promote.

---

# Rollback

Promotion must preserve the previous active version.

Rollback should be possible by switching the active version back to
the previous known-good version.

Do not delete previous versions after promotion.

---

# MCP

The custom RAG Doctor MCP exposes controlled application operations.

Keep the MCP surface small.

Preferred tools:

- inspect_rag_health
- get_evaluation_results
- get_failed_queries
- inspect_retrieval
- get_pipeline_config
- create_experiment
- compare_experiments
- promote_experiment
- rollback_experiment

Do not expose arbitrary code execution through MCP.

Do not bypass the sandbox.

Do not create generic MCP tools merely to increase MCP count.

---

# Model

Use Groq API for:

- answer generation
- LLM-as-judge
- agent reasoning

Keep model configuration in environment/configuration rather than
hardcoding credentials or model names throughout the codebase.

---

# Embeddings

Use Sentence Transformers locally.

Embeddings must not depend on an external embedding API.

---

# Vector Index

Use FAISS.

Index versions must be stored on the local filesystem.

Recommended structure:

    indexes/
        v001/
        v002/
        v003/

Each version should contain enough metadata to reproduce or identify
the configuration used.

---

# Storage

Use local filesystem storage.

Do not introduce a database for the MVP.

Keep candidate and active artifacts versioned and distinguishable.

---

# TrueForge

TrueForge is the agent harness.

Use it for:

- orchestration
- Diagnose subagent
- Fix subagent
- MCP tools
- sandbox execution
- approval gate

Do not introduce another agent framework.

Do not recreate TrueForge features inside Python.

---

# Python

Use Python 3.12 (the default python3 on the system may point to python3.14, always use python3.12 or the python3.12 venv) for:

- RAG pipeline
- evaluation
- agent integration
- experiment logic
- remediation scripts

Prefer simple functions and modules.

Avoid unnecessary abstraction.

---

# Testing

Write tests for:

- chunking
- embedding/index creation
- retrieval
- generation interface
- evaluation parsing
- degradation detection
- diagnosis output validation
- candidate isolation
- promotion rules
- rollback

Tests must verify that candidate changes cannot accidentally overwrite
the active index.

---

# Development Workflow

Before coding:

1. inspect relevant files
2. understand existing implementation
3. identify the smallest required change

Then:

1. implement
2. run tests
3. inspect failures
4. fix failures
5. inspect git diff
6. summarize the result

Do not modify unrelated files.

---

# Scope

Do NOT implement unless explicitly requested:

- authentication
- user accounts
- multi-tenancy
- realtime monitoring
- custom web frontend
- mobile app
- cloud deployment
- PostgreSQL
- Redis
- multiple vector databases
- multiple agent frameworks
- generic long-term memory
- production observability

The hackathon MVP is intentionally small.

---

# Hackathon Priority

Prioritize in this order:

1. Patient RAG pipeline
2. Eval harness
3. Intentional degradation/regression
4. TrueForge orchestration
5. Diagnose subagent
6. Fix subagent
7. Sandbox remediation
8. Candidate evaluation
9. Human approval
10. Promotion
11. Rollback
12. Demo polish

Do not sacrifice the agent loop to build UI features.

---

# Coding Agent Behavior

When given a task:

1. read ARCHITECTURE.md
2. inspect existing code
3. state a concise implementation plan
4. implement only the requested scope
5. run tests
6. report exactly what changed

Do not silently redesign architecture.

Do not add dependencies unless necessary.

Do not remove tests to make implementation pass.

Do not fabricate successful evaluation results.

If something is uncertain, make the smallest reasonable assumption and
keep it isolated so it can be changed later.
