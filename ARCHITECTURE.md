# RAG Doctor — Architecture & Data Flow

Self-healing RAG pipeline agent, built for the TrueForge Hackathon. The agent monitors a RAG pipeline's answer quality via an LLM-as-judge, diagnoses the cause when quality drops, tests a fix inside a sandbox, and promotes the fix only after human approval.

## System Architecture

```mermaid
graph TB
    subgraph Patient[RAG Pipeline - the Patient]
        A1[Document Corpus]
        A2[Chunker]
        A3[Embedder - Sentence Transformers]
        A4[(FAISS Index)]
        A5[Retriever]
        A6[Generator - Groq LLM]
    end

    subgraph Eval[Eval Harness]
        B1[(Held-out Eval Set)]
        B2[LLM-as-Judge]
    end

    subgraph Doctor[TrueForge Agent - the Doctor]
        C1[Orchestrator]
        C2[Diagnose Subagent]
        C3[Fix Subagent]
        C4[[Sandbox]]
        C5{Approval Gate}
    end

    U((You))

    A1 --> A2 --> A3 --> A4
    A4 --> A5 --> A6
    B1 --> B2
    A6 --> B2
    B2 -->|score + failing examples| C1
    C1 -->|below threshold| C2
    C2 -->|hypothesis| C3
    C3 -->|remediation code| C4
    C4 -->|candidate index| A4
    C4 -->|triggers re-run| B2
    B2 -->|before + after scores| C5
    C5 <-->|approve or reject| U
    C5 -->|on approval| A4
```

**Components**

- **Patient (the RAG pipeline being healed):** a small, fixed toy corpus, chunked and embedded into a FAISS index, served through a standard retrieve-then-generate pipeline. Keep this simple — the interesting work happens in the Doctor layer, not here.
- **Eval Harness:** a held-out set of Q&A pairs the pipeline never sees in normal operation, scored by an LLM-as-judge on faithfulness and answer relevancy (1–5, structured JSON) — same pattern as your existing eval work.
- **Doctor (TrueForge agent):** the orchestrator runs the health check. On a bad score it hands off to Diagnose (forms a hypothesis), then Fix (executes a remediation in the sandbox and re-evaluates). Nothing reaches the active index without your approval at the gate.

## Data Flow (execution sequence)

```mermaid
sequenceDiagram
    participant You
    participant Orchestrator
    participant EvalHarness as Eval Harness
    participant Diagnose as Diagnose Subagent
    participant Fix as Fix Subagent
    participant Sandbox
    participant Index as Active Index

    You->>Orchestrator: trigger health check
    Orchestrator->>EvalHarness: run judge eval on active index
    EvalHarness->>Index: query held-out set
    Index-->>EvalHarness: retrieved chunks + answers
    EvalHarness-->>Orchestrator: score + failing examples

    alt score below threshold
        Orchestrator->>Diagnose: failing examples + retrieved chunks
        Diagnose-->>Orchestrator: hypothesis (suspected cause)
        Orchestrator->>Fix: hypothesis
        Fix->>Sandbox: execute remediation code
        Sandbox->>Sandbox: re-chunk / re-embed corpus copy
        Sandbox-->>Fix: candidate index
        Fix->>EvalHarness: run judge eval on candidate index
        EvalHarness-->>Fix: candidate score
        Fix-->>Orchestrator: diagnosis + fix + before/after scores
        Orchestrator->>You: present diagnosis + scores, pause
        You-->>Orchestrator: approve or reject
        alt approved
            Orchestrator->>Index: promote candidate to active
        else rejected
            Orchestrator->>Sandbox: discard candidate
        end
    else score healthy
        Orchestrator->>You: report all clear
    end
```

**In plain steps:**
1. Trigger a health check — a manual command is fine, don't build real-time monitoring.
2. Eval harness scores the active index against the held-out set.
3. Below threshold (suggest: avg score < 3.5/5, or a 15%+ drop from baseline) → Diagnose subagent inspects failing queries + retrieved chunks, forms a hypothesis.
4. Fix subagent turns the hypothesis into remediation code, runs it in the sandbox against a **copy** of the corpus — never touch the active index directly.
5. Re-run the eval harness against the candidate index → before/after scores.
6. Orchestrator pauses, shows you the diagnosis and both scores.
7. You approve → candidate becomes active. You reject → candidate is discarded, active index untouched.

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Agent harness | TrueForge | sponsor tool; subagents, sandbox, and approval gates are built in |
| Agent + judge + generation model | Groq API | fast inference, you already use it, generous free tier |
| Embeddings | Sentence Transformers (local) | no API rate limits, matches your existing RAG project |
| Vector index | FAISS (in-memory) | fastest to stand up, no DB server to manage this week |
| Code execution | TrueForge's built-in sandbox | no separate setup — this is where re-chunk/re-embed scripts actually run |
| Corpus/index storage | local filesystem, versioned folders | simplest possible versioning for a week-long build |
| Language | Python | matches your stack, runs cleanly in the sandbox |
| Frontend | TrueForge's built-in chat UI | skip building a custom UI, spend the saved time on agent logic |
| Version control | GitHub (public repo) | required for submission |

## Suggested Repo Structure

```
rag-doctor/
├── corpus/
│   ├── active/          # current production documents
│   └── candidate/       # working copy used to test fixes
├── pipeline/
│   ├── chunker.py
│   ├── embedder.py
│   ├── index.py         # FAISS wrapper, versioned (active/candidate)
│   ├── retriever.py
│   └── generator.py
├── eval/
│   ├── eval_set.json    # held-out Q&A pairs
│   └── judge.py         # LLM-as-judge scorer
├── agent/
│   ├── orchestrator.py  # TrueForge entrypoint, ties the loop together
│   ├── diagnose.py      # diagnose subagent
│   └── fix.py           # fix subagent
├── sandbox_scripts/
│   └── remediate.py     # the actual re-chunk/re-embed code the sandbox runs
└── README.md
```
