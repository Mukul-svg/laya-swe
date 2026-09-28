# LAYA-SWE: Fast System-One Decision Models for Autonomous Coding Agent Memory

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/System--1-convaiinnovations%2Flaya-orange.svg)](https://huggingface.co/convaiinnovations/laya)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LAYA-SWE** is an autonomous software engineering memory architecture that adapts **LAYA** (`convaiinnovations/laya`, a 421M-parameter calibrated, non-autoregressive decision model based on ModernBERT-large) as an in-the-loop **System-1 memory controller** for coding agents.

During multi-step debugging and repository modification, coding agents accumulate thousands of lines of terminal dumps, tracebacks, and code reads. Naive context truncation often discards non-negotiable architectural invariants or breaks API tool-calling protocols. LAYA-SWE triages observations into distinct memory planes in a single forward pass, pinning architectural invariants and compacting intermediate execution logs while preserving tool-call schemas.

Read the documentation:
- **Technical Manuscript:** [`LAYA_SWE_RESEARCH_PAPER.md`](LAYA_SWE_RESEARCH_PAPER.md)
- **Architecture Specification:** [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## 1. Benchmark Results

Evaluated across 5 complex SWE benchmark tasks executed against `openai/gpt-oss-20b` on Groq hardware:

| Architecture Candidate | Invariant Retention (%) | Token Savings (%) | Pytest Success Rate (%) | Avg Turn Latency (s) |
|---|:---:|:---:|:---:|:---:|
| **Candidate A: Full Context (Control)** | 100.0% | 0.00% (Baseline) | 100.0% (18/18) | 1.53s |
| **Candidate B: Heuristic Structural Engine** | 100.0% | 10.02% | 100.0% (18/18) | 1.47s |
| **Candidate C: Pure LAYA Neural Controller** | 80.0% | 12.13% | 100.0% (18/18) | 7.24s |
| **Candidate D: LAYA-SWE Hybrid (Champion)** | **100.0%** | **12.72%** (Peak 17.5%) | **100.0% (18/18)** | 5.52s |

```text
Token Savings Comparison:
Candidate A (Control)      [0.0%]
Candidate B (Heuristic)    [========== 10.02%]
Candidate C (Pure LAYA)    [============ 12.13%]
Candidate D (LAYA-SWE)     [============= 12.72% - Peak: 17.5%]
```

---

## 2. Architecture

LAYA-SWE organizes agent observations across **Four Execution Planes**:

1. **Invariant Plane ($P_{\text{inv}}$):** Pinned architectural constraints, security rules, and negative invariants (immune from LRU eviction).
2. **Symbolic Plane ($P_{\text{sym}}$):** Structural AST outlines, symbol dependencies, and module signatures.
3. **Trajectory Plane ($P_{\text{traj}}$):** Shell execution logs, test outputs, tracebacks, and tool calls.
4. **Resolution Plane ($P_{\text{res}}$):** Causal bug resolutions, passing test patches, and verified fixes.

Coupled with **Schema-Safe Atomic Tail Slicing**, LAYA-SWE guarantees that assistant messages with `tool_calls` are never separated from their respective `tool` results.

---

## 3. Quickstart & Reproducibility

### Step 1: Setup Virtual Environment
```powershell
python -m venv .vent
.\.vent\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure Environment
Copy `.env.example` to `.env` and set your Groq API key:
```ini
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
PROXY_PORT=8080
LAYA_CONTROLLER_MODE=laya_hybrid
```

### Step 3: Run the Test Suite
```powershell
pytest -v
```

### Step 4: Run the Multi-Candidate Benchmark
```powershell
python run_autoresearch_benchmark.py
```

---

## 4. Repository Structure

```
├── proxy/
│   ├── server.py                   # Reverse proxy with session manager and compaction
│   ├── memory_engine.py            # 4-plane SWE memory controller & LAYA neural triage
│   └── compactor.py                # Schema-safe tail slicer & observation compactor
├── tasks/                          # 5 Multi-step SWE benchmark task repositories
│   ├── task_1_tenant_scoping/
│   ├── task_2_negative_constraint/
│   ├── task_3_iterative_debugging/
│   ├── task_4_distributed_2pc/
│   └── task_5_ast_optimizer/
├── tests/                          # Automated unit and integration tests (29 passing)
│   ├── test_memory_engine.py       # Deterministic verification of graph and invariants
│   └── test_proxy.py               # Route, compaction, and concurrency tests
├── scripts/                        # Analysis and evaluation tools
│   ├── analyze_telemetry.py        # Telemetry log parser and statistics
│   ├── test_live_production.py     # Live upstream connection verification
│   ├── verify_laya_classes.py      # LAYA neural classification tester
│   └── verify_scaling_law.py       # Compaction scaling simulator
├── extensions/                     # Agent extension hooks
│   └── laya-swe.ts                 # Native lifecycle hook for coding agents
├── ARCHITECTURE.md                 # In-depth technical architecture specification
├── LAYA_SWE_RESEARCH_PAPER.md       # Full academic manuscript / technical report
├── run_autoresearch_benchmark.py   # 4-candidate comparative benchmark runner
├── autoresearch_results.json       # Persisted empirical benchmark scorecard
├── benchmark_telemetry.jsonl       # Raw production telemetry logs
├── requirements.txt                # Python package dependencies
└── pytest.ini                      # Pytest configuration
```
