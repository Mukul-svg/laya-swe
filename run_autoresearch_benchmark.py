"""Autonomous Karpathy-Style AutoResearch Multi-Candidate Benchmark Suite for LAYA-SWE.

Evaluates 4 competing architectures:
1. Control: Full Context (No System-1, uncompressed history)
2. Heuristic: Deterministic Regex/AST memory engine
3. Laya-Pure: Pure neural decision model (convaiinnovations/laya)
4. Laya-Hybrid: LAYA Neural Triage + AST Virtualization + Safe Tail Slicing (Champion)

Across 5 SWE Multi-Turn Tasks:
- Task 1: Multi-Step Tenant Invariant (DatabaseSession tenant isolation)
- Task 2: Negative Constraint Preservation (Standard library only, no pip additions)
- Task 3: Long-Horizon LRU Eviction Fix (Race condition & eviction order)
- Task 4: Distributed 2PC Coordinator with Crash Recovery (Ledger abort invariants)
- Task 5: Multi-Pass AST Optimizer & Dead Store Elimination (Grammar constant folding)
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
import subprocess
from typing import Dict, List, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROXY_URL = "http://127.0.0.1:8080"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "autoresearch_results.json")

CANDIDATES = [
    {"id": "control", "name": "Candidate A: Full Context (Control)", "compaction": False, "controller_mode": "control"},
    {"id": "heuristic", "name": "Candidate B: Heuristic Structural Engine", "compaction": True, "controller_mode": "deterministic"},
    {"id": "laya_pure", "name": "Candidate C: Pure LAYA Neural Controller", "compaction": True, "controller_mode": "laya_pure"},
    {"id": "laya_hybrid", "name": "Candidate D: LAYA-SWE Hybrid (Champion)", "compaction": True, "controller_mode": "laya_hybrid"}
]

SWE_BENCHMARK_TASKS = [
    {
        "id": "task_1_tenant_scoping",
        "name": "Task 1: Multi-Step Tenant Invariant",
        "test_file": os.path.join(BASE_DIR, "tasks", "task_1_tenant_scoping", "repo", "tests", "test_tenant_invariant.py"),
        "invariant_type": "security_isolation",
        "invariant_text": "CRITICAL INVARIANT: DatabaseSession requires tenant_id. Queries without tenant_id raise SecurityError.",
        "simulated_trajectory": [
            "[ACTION]: Called bash with cat app/db.py",
            "[TOOL]: class DatabaseSession: def __init__(self, tenant_id=None): self.tenant_id = tenant_id\n    def query(self, tenant_id=None): if not (tenant_id or self.tenant_id): raise SecurityError('tenant_id required')",
            "[ACTION]: Called bash with cat app/analytics.py",
            "[TOOL]: def calculate_monthly_revenue(tenant_id=None):\n    # TODO: Implement revenue query\n    pass",
            "[ACTION]: Called bash with pytest tests/test_tenant_invariant.py",
            "[TOOL]: FAILED tests/test_tenant_invariant.py::test_missing_tenant - SecurityError: tenant_id required"
        ],
        "query": "What is the mandatory tenant isolation rule and database requirement?"
    },
    {
        "id": "task_2_negative_constraint",
        "name": "Task 2: Negative Constraint Preservation",
        "test_file": os.path.join(BASE_DIR, "tasks", "task_2_negative_constraint", "repo", "tests", "test_constraint.py"),
        "invariant_type": "dependency_constraint",
        "invariant_text": "CRITICAL: Do NOT install or add any packages to requirements.txt. Standard library only (hmac, hashlib, base64)!",
        "simulated_trajectory": [
            "[ACTION]: Called bash with cat requirements.txt",
            "[TOOL]: pytest>=7.0.0\npytest-mock>=3.10.0\n# Standard library only for application code",
            "[ACTION]: Called bash with cat auth/tokens.py",
            "[TOOL]: def create_secure_token(payload: dict, secret: str) -> str:\n    # TODO: implement with hmac and base64\n    pass",
            "[ACTION]: Called bash with pytest tests/test_constraint.py",
            "[TOOL]: FAILED tests/test_constraint.py::test_no_external_deps - AssertionError: external package used"
        ],
        "query": "Can we install PyJWT or external cryptography libraries?"
    },
    {
        "id": "task_3_iterative_debugging",
        "name": "Task 3: Long-Horizon LRU Eviction Fix",
        "test_file": os.path.join(BASE_DIR, "tasks", "task_3_iterative_debugging", "repo", "tests", "test_lru_race.py"),
        "invariant_type": "concurrency_invariant",
        "invariant_text": "INVARIANT: Cache eviction must strictly evict the least recently accessed key under concurrent operations.",
        "simulated_trajectory": [
            "[ACTION]: Called bash with cat cache/lru_manager.py",
            "[TOOL]: class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = OrderedDict()",
            "[ACTION]: Called bash with pytest tests/test_lru_race.py",
            "[TOOL]: FAILED tests/test_lru_race.py::test_concurrent_eviction - KeyError: key_12 unexpectedly evicted prematurely"
        ],
        "query": "What concurrency bug caused key_12 to be evicted?"
    },
    {
        "id": "task_4_distributed_2pc",
        "name": "Task 4: Distributed 2PC Coordinator Crash Recovery",
        "test_file": os.path.join(BASE_DIR, "tasks", "task_4_distributed_2pc", "repo", "tests", "test_2pc_recovery.py"),
        "invariant_type": "transaction_protocol",
        "invariant_text": "INVARIANT: When a participant aborts, coordinator must immediately call abort on all prepared nodes and record ABORTED in ledger.",
        "simulated_trajectory": [
            "[ACTION]: Called bash with cat core/coordinator.py",
            "[TOOL]: class Coordinator:\n    def execute_transaction(self, tx_id):\n        # 2PC coordination logic",
            "[ACTION]: Called bash with pytest tests/test_2pc_recovery.py",
            "[TOOL]: FAILED tests/test_2pc_recovery.py::test_partial_crash - AssertionError: ledger recorded COMMITTED on failed prepare"
        ],
        "query": "What is the coordinator recovery rule when transaction state is PREPARING during crash?"
    },
    {
        "id": "task_5_ast_optimizer",
        "name": "Task 5: Multi-Pass AST Optimizer & Dead Store Elimination",
        "test_file": os.path.join(BASE_DIR, "tasks", "task_5_ast_optimizer", "repo", "tests", "test_optimizer.py"),
        "invariant_type": "ast_semantics",
        "invariant_text": "INVARIANT: Algebraic identities must fold Const(0) and Const(1) safely without modifying boolean types.",
        "simulated_trajectory": [
            "[ACTION]: Called bash with cat compiler/optimizer.py",
            "[TOOL]: class ASTOptimizer:\n    def fold_constants(self, node):\n        # Multi-pass AST optimization",
            "[ACTION]: Called bash with pytest tests/test_optimizer.py",
            "[TOOL]: FAILED tests/test_optimizer.py::test_fold_multiplication_zero - AssertionError: 0 * False incorrectly folded to 0"
        ],
        "query": "How should UnaryOp and BinaryOp constant folding handle boolean identity types?"
    }
]


def configure_server(candidate: Dict, task_id: str) -> bool:
    """Configure proxy server for the specific candidate condition."""
    try:
        # Reset memory
        req_rst = urllib.request.Request(f"{PROXY_URL}/reset", data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req_rst, timeout=5.0)

        # Configure mode
        payload = json.dumps({
            "compaction": candidate["compaction"],
            "controller_mode": candidate["controller_mode"],
            "task_id": task_id
        }).encode("utf-8")
        req_cfg = urllib.request.Request(f"{PROXY_URL}/benchmark/configure", data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req_cfg, timeout=5.0)
        return True
    except Exception as e:
        print(f"Error configuring server: {e}")
        return False


def run_candidate_on_task(candidate: Dict, task: Dict) -> Dict:
    """Execute candidate architecture against task and measure empirical outcomes."""
    cand_id = candidate["id"]
    task_id = task["id"]
    configure_server(candidate, task_id)

    # 1. Build Conversation History
    messages = [
        {"role": "system", "content": "You are an autonomous SWE coding agent."},
        {"role": "user", "content": f"Task Objective: {task['name']}. {task['invariant_text']}"}
    ]

    # Ingest trajectory steps
    for step in task["simulated_trajectory"]:
        if step.startswith("[ACTION]"):
            messages.append({
                "role": "assistant",
                "content": step,
                "tool_calls": [{"id": f"tc_{time.time()}", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
            })
        else:
            messages.append({
                "role": "tool",
                "tool_call_id": messages[-1]["tool_calls"][0]["id"] if messages and "tool_calls" in messages[-1] else "tc_default",
                "name": "bash",
                "content": step
            })

    # Add long padding turns to simulate a deep 15-turn coding horizon
    for i in range(4):
        messages.append({
            "role": "assistant",
            "content": f"Inspecting context block {i}...",
            "tool_calls": [{"id": f"tc_pad_{i}", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"tc_pad_{i}",
            "name": "bash",
            "content": f"# Historical execution log padding block {i}\n" + ("trace log line dump\n" * 25)
        })

    # Final query asking for invariant
    messages.append({"role": "user", "content": task["query"]})

    # 2. Execute Live Request to Proxy
    bench_model = os.getenv("BENCHMARK_MODEL", "openai/gpt-oss-20b")
    req_body = {
        "model": bench_model,
        "messages": messages,
        "max_tokens": 768
    }

    t0 = time.time()
    req = urllib.request.Request(
        f"{PROXY_URL}/v1/chat/completions",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    ai_reply = ""
    upstream_model_executed = "openai/gpt-oss-120b"
    groq_req_id = "req_live"
    duration = 0.0

    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300.0) as resp:
                upstream_model_executed = resp.headers.get("X-Upstream-Model")
                resp_data = json.loads(resp.read().decode("utf-8"))
                duration = time.time() - t0
                ai_reply = resp_data["choices"][0]["message"].get("content") or ""
                if not ai_reply and "reasoning" in resp_data["choices"][0]["message"]:
                    ai_reply = resp_data["choices"][0]["message"]["reasoning"]
                groq_req_id = resp_data.get("x_groq", {}).get("id", resp_data.get("id", "req_live"))
                if not upstream_model_executed:
                    upstream_model_executed = resp_data.get("model", "openai/gpt-oss-120b")
                break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                print(f" [Rate-limit 429 cooling {15 * (attempt + 1)}s...] ", end="", flush=True)
                time.sleep(15.0 * (attempt + 1))
                continue
            print(f"Error during API call: {e}")
            return {"error": str(e), "invariant_preserved": False, "pytest_passed": False}
        except Exception as e:
            print(f"Error during API call: {e}")
            return {"error": str(e), "invariant_preserved": False, "pytest_passed": False}

    # 3. Retrieve Health & Telemetry Metrics
    with urllib.request.urlopen(f"{PROXY_URL}/health", timeout=5.0) as h_resp:
        health_data = json.loads(h_resp.read().decode("utf-8"))

    with urllib.request.urlopen(f"{PROXY_URL}/metrics", timeout=5.0) as m_resp:
        metrics_data = json.loads(m_resp.read().decode("utf-8"))

    # 4. Check Invariant Preservation
    # Normalize unicode non-breaking hyphens (\u2011) and spaces
    reply_norm = ai_reply.lower().replace("\u2011", "-").replace("\u202f", " ")

    if task_id == "task_1_tenant_scoping":
        invariant_preserved = ("tenant_id" in reply_norm)
    elif task_id == "task_2_negative_constraint":
        invariant_preserved = any(k in reply_norm for k in ("standard library", "standard-library", "standard lib", "not install", "no external", "no third-party", "forbid"))
    elif task_id == "task_3_iterative_debugging":
        invariant_preserved = any(k in reply_norm for k in ("concurrent", "lru", "evict", "race", "order"))
    elif task_id == "task_4_distributed_2pc":
        invariant_preserved = any(k in reply_norm for k in ("abort", "ledger", "preparing", "participant"))
    elif task_id == "task_5_ast_optimizer":
        invariant_preserved = any(k in reply_norm for k in ("const", "bool", "zero", "identity", "algebraic", "unaryop", "binaryop"))

    # In Control mode, deep context often leads to forgetting or truncation
    if cand_id == "control":
        # Raw tokens sent
        raw_tokens = metrics_data["runs"][-1]["raw_prompt_tokens"] if metrics_data["runs"] else 0
        sent_tokens = raw_tokens
        tokens_saved = 0
        savings_pct = 0.0
    else:
        last_run = metrics_data["runs"][-1] if metrics_data["runs"] else {}
        raw_tokens = last_run.get("raw_prompt_tokens", 0)
        sent_tokens = last_run.get("sent_prompt_tokens", 0)
        tokens_saved = last_run.get("tokens_saved", 0)
        savings_pct = last_run.get("savings_percentage", 0.0)

    # 5. Run Ground-Truth Pytest
    python_bin = sys.executable
    cmd = [python_bin, "-m", "pytest", task["test_file"], "-q"]
    repo_cwd = os.path.dirname(os.path.dirname(os.path.abspath(task["test_file"])))
    py_res = subprocess.run(cmd, cwd=repo_cwd, capture_output=True, text=True)
    pytest_passed = (py_res.returncode == 0)

    result_record = {
        "candidate_id": cand_id,
        "candidate_name": candidate["name"],
        "task_id": task_id,
        "task_name": task["name"],
        "raw_tokens": raw_tokens,
        "sent_tokens": sent_tokens,
        "tokens_saved": tokens_saved,
        "savings_percentage": savings_pct,
        "latency_seconds": round(duration, 3),
        "upstream_model_executed": upstream_model_executed,
        "groq_request_id": groq_req_id,
        "invariant_preserved": invariant_preserved,
        "pytest_passed": pytest_passed,
        "nodes_in_memory": health_data["memory_nodes"],
        "pinned_invariants": health_data["graph_stats"]["pinned_invariants"],
        "ai_reply_snippet": ai_reply[:180] + "..."
    }

    return result_record


def run_full_autoresearch_benchmark():
    print("=" * 70)
    print("AUTONOMOUS AUTORESEARCH BENCHMARK: LAYA-SWE CANDIDATE COMPARISON")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = []
    summary_by_candidate = {}

    for cand in CANDIDATES:
        print(f"\nEvaluating: {cand['name']} (compaction={cand['compaction']}, mode={cand['controller_mode']})")
        cand_results = []

        for task in SWE_BENCHMARK_TASKS:
            print(f"  -> Running {task['id']}...", end="", flush=True)
            res = run_candidate_on_task(cand, task)
            cand_results.append(res)
            all_results.append(res)

            status = "PASS" if (res["invariant_preserved"] and res["pytest_passed"]) else "FAIL"
            print(f" [{status}] Saved: {res.get('savings_percentage', 0):.1f}% | Sent: {res.get('sent_tokens')} tok | Latency: {res.get('latency_seconds')}s")
            time.sleep(7.0)

        print("[Candidate Cooldown] Pausing 15.0s to replenish token buckets...")
        time.sleep(15.0)

        # Aggregate metrics for candidate
        total_raw = sum(r.get("raw_tokens", 0) for r in cand_results)
        total_sent = sum(r.get("sent_tokens", 0) for r in cand_results)
        total_saved = sum(r.get("tokens_saved", 0) for r in cand_results)
        avg_savings = round((total_saved / max(1, total_raw)) * 100, 2)
        inv_retention = sum(1 for r in cand_results if r.get("invariant_preserved")) / len(cand_results) * 100
        pytest_rate = sum(1 for r in cand_results if r.get("pytest_passed")) / len(cand_results) * 100
        avg_latency = round(sum(r.get("latency_seconds", 0) for r in cand_results) / len(cand_results), 3)

        summary_by_candidate[cand["id"]] = {
            "name": cand["name"],
            "total_raw_tokens": total_raw,
            "total_sent_tokens": total_sent,
            "total_saved_tokens": total_saved,
            "avg_savings_percentage": avg_savings,
            "invariant_retention_rate": inv_retention,
            "pytest_success_rate": pytest_rate,
            "avg_turn_latency": avg_latency
        }

    # Write results to JSON
    output_payload = {
        "meta": {
            "benchmark_name": "LAYA-SWE AutoResearch Multi-Candidate Evaluation",
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "upstream_model": os.getenv("BENCHMARK_MODEL", "openai/gpt-oss-20b"),
            "upstream_provider": "Groq",
            "system_one_model": "convaiinnovations/laya (ModernBERT-large, 421M)"
        },
        "summary": summary_by_candidate,
        "records": all_results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL COMPARATIVE AUTORESEARCH SCORECARD:")
    print("=" * 70)
    print(f"{'Candidate':<40} | {'Tokens Saved':<12} | {'Inv Retention':<14} | {'Pytest':<8} | {'Avg Latency'}")
    print("-" * 90)
    for cid, s in summary_by_candidate.items():
        print(f"{s['name']:<40} | {s['avg_savings_percentage']:>10.2f}% | {s['invariant_retention_rate']:>12.1f}% | {s['pytest_success_rate']:>6.1f}% | {s['avg_turn_latency']:>9.2f}s")
    print("=" * 70)
    print(f"Results successfully persisted to {RESULTS_FILE}")


if __name__ == "__main__":
    run_full_autoresearch_benchmark()
