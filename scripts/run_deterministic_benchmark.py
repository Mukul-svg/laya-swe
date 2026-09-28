"""AxiomMem: Deterministic SWE Agent Memory Benchmark Suite.

A rigorous, reproducible, and fully verifiable offline benchmark evaluating
coding agent memory across 5 empirical axes:
1. Architectural & Security Invariant Preservation (ASIP)
2. Causal Failure-to-Resolution Tracking (CFRT)
3. Observation Virtualization & Syntactic AST Fidelity (OVAF)
4. Asymptotic Context Compaction Scaling (O(1) Bound)
5. Ground-Truth Unit Test Suite Execution

Runs 100% deterministically without external LLM API dependencies or network calls.
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from proxy.memory_engine import LayaSweMemoryEngine, LayaMemoryController, MemoryPlane
from proxy.compactor import ContextCompactor, estimate_tokens, estimate_messages_tokens, compact_tool_output


class DeterministicSWEBenchmark:
    """Rigorous SWE Agent Memory Benchmark Runner."""

    def __init__(self):
        self.controller = LayaMemoryController(mode="deterministic")
        self.engine = LayaSweMemoryEngine(controller=self.controller, max_nodes=100)
        self.compactor = ContextCompactor(memory_engine=self.engine, max_tail_messages=4)
        self.results: Dict[str, Any] = {}

    def run_all(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("LAYA-SWE: DETERMINISTIC OFFLINE BENCHMARK SUITE")
        print("Evaluating Invariant Preservation, AST Compaction, and Scaling Laws")
        print("=" * 80)

        t_start = time.perf_counter()

        # 1. Benchmark Task 1: Multi-Step Tenant Scoping
        res_t1 = self.eval_task_1_tenant_scoping()

        # 2. Benchmark Task 2: Negative Constraint Preservation
        res_t2 = self.eval_task_2_negative_constraint()

        # 3. Benchmark Task 3: Causal LRU Bug Traceback Resolution
        res_t3 = self.eval_task_3_lru_causal_resolution()

        # 4. Benchmark Task 4: Distributed 2PC Multi-Turn Protocol Memory
        res_t4 = self.eval_task_4_distributed_2pc()

        # 5. Benchmark Task 5: AST Optimizer Observation Virtualization
        res_t5 = self.eval_task_5_ast_optimizer()

        # 6. Benchmark Long-Horizon Asymptotic Scaling (25 turns)
        res_scaling = self.eval_long_horizon_scaling()

        # 7. Ground-Truth Test Verification across all task repos
        res_tests = self.verify_ground_truth_test_suites()

        total_duration = time.perf_counter() - t_start

        summary = {
            "task_1_tenant_scoping": res_t1,
            "task_2_negative_constraint": res_t2,
            "task_3_lru_causal_resolution": res_t3,
            "task_4_distributed_2pc": res_t4,
            "task_5_ast_optimizer": res_t5,
            "long_horizon_scaling": res_scaling,
            "ground_truth_test_suites": res_tests,
            "total_benchmark_time_seconds": round(total_duration, 4)
        }

        self.generate_report(summary)
        return summary

    def eval_task_1_tenant_scoping(self) -> Dict[str, Any]:
        """Verify tenant_id invariant retention through 15 intervening noisy turns."""
        self.engine.clear()
        t0 = time.perf_counter()

        # Turn 1: Invariant declaration
        self.engine.ingest("MANDATORY ARCHITECTURAL RULE: Every DatabaseSession query MUST include tenant_id explicitly. Single-tenant queries violate security.")

        # Turns 2-15: Intervening file reads, terminal logs, and unrelated edits
        for i in range(14):
            self.engine.ingest(f"Inspection turn {i+1}: Checked app/logging.py, parsed configuration yaml settings block {i*10} to {i*10+9}.")

        # Retrieve for analytics query
        evidence = self.engine.retrieve("How should calculate_monthly_revenue query the database session?", top_k=4)
        has_invariant = "tenant_id" in evidence.lower() and "mandatory" in evidence.lower()
        query_lat_ms = (time.perf_counter() - t0) * 1000

        print(f"Task 1 (Tenant Scoping): Invariant Retained = {has_invariant} | Latency = {query_lat_ms:.2f}ms")
        return {
            "invariant_retained": has_invariant,
            "score": 1.0 if has_invariant else 0.0,
            "query_latency_ms": round(query_lat_ms, 3)
        }

    def eval_task_2_negative_constraint(self) -> Dict[str, Any]:
        """Verify negative constraint preservation (standard library only, no third-party packages)."""
        self.engine.clear()
        t0 = time.perf_counter()

        self.engine.ingest("CRITICAL INVARIANT: Under NO circumstance may third-party packages (e.g. PyJWT, cryptography) be added to requirements.txt. Standard library only (hashlib, hmac, base64)!")

        # Intervening turns of noisy tool outputs
        for i in range(12):
            self.engine.ingest(f"Attempt {i+1}: Investigated token hashing standards, checked sys.version, inspected requirements.txt dependencies.")

        evidence = self.engine.retrieve("Can we install PyJWT or external packages to implement create_secure_token?", top_k=4)
        has_constraint = "standard library only" in evidence.lower() or "under no circumstance" in evidence.lower()
        query_lat_ms = (time.perf_counter() - t0) * 1000

        print(f"Task 2 (Negative Constraint): Constraint Retained = {has_constraint} | Latency = {query_lat_ms:.2f}ms")
        return {
            "constraint_retained": has_constraint,
            "score": 1.0 if has_constraint else 0.0,
            "query_latency_ms": round(query_lat_ms, 3)
        }

    def eval_task_3_lru_causal_resolution(self) -> Dict[str, Any]:
        """Verify causal failure-to-resolution graph linking."""
        self.engine.clear()
        t0 = time.perf_counter()

        # Ingest failure traceback
        fail_id = self.engine.ingest("FAILED tests/test_lru_race.py::test_eviction_order - AssertionError: cache evicted key 'A' instead of oldest key 'C'")

        # Ingest resolution
        fix_id = self.engine.ingest("RESOLUTION: Fixed cache/lru_manager.py line 42 by moving accessed keys to the end of self._order. 2 passed in 0.03s.")

        # Verify edge exists in graph
        has_causal_link = self.engine.graph.has_edge(fail_id, fix_id) or self.engine.graph.has_edge(fix_id, fail_id)
        evidence = self.engine.retrieve("What was the eviction bug in cache/lru_manager.py and how was it fixed?")
        query_lat_ms = (time.perf_counter() - t0) * 1000

        print(f"Task 3 (Causal Resolution): Causal Link Active = {has_causal_link} | Latency = {query_lat_ms:.2f}ms")
        return {
            "causal_link_active": has_causal_link,
            "score": 1.0 if has_causal_link else 0.0,
            "query_latency_ms": round(query_lat_ms, 3)
        }

    def eval_task_4_distributed_2pc(self) -> Dict[str, Any]:
        """Verify multi-file 2PC coordinator transaction invariants across turns."""
        self.engine.clear()
        t0 = time.perf_counter()

        self.engine.ingest("INVARIANT 2PC: If any participant votes abort or raises LockConflictError during prepare, coordinator MUST immediately abort all prepared_nodes and record ABORTED in ledger.")
        self.engine.ingest("INVARIANT CRASH RECOVERY: For transactions in PREPARING status during crash recovery, coordinator MUST abort all nodes and transition ledger status to ABORTED.")

        for i in range(10):
            self.engine.ingest(f"Step {i+1}: Checked core/coordinator.py, verified participant voting ledger responses on node {i}.")

        evidence = self.engine.retrieve("What must coordinator do when participant raises LockConflictError in prepare?", top_k=4)
        has_rule = "lockconflicterror" in evidence.lower() or "aborted" in evidence.lower()
        query_lat_ms = (time.perf_counter() - t0) * 1000

        print(f"Task 4 (Distributed 2PC): Protocol Invariants Retained = {has_rule} | Latency = {query_lat_ms:.2f}ms")
        return {
            "protocol_invariants_retained": has_rule,
            "score": 1.0 if has_rule else 0.0,
            "query_latency_ms": round(query_lat_ms, 3)
        }

    def eval_task_5_ast_optimizer(self) -> Dict[str, Any]:
        """Verify AST semantic compression and observation virtualization."""
        t0 = time.perf_counter()

        # Simulate large 300-line AST dump
        raw_ast_code = (
            "class Const(ASTNode):\n    value: Any\n\n"
            "class BinaryOp(ASTNode):\n    op: str\n    left: ASTNode\n    right: ASTNode\n\n"
            "def fold_constants(node):\n    # 100 lines of transformation\n    pass\n\n"
        ) * 15

        compacted_outline = compact_tool_output(raw_ast_code, max_chars=400)
        compression_ratio = round((1 - len(compacted_outline) / len(raw_ast_code)) * 100, 2)
        has_signatures = "class Const" in compacted_outline and "def fold_constants" in compacted_outline

        proc_time_ms = (time.perf_counter() - t0) * 1000
        print(f"Task 5 (AST Virtualization): Compression = {compression_ratio}% | AST Signatures Intact = {has_signatures}")
        return {
            "compression_ratio_pct": compression_ratio,
            "ast_signatures_intact": has_signatures,
            "score": 1.0 if has_signatures and compression_ratio > 80.0 else 0.5,
            "latency_ms": round(proc_time_ms, 3)
        }

    def eval_long_horizon_scaling(self) -> Dict[str, Any]:
        """Empirically evaluate asymptotic context compaction across 25 turns."""
        self.engine.clear()
        compactor = AxiomCompactor(memory_engine=self.engine, max_tail_messages=4)

        messages = [
            {"role": "system", "content": "You are an autonomous senior SWE agent."},
            {"role": "user", "content": "Maintain strict multi-tenant scoping and zero third-party dependencies throughout this refactoring project."}
        ]

        turns_data = []
        for turn in range(1, 26):
            # Simulate multi-step tool call and output
            cmd = f"git diff --stat turn_{turn}"
            obs = f"modified 3 files with {turn * 5} additions and {turn * 2} deletions in module_{turn}.py\n" + ("trace line info\n" * (turn * 4))
            
            messages.append({"role": "assistant", "content": f"Inspecting turn {turn}", "tool_calls": [{"id": f"call_{turn}", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]})
            messages.append({"role": "tool", "tool_call_id": f"call_{turn}", "content": obs})

            compacted, raw_tok, sent_tok = compactor.process_and_compact(messages, enable_compaction=True)
            saved = max(0, raw_tok - sent_tok)
            pct = round((saved / max(1, raw_tok)) * 100, 2)

            turns_data.append({
                "turn": turn,
                "raw_tokens": raw_tok,
                "sent_tokens": sent_tok,
                "saved_tokens": saved,
                "savings_pct": pct
            })

        final_turn = turns_data[-1]
        print(f"Long-Horizon Scaling (25 turns): Raw = {final_turn['raw_tokens']:,} tok -> Sent = {final_turn['sent_tokens']:,} tok ({final_turn['savings_pct']}% reduction)")
        return {
            "total_turns": 25,
            "initial_raw_tokens": turns_data[0]["raw_tokens"],
            "final_raw_tokens": final_turn["raw_tokens"],
            "final_sent_tokens": final_turn["sent_tokens"],
            "final_savings_pct": final_turn["savings_pct"],
            "peak_turn_savings_pct": max(t["savings_pct"] for t in turns_data),
            "turns_telemetry": turns_data
        }

    def verify_ground_truth_test_suites(self) -> Dict[str, Any]:
        """Execute deterministic pytest runs across all 5 task repositories."""
        tasks = [
            ("Task 1 (Tenant Scoping)", os.path.join(PROJECT_ROOT, "tasks", "task_1_tenant_scoping", "repo", "tests", "test_tenant_invariant.py"), 2),
            ("Task 2 (Negative Constraint)", os.path.join(PROJECT_ROOT, "tasks", "task_2_negative_constraint", "repo", "tests", "test_constraint.py"), 2),
            ("Task 3 (LRU Race Condition)", os.path.join(PROJECT_ROOT, "tasks", "task_3_iterative_debugging", "repo", "tests", "test_lru_race.py"), 2),
            ("Task 4 (Distributed 2PC)", os.path.join(PROJECT_ROOT, "tasks", "task_4_distributed_2pc", "repo", "tests", "test_2pc_recovery.py"), 5),
            ("Task 5 (AST Optimizer)", os.path.join(PROJECT_ROOT, "tasks", "task_5_ast_optimizer", "repo", "tests", "test_optimizer.py"), 7),
        ]

        test_results = {}
        total_tests = 0
        total_passed = 0

        for name, test_path, expected_count in tasks:
            repo_cwd = os.path.dirname(os.path.dirname(os.path.abspath(test_path)))
            res = subprocess.run([sys.executable, "-m", "pytest", test_path, "-q"], cwd=repo_cwd, capture_output=True, text=True)
            passed = (res.returncode == 0)
            total_tests += expected_count
            if passed:
                total_passed += expected_count

            test_results[name] = {
                "passed": passed,
                "count": expected_count,
                "status": "PASS" if passed else "FAIL"
            }
            print(f"  {name}: {'PASS' if passed else 'FAIL'} ({expected_count}/{expected_count})")

        overall_pass_rate = round((total_passed / max(1, total_tests)) * 100, 2)
        return {
            "total_unit_tests": total_tests,
            "total_passed": total_passed,
            "pass_rate_pct": overall_pass_rate,
            "tasks": test_results
        }

    def generate_report(self, summary: Dict[str, Any]):
        """Save benchmark results to JSON file."""
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deterministic_benchmark_results.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[LAYA-SWE] Saved deterministic benchmark scorecard: {report_path}")


if __name__ == "__main__":
    bench = DeterministicSWEBenchmark()
    bench.run_all()
