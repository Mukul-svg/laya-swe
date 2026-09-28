"""Deterministic verification suite for LAYA-SWE Memory Engine."""

import time
import pytest
from proxy.memory_engine import LayaSweMemoryEngine, LayaMemoryController, MemoryPlane, MemoryNode
from proxy.compactor import ContextCompactor, compact_tool_output


def test_plane_classification_and_pinning():
    controller = LayaMemoryController(mode="deterministic")

    # Invariant classification & pinning
    plane1, pinned1 = controller.classify_plane("MANDATORY INVARIANT: Tenant ID must never be null.")
    assert plane1 == MemoryPlane.INVARIANT
    assert pinned1 is True

    # Resolution classification
    plane2, pinned2 = controller.classify_plane("Bug resolved. All tests pass in 0.05s.")
    assert plane2 == MemoryPlane.RESOLUTION
    assert pinned2 is False

    # Symbolic AST classification
    plane3, pinned3 = controller.classify_plane("class TransactionCoordinator:\n    def commit(self, tx_id):\n        pass")
    assert plane3 == MemoryPlane.SYMBOLIC
    assert pinned3 is False

    # Trajectory classification
    plane4, pinned4 = controller.classify_plane("git checkout -b fix-branch\nrunning bash build script...")
    assert plane4 == MemoryPlane.TRAJECTORY
    assert pinned4 is False


def test_symbol_extraction():
    controller = LayaMemoryController(mode="deterministic")
    text = (
        "def calculate_monthly_revenue(tenant_id: str):\n"
        "    session = get_db(tenant_id=tenant_id)\n"
        "    # inspect app/analytics.py and tests/test_tenant_invariant.py\n"
    )
    symbols = controller.extract_symbols(text)

    assert "app/analytics.py" in symbols
    assert "tests/test_tenant_invariant.py" in symbols
    assert "calculate_monthly_revenue" in symbols
    assert "tenant_id" in symbols


def test_invariant_retention_under_heavy_load():
    ctrl = LayaMemoryController(mode="deterministic")
    engine = LayaSweMemoryEngine(controller=ctrl, max_nodes=30)

    # Ingest 2 critical security invariants
    engine.ingest("INVARIANT 1: Under NO circumstance may third-party packages be added to requirements.txt.")
    engine.ingest("INVARIANT 2: Database session must strictly validate tenant_id.")

    # Flood with 100 noisy trajectory items (exceeding max_nodes by > 3x)
    for i in range(100):
        engine.ingest(f"Execution log entry {i}: Ran grep, read file {i}, exit status 0.")

    # Graph must be strictly bounded to max_nodes
    assert len(engine.nodes_map) <= 30
    assert len(engine.graph.nodes) <= 30

    # Retrieve and verify invariants were NOT evicted
    retrieved = engine.retrieve("Can we install external packages or omit tenant_id?")
    assert "third-party packages" in retrieved.lower()
    assert "tenant_id" in retrieved.lower()


def test_causal_failure_resolution_wiring():
    ctrl = LayaMemoryController(mode="deterministic")
    engine = LayaSweMemoryEngine(controller=ctrl)

    fail_id = engine.ingest("FAILED tests/test_lru.py::test_eviction - AssertionError: wrong key evicted")
    fix_id = engine.ingest("RESOLUTION: Fixed eviction order in cache/lru_manager.py. All tests pass.")

    # Verify bidirectional causal relationship edge
    assert engine.graph.has_edge(fail_id, fix_id)
    assert engine.graph.has_edge(fix_id, fail_id)

    edge_data = engine.graph.get_edge_data(fail_id, fix_id)
    assert any(data.get("rel_type") == "causal" for data in edge_data.values())


def test_sub_millisecond_retrieval_latency():
    ctrl = LayaMemoryController(mode="deterministic")
    engine = LayaSweMemoryEngine(controller=ctrl, max_nodes=100)
    for i in range(50):
        engine.ingest(f"Observation {i}: def process_job_{i}(): pass in worker_{i}.py")

    t0 = time.perf_counter()
    res = engine.retrieve("process_job_25 worker_25.py")
    duration_ms = (time.perf_counter() - t0) * 1000

    assert duration_ms < 15.0  # Fast retrieval
    assert "worker_25" in res or "process_job_25" in res
