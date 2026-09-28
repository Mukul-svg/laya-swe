"""Verification tests for the Proxy, Context Compactor, and FastAPI Endpoints."""
import pytest
from starlette.testclient import TestClient
from proxy.memory_engine import LayaSweMemoryEngine, LayaMemoryController
from proxy.compactor import ContextCompactor
from proxy.server import app

def test_memory_engine_ingest_and_retrieve():
    ctrl = LayaMemoryController(mode="deterministic")
    engine = LayaSweMemoryEngine(controller=ctrl)
    engine.ingest("We mandate that tenant_id must be provided on every database query.")
    engine.ingest("Never add new packages to requirements.txt.")
    engine.ingest("LRUCache should evict oldest items from the front.")
    
    assert len(engine.nodes_map) == 3
    assert len(engine.graph.nodes) == 3
    
    # Query for database rules
    evidence = engine.retrieve("How should database queries handle tenant_id?")
    assert "tenant_id" in evidence.lower()
    
    # Query for dependency constraints
    evidence_dep = engine.retrieve("Can we install third-party libraries in requirements.txt?")
    assert "requirements.txt" in evidence_dep.lower()

def test_compactor_preserves_tool_pairs():
    ctrl = LayaMemoryController(mode="deterministic")
    engine = LayaSweMemoryEngine(controller=ctrl)
    compactor = ContextCompactor(memory_engine=engine, max_tail_messages=3)
    
    messages = [
        {"role": "system", "content": "You are an expert coding assistant tasked with codebase maintenance and multi-tenant security enforcement."},
        {"role": "user", "content": "Step 1: Check the codebase architecture and note tenant rules. Ensure that every single database query passes tenant_id explicitly as a security constraint."},
        {"role": "assistant", "content": "I have thoroughly analyzed the architecture guidelines and will strictly ensure tenant_id scoping across all subsequent modules."},
        {"role": "user", "content": "Step 2: Inspect db module and explain all classes, methods, and connection pool semantics in detail."},
        {"role": "assistant", "content": "Here is the detailed summary of the connection pooling and transaction lifecycle across all tenant databases.\n" + ("Database connection pool initialized with 20 connections.\n" * 40)},
        {"role": "user", "content": "Step 3: Can you also check how logging is configured in app/logging.py?"},
        {"role": "assistant", "content": "Logging is set up with JSON formatting and logs tenant context in every log record."},
        {"role": "user", "content": "Step 4: Review the configuration settings in config/settings.yaml."},
        {"role": "assistant", "content": "Settings loaded: database pool size 20, max overflow 10, timeout 30s."},
        {"role": "user", "content": "Step 5: Inspect db module..."},
        {"role": "assistant", "content": "Inspecting db module...", "tool_calls": [{"id": "call_123", "function": {"name": "read", "arguments": "{\"path\": \"app/db.py\"}"}}]},
        {"role": "tool", "tool_call_id": "call_123", "content": "class DatabaseSession:\n    def __init__(self, tenant_id: str = None):\n        self.tenant_id = tenant_id\n    def execute_query(self, query: str):\n        if not self.tenant_id:\n            raise ValueError('SECURITY VIOLATION: Database session requires an explicit tenant_id!')\n        return [{'tenant_id': self.tenant_id, 'result': 'ok'}]"},
        {"role": "assistant", "content": "The DatabaseSession enforces that tenant_id must be provided or it raises a ValueError. Now ready for the next task."},
        {"role": "user", "content": "Step 6: Now implement the analytics query endpoint."}
    ]
    
    compacted, raw_tokens, comp_tokens = compactor.process_and_compact(messages, enable_compaction=True)
    
    # 1. System prompt is preserved
    assert compacted[0]["role"] == "system"
    # 2. System-1 Memory block is injected
    assert any("SYSTEM-1 MEMORY" in m.get("content", "") for m in compacted)
    # 3. Tokens are reduced
    assert comp_tokens < raw_tokens
    
    # 4. Check tool call pairing: tool message must have its assistant tool_calls message immediately preceding it
    roles = [m["role"] for m in compacted]
    if "tool" in roles:
        tool_idx = roles.index("tool")
        assert tool_idx > 0
        assert compacted[tool_idx - 1]["role"] == "assistant"
        assert "tool_calls" in compacted[tool_idx - 1]

def test_proxy_server_routes():
    client = TestClient(app)

    # 1. Health endpoint
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"

    # 2. Benchmark configure endpoint
    resp = client.post("/benchmark/configure", json={"compaction": False, "task_id": "audit_task"})
    assert resp.status_code == 200
    assert resp.json()["config"]["compaction"] is False
    assert resp.json()["config"]["task_id"] == "audit_task"

    # 3. Reset endpoint
    resp = client.post("/reset")
    assert resp.status_code == 200
    assert resp.json()["status"] == "memory_cleared"

    # 4. Ingest endpoint (used by native extensions & hooks)
    resp = client.post("/ingest", json={"observation": "Rule: Always validate session tenant.", "metadata": {"tool": "test"}})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "node_id" in resp.json()

    # 5. Query endpoint (used by native extensions & hooks)
    resp = client.post("/query", json={"query": "tenant session rule", "top_k": 3})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "tenant" in resp.json()["evidence"].lower()

    # 6. Metrics endpoint
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "total_requests" in resp.json()


def test_compactor_edge_cases():
    from proxy.compactor import compact_tool_output
    # 1. Empty and None content
    assert compact_tool_output("") == ""
    assert compact_tool_output(None) == ""

    # 2. Non-string types (dict, list, int)
    dict_content = {"status": "error", "code": 500, "details": "stack trace line\n" * 20}
    res_dict = compact_tool_output(dict_content, max_chars=100)
    assert isinstance(res_dict, str)
    assert len(res_dict) <= 110

    # 3. Very small max_chars boundaries
    for mc in [10, 30, 50, 75, 120]:
        long_str = "x" * 500
        compacted = compact_tool_output(long_str, max_chars=mc)
        assert len(compacted) <= mc + 5, f"Exceeded max_chars {mc}: got {len(compacted)}"

    # 4. Unicode and multi-byte characters
    unicode_str = "🚀 统一码测试 éàç üöß " * 40
    res_unicode = compact_tool_output(unicode_str, max_chars=150)
    assert len(res_unicode) <= 160
    assert "omitted" in res_unicode


def test_memory_engine_lru_capacity_and_constraint_preservation():
    # Test bounded capacity of 20 nodes
    ctrl = LayaMemoryController(mode="deterministic")
    engine = LayaSweMemoryEngine(controller=ctrl, max_nodes=20)

    # Ingest critical security invariant first
    engine.ingest("MANDATORY CONSTRAINT: Tenant ID must never be null under any circumstances.")

    # Ingest 40 generic log messages
    for i in range(40):
        engine.ingest(f"Generic temporary execution log message number {i}")

    # Verify capacity strictly bounded
    assert len(engine.nodes_map) <= 20
    assert len(engine.graph.nodes) <= 20

    # Verify that the critical constraint node was NOT evicted
    retrieved = engine.retrieve("tenant id invariant rules")
    assert "tenant id" in retrieved.lower() or "mandatory constraint" in retrieved.lower()


def test_proxy_concurrent_stress():
    from concurrent.futures import ThreadPoolExecutor
    client = TestClient(app)

    def send_req(i):
        # Interleave configure and metrics requests concurrently
        if i % 2 == 0:
            return client.post("/benchmark/configure", json={"compaction": True, "task_id": f"concurrent_{i}"})
        else:
            return client.get("/metrics")

    with ThreadPoolExecutor(max_workers=10) as executor:
        responses = list(executor.map(send_req, range(20)))

    for resp in responses:
        assert resp.status_code == 200
