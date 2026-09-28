import time
import httpx

cases = [
    ("CRITICAL INVARIANT: DatabaseSession requires tenant_id. All queries missing tenant_id will raise SecurityError.", "Invariant"),
    ("FAILED tests/test_lru_race.py::test_lru_concurrent_access - KeyError: key_42 not found", "Failure Log"),
    ("class Coordinator:\n    def __init__(self, participants):\n        self.participants = participants\n    def execute_transaction(self, tx_id):\n        return True", "Source Code"),
    ("pytest tests/test_tenant.py ... 5 passed in 0.12s (100% passing)", "Resolution Fix")
]

print("=" * 60)
print("TESTING LIVE LAYA CLASSIFICATION THROUGH PROXY")
print("=" * 60)

for text, label in cases:
    t0 = time.time()
    r = httpx.post("http://127.0.0.1:8080/debug/classify", json={"text": text}, timeout=45.0)
    data = r.json()
    plane = data["plane"]
    pinned = data["is_pinned"]
    choice = data["laya"].get("choice")
    conf = data["laya"].get("confidence")
    probs = data["laya"].get("probabilities", {})
    dur = time.time() - t0
    print(f"\nLabel: {label}")
    print(f"  Plane:      {plane} (Pinned: {pinned})")
    print(f"  LAYA Choice:{choice} (Conf: {conf})")
    print(f"  LAYA Probs: {probs}")
    print(f"  Latency:    {dur:.3f}s")
