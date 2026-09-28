# Task 4: Distributed Two-Phase Commit Coordinator with Crash Recovery

### Turn 1: Discovery & Investigation
Inspect `core/coordinator.py`, `core/participant.py`, and `core/ledger.py`.
Notice how the coordinator executes distributed transactions across participant nodes using the 2PC protocol.
Run `pytest tests/test_2pc_recovery.py` to see the failing tests.

### Turn 2: Bug Resolution & Invariant Enforcement
1. In `core/coordinator.py`, fix the partial rollback leak: if ANY participant fails or votes abort during the prepare phase, the coordinator MUST immediately abort all participants that already prepared, releasing their key locks!
2. In `core/coordinator.py`, implement `recover_crashed_transactions()`: inspect in-flight transactions recorded in `self.ledger`. If a transaction is `COMMITTED`, commit participants; if `ABORTED` or in-doubt (`PREPARING`), ensure all participants abort and release locks.
3. Run `pytest tests/test_2pc_recovery.py` to verify all 5 tests pass.
