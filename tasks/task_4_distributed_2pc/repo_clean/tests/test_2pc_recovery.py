"""Ground truth validation tests for Task 4: Distributed 2PC Coordinator."""
import os
import sys
import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(cur_dir)
for candidate in [parent_dir, os.path.join(parent_dir, "repo"), os.getcwd()]:
    if candidate not in sys.path and os.path.isdir(candidate):
        sys.path.insert(0, candidate)

from core.ledger import TransactionLedger
from core.participant import ParticipantNode
from core.coordinator import TransactionCoordinator


def test_successful_2pc_commit():
    """Verify standard happy-path 2PC commit across multiple nodes."""
    node1 = ParticipantNode("node_1")
    node2 = ParticipantNode("node_2")
    ledger = TransactionLedger()
    coordinator = TransactionCoordinator([node1, node2], ledger=ledger)

    ops = {
        "node_1": {"account_A": 100},
        "node_2": {"account_B": 200}
    }
    success = coordinator.execute_transaction("tx_001", ops)
    assert success is True, "Transaction should have committed"
    assert node1.get("account_A") == 100
    assert node2.get("account_B") == 200
    assert node1.is_locked("account_A") is False
    assert node2.is_locked("account_B") is False
    assert ledger.get_status("tx_001") == "COMMITTED"


def test_participant_abort_global_rollback():
    """Verify that a single participant voting abort causes global transaction failure."""
    node1 = ParticipantNode("node_1")
    node2 = ParticipantNode("node_2", should_fail_on_key="forbidden_key")
    ledger = TransactionLedger()
    coordinator = TransactionCoordinator([node1, node2], ledger=ledger)

    ops = {
        "node_1": {"account_A": 500},
        "node_2": {"forbidden_key": 999}
    }
    success = coordinator.execute_transaction("tx_002", ops)
    assert success is False, "Transaction should have aborted"
    assert node1.get("account_A") is None, "Node 1 should not have committed data"
    assert ledger.get_status("tx_002") == "ABORTED"


def test_no_lock_leaks_after_abort():
    """Verify that when a transaction aborts, already-prepared participants release their locks."""
    node1 = ParticipantNode("node_1")
    node2 = ParticipantNode("node_2", should_fail_on_key="forbidden_key")
    ledger = TransactionLedger()
    coordinator = TransactionCoordinator([node1, node2], ledger=ledger)

    # First tx fails on node2, but node1 was prepared first
    ops1 = {
        "node_1": {"shared_key": 100},
        "node_2": {"forbidden_key": 999}
    }
    success1 = coordinator.execute_transaction("tx_fail", ops1)
    assert success1 is False

    # Invariant: Node 1 MUST NOT leak locks from the aborted transaction!
    assert node1.is_locked("shared_key") is False, "Node 1 leaked lock on shared_key after aborted transaction!"

    # A subsequent transaction on node1 for shared_key MUST succeed without LockConflictError
    ops2 = {
        "node_1": {"shared_key": 250}
    }
    success2 = coordinator.execute_transaction("tx_retry", ops2)
    assert success2 is True, "Subsequent transaction failed due to lock leak from prior aborted tx!"
    assert node1.get("shared_key") == 250


def test_crash_recovery_resolves_in_doubt():
    """Verify that coordinator recovery resolves in-doubt transactions and cleans locks."""
    node1 = ParticipantNode("node_1")
    node2 = ParticipantNode("node_2")
    ledger = TransactionLedger()
    coordinator = TransactionCoordinator([node1, node2], ledger=ledger)

    # Simulate coordinator crash mid-flight: node1 prepared, node2 prepared, ledger recorded PREPARING
    tx_crash = "tx_crashed_midway"
    node1.prepare(tx_crash, {"k1": "v1"})
    node2.prepare(tx_crash, {"k2": "v2"})
    ledger.record(tx_crash, "PREPARING")

    assert node1.is_locked("k1") is True
    assert node2.is_locked("k2") is True

    # Crash recovery should resolve in-doubt PREPARING tx by aborting and releasing locks
    coordinator.recover_crashed_transactions()

    assert node1.is_locked("k1") is False, "Node 1 lock not cleared by crash recovery"
    assert node2.is_locked("k2") is False, "Node 2 lock not cleared by crash recovery"
    assert node1.get("k1") is None, "In-doubt transaction must not be committed"


def test_ledger_invariants():
    """Verify that ledger reflects accurate terminal statuses without ghost entries."""
    node1 = ParticipantNode("node_1")
    ledger = TransactionLedger()
    coordinator = TransactionCoordinator([node1], ledger=ledger)

    coordinator.execute_transaction("tx_ok", {"node_1": {"x": 1}})
    coordinator.execute_transaction("tx_err", {"node_1": {"x": 2}})

    all_txs = ledger.get_all_transactions()
    assert "tx_ok" in all_txs
    assert "tx_err" in all_txs
    assert all(status in ("COMMITTED", "ABORTED") for status in all_txs.values())
