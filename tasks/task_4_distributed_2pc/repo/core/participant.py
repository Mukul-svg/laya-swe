"""Distributed 2PC Participant Node with WAL and Record Locking."""
from typing import Dict, Any, List, Optional


class LockConflictError(Exception):
    """Raised when an operation cannot acquire a key lock."""
    pass


class ParticipantNode:
    """A participant database node in a distributed 2PC cluster."""

    def __init__(self, node_id: str, should_fail_on_key: Optional[str] = None):
        self.node_id = node_id
        self.should_fail_on_key = should_fail_on_key
        self.data: Dict[str, Any] = {}
        self.locks: Dict[str, str] = {}  # key -> tx_id
        self.pending_ops: Dict[str, Dict[str, Any]] = {}  # tx_id -> {key: value}
        self.wal: List[Dict[str, Any]] = []

    def prepare(self, tx_id: str, operations: Dict[str, Any]) -> bool:
        """Phase 1: Acquire locks and validate operations.
        
        Returns True (Vote COMMIT) if locks acquired and validation passes.
        Returns False (Vote ABORT) if validation fails.
        Raises LockConflictError if a lock cannot be acquired.
        """
        # Simulated node failure condition
        if self.should_fail_on_key and self.should_fail_on_key in operations:
            self.wal.append({"tx_id": tx_id, "action": "VOTE_ABORT", "reason": "simulated_failure"})
            return False

        # Attempt to acquire locks for all keys in this transaction
        for key in operations:
            holder = self.locks.get(key)
            if holder is not None and holder != tx_id:
                raise LockConflictError(f"Node {self.node_id}: Lock conflict on key '{key}', held by {holder}")

        # Lock keys and stage pending updates
        for key in operations:
            self.locks[key] = tx_id

        self.pending_ops[tx_id] = dict(operations)
        self.wal.append({"tx_id": tx_id, "action": "VOTE_COMMIT", "operations": operations})
        return True

    def commit(self, tx_id: str):
        """Phase 2a: Apply pending operations and release locks."""
        ops = self.pending_ops.pop(tx_id, {})
        for key, val in ops.items():
            self.data[key] = val

        # Release all locks held by this transaction
        held_keys = [k for k, v in self.locks.items() if v == tx_id]
        for k in held_keys:
            del self.locks[k]

        self.wal.append({"tx_id": tx_id, "action": "COMMITTED"})

    def abort(self, tx_id: str):
        """Phase 2b: Discard pending operations and release locks."""
        self.pending_ops.pop(tx_id, None)

        # Release all locks held by this transaction
        held_keys = [k for k, v in self.locks.items() if v == tx_id]
        for k in held_keys:
            del self.locks[k]

        self.wal.append({"tx_id": tx_id, "action": "ABORTED"})

    def is_locked(self, key: str) -> bool:
        """Check if key is currently locked."""
        return key in self.locks

    def get(self, key: str) -> Any:
        """Read committed value."""
        return self.data.get(key)
